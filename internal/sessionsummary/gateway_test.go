package sessionsummary

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/v2"
)

func TestGatewayClientMapCallContract(t *testing.T) {
	t.Parallel()

	type capturedRequest struct {
		Path   string
		Header http.Header
		Body   map[string]any
	}
	captured := make(chan capturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		captured <- capturedRequest{Path: r.URL.Path, Header: r.Header.Clone(), Body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_summary","type":"message","role":"assistant","model":"claude-haiku-4-5:main","stop_reason":"tool_use",
			"content":[{"type":"tool_use","id":"tool_1","name":"record_chunk_summary","input":{"evidence":["a","b","c"],"summary":"mapped"}}],
			"usage":{"input_tokens":1,"cache_creation_input_tokens":20,"cache_read_input_tokens":80,"output_tokens":23}
		}`))
	}))
	defer server.Close()

	client, err := NewGatewayClient(GatewayClientConfig{
		BaseURL: server.URL + "/v1", APIKey: "gateway-test-token",
	})
	if err != nil {
		t.Fatal(err)
	}
	got, usage, err := MapCall(t.Context(), client, EmbeddedPromptBundle,
		CallOptions{ModelID: "claude-haiku-4-5:main", MaxTokens: 1024},
		[]byte(`{"role":"user","content":"hello"}`), 1, 1)
	if err != nil {
		t.Fatalf("MapCall: %v", err)
	}
	if got.Summary != "mapped" || len(got.Evidence) != 3 {
		t.Fatalf("unexpected summary: %+v", got)
	}
	if usage.InputTokens != 101 || usage.OutputTokens != 23 {
		t.Fatalf("unexpected usage: %+v", usage)
	}

	req := <-captured
	if req.Path != "/v1/messages" {
		t.Fatalf("path = %q", req.Path)
	}
	if req.Header.Get("Authorization") != "Bearer gateway-test-token" {
		t.Fatalf("Authorization = %q", req.Header.Get("Authorization"))
	}
	if req.Header.Get("anthropic-version") != "2023-06-01" || req.Header.Get("X-Api-Key") != "" {
		t.Fatalf("unexpected Anthropic headers: %v", req.Header)
	}
	if req.Header.Get("x-bedrock-metadata") != "" {
		t.Fatalf("x-bedrock-metadata = %q, want absent", req.Header.Get("x-bedrock-metadata"))
	}
	if req.Body["model"] != "claude-haiku-4-5:main" {
		t.Fatalf("model = %v", req.Body["model"])
	}
	for _, field := range []string{"input", "instructions", "store", "parallel_tool_calls", "max_output_tokens", "prompt_cache_retention"} {
		if _, ok := req.Body[field]; ok {
			t.Fatalf("Responses-only field %q must be absent", field)
		}
	}
	if req.Body["max_tokens"] != float64(1024) {
		t.Fatalf("max_tokens = %v", req.Body["max_tokens"])
	}
	system := req.Body["system"].([]any)[0].(map[string]any)
	if system["type"] != "text" || system["text"] != EmbeddedPromptBundle.MapSystem {
		t.Fatalf("system = %#v", system)
	}
	cacheControl := system["cache_control"].(map[string]any)
	if cacheControl["type"] != "ephemeral" || cacheControl["ttl"] != nil {
		t.Fatalf("want default five-minute prompt cache: %#v", cacheControl)
	}
	message := req.Body["messages"].([]any)[0].(map[string]any)
	content := message["content"].([]any)[0].(map[string]any)
	if message["role"] != "user" || content["type"] != "text" || !strings.Contains(content["text"].(string), `<chunk index="1" of="1">`) {
		t.Fatalf("messages = %#v", req.Body["messages"])
	}
	tools, ok := req.Body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v", req.Body["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["name"] != "record_chunk_summary" {
		t.Fatalf("tool = %#v", tool)
	}
	if _, ok := tool["strict"]; ok {
		t.Fatalf("strict must be omitted to preserve non-strict tool semantics")
	}
	parameters := tool["input_schema"].(map[string]any)
	properties := parameters["properties"].(map[string]any)
	evidence := properties["evidence"].(map[string]any)
	if evidence["minItems"] != float64(3) || evidence["maxItems"] != float64(5) {
		t.Fatalf("evidence schema constraints must be preserved: %#v", evidence)
	}
	choice := req.Body["tool_choice"].(map[string]any)
	if choice["type"] != "tool" || choice["name"] != "record_chunk_summary" || choice["disable_parallel_tool_use"] != true {
		t.Fatalf("tool_choice = %#v", choice)
	}
}

func TestGatewayClientRejectsMaxTokens(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"type":"message","role":"assistant","stop_reason":"max_tokens",
			"content":[],"usage":{"input_tokens":10,"output_tokens":4096}
		}`))
	}))
	defer server.Close()
	client, err := NewGatewayClient(GatewayClientConfig{BaseURL: server.URL, APIKey: "test"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = ReduceCall(t.Context(), client, EmbeddedPromptBundle,
		CallOptions{ModelID: "claude-sonnet-5:main", MaxTokens: 4096},
		[]ChunkSummary{{Summary: "chunk"}})
	if err == nil || !strings.Contains(err.Error(), "max_tokens") {
		t.Fatalf("want max_tokens error, got %v", err)
	}
}

func TestGatewayClientRejectsRedirectWithoutForwardingBearer(t *testing.T) {
	t.Parallel()

	var targetAuthorization string
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		targetAuthorization = r.Header.Get("Authorization")
		http.Error(w, "must not arrive", http.StatusInternalServerError)
	}))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()

	client, err := NewGatewayClient(GatewayClientConfig{BaseURL: redirect.URL, APIKey: "secret"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = MapCall(context.Background(), client, EmbeddedPromptBundle,
		CallOptions{ModelID: "claude-haiku-4-5:main", MaxTokens: 1024}, []byte(`{}`), 1, 1)
	if err == nil || !strings.Contains(err.Error(), "status 307") {
		t.Fatalf("want redirect status error, got %v", err)
	}
	if targetAuthorization != "" {
		t.Fatalf("gateway bearer reached redirect target: %q", targetAuthorization)
	}
}

func TestNewGatewayClientValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  GatewayClientConfig
		want string
	}{
		{name: "missing URL", cfg: GatewayClientConfig{APIKey: "key"}, want: "base URL is required"},
		{name: "bad scheme", cfg: GatewayClientConfig{BaseURL: "ftp://gateway", APIKey: "key"}, want: "must use http or https"},
		{name: "userinfo", cfg: GatewayClientConfig{BaseURL: "https://user:password@gateway/v1", APIKey: "key"}, want: "must not contain userinfo"},
		{name: "query", cfg: GatewayClientConfig{BaseURL: "https://gateway/v1?token=bad", APIKey: "key"}, want: "query or fragment"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewGatewayClient(tc.cfg)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
			if strings.Contains(err.Error(), "password") {
				t.Fatalf("error leaked URL credentials: %v", err)
			}
		})
	}
}

func TestGatewayClientMessagesPath(t *testing.T) {
	t.Parallel()
	for _, prefix := range []string{"", "/", "/v1", "/v1/", "/proxy", "/proxy/v1/"} {
		t.Run(prefix, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				want := "/v1/messages"
				if strings.HasPrefix(prefix, "/proxy") {
					want = "/proxy" + want
				}
				if r.Method != http.MethodPost || r.URL.Path != want {
					t.Errorf("got %s %s, want POST %s", r.Method, r.URL.Path, want)
				}
				_, _ = w.Write([]byte(`{}`))
			}))
			defer server.Close()
			client, err := NewGatewayClient(GatewayClientConfig{BaseURL: server.URL + prefix, APIKey: "test"})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := client.CreateMessage(t.Context(), anthropic.MessageNewParams{}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGatewayClientReduceCallContract(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var params anthropic.MessageNewParams
		if err := json.NewDecoder(r.Body).Decode(&params); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if params.Model != "claude-sonnet-5:main" || params.MaxTokens != 4096 || params.System[0].Text != EmbeddedPromptBundle.ReduceSystem {
			t.Errorf("incorrect reduce model, limit, or prompt: %+v", params)
		}
		if params.ToolChoice.OfTool == nil || params.ToolChoice.OfTool.Name != "record_session_summary" || !params.ToolChoice.OfTool.DisableParallelToolUse.Value {
			t.Errorf("unexpected reduce tool choice: %+v", params.ToolChoice)
		}
		if len(params.Tools) != 1 || params.Tools[0].OfTool == nil || params.Tools[0].OfTool.Name != "record_session_summary" {
			t.Errorf("unexpected reduce tools: %+v", params.Tools)
			return
		}
		gotSchema, _ := json.Marshal(params.Tools[0].OfTool.InputSchema)
		wantSchema, _ := json.Marshal(sessionSummaryToolSchema(EmbeddedPromptBundle))
		if string(gotSchema) != string(wantSchema) {
			t.Errorf("reduce schema = %s, want %s", gotSchema, wantSchema)
		}
		if !strings.Contains(params.Messages[0].Content[0].OfText.Text, "mapped chunk") {
			t.Errorf("reduce input lost chunk summaries")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"stop_reason":"tool_use","content":[{"type":"tool_use","name":"record_session_summary","input":{"narrative":"final narrative","embedding_text":"keywords","context_hint":"repo: task","evidence":["a"],"tags":{"projects":["diderot"]}}}],"usage":{"input_tokens":10,"output_tokens":20}}`))
	}))
	defer server.Close()
	client, err := NewGatewayClient(GatewayClientConfig{BaseURL: server.URL + "/v1", APIKey: "test"})
	if err != nil {
		t.Fatal(err)
	}
	result, usage, err := ReduceCall(t.Context(), client, EmbeddedPromptBundle,
		CallOptions{ModelID: "claude-sonnet-5:main", MaxTokens: 4096}, []ChunkSummary{{Summary: "mapped chunk"}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Narrative != "final narrative" || len(result.Tags.Projects) != 1 || usage.InputTokens != 10 || usage.OutputTokens != 20 {
		t.Fatalf("unexpected reduce result: %+v, usage: %+v", result, usage)
	}
}

func TestGatewayClientRejectsInvalidResponses(t *testing.T) {
	t.Parallel()
	for _, scenario := range []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{name: "invalid JSON", status: http.StatusOK, body: "not JSON", want: "decode response"},
		{name: "old Responses format", status: http.StatusOK, body: `{"status":"completed","output":[]}`, want: "stop reason"},
		{name: "rate limited", status: http.StatusTooManyRequests, body: `{"error":"slow down"}`, want: "status 429"},
		{name: "oversized response", status: http.StatusOK, body: strings.Repeat(" ", int(gatewaySummaryMaxResponseBytes)+1), want: "exceeds"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			var calls int
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				w.WriteHeader(scenario.status)
				_, _ = w.Write([]byte(scenario.body))
			}))
			defer server.Close()
			client, err := NewGatewayClient(GatewayClientConfig{BaseURL: server.URL, APIKey: "test"})
			if err != nil {
				t.Fatal(err)
			}
			_, _, err = MapCall(t.Context(), client, EmbeddedPromptBundle, CallOptions{ModelID: "claude-haiku-4-5:main", MaxTokens: 1024}, []byte(`{}`), 1, 1)
			if err == nil || !strings.Contains(err.Error(), scenario.want) {
				t.Fatalf("want %q error, got %v", scenario.want, err)
			}
			if calls != 1 {
				t.Fatalf("client retried %d requests; retries belong to the worker", calls)
			}
		})
	}
}
