package sessionsummary

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/anthropics/anthropic-sdk-go/v2"
	"strings"
	"testing"
	"time"
)

type deadlineMessageFunc func(context.Context) (*anthropic.Message, error)

func (client deadlineMessageFunc) CreateMessage(ctx context.Context, _ anthropic.MessageNewParams) (*anthropic.Message, error) {
	return client(ctx)
}

func TestInvokeMessageBoundsEachRequest(test *testing.T) {
	var requestContexts []context.Context
	client := deadlineMessageFunc(func(ctx context.Context) (*anthropic.Message, error) {
		deadline, ok := ctx.Deadline()
		if !ok || time.Until(deadline) > 3*time.Minute || time.Until(deadline) < 2*time.Minute {
			test.Fatalf("model request deadline missing or wrong: %s", deadline)
		}
		requestContexts = append(requestContexts, ctx)
		return toolUseOutput("test_tool", `{}`, anthropic.StopReasonToolUse), nil
	})
	for range 2 {
		if _, _, err := invokeTool(test.Context(), OverMessages(client), ToolRequest{System: "system", User: "user", ToolName: "test_tool", ToolDescription: "test"}); err != nil {
			test.Fatal(err)
		}
	}
	if requestContexts[0] == requestContexts[1] {
		test.Fatal("separate model calls shared a deadline context")
	}
	for _, ctx := range requestContexts {
		if !errors.Is(ctx.Err(), context.Canceled) {
			test.Fatal("completed request retained its timer")
		}
	}
}

func TestInvokeMessageHonorsCallerDeadline(test *testing.T) {
	ctx, cancel := context.WithTimeout(test.Context(), 25*time.Millisecond)
	defer cancel()
	client := deadlineMessageFunc(func(requestCtx context.Context) (*anthropic.Message, error) {
		deadline, _ := requestCtx.Deadline()
		parentDeadline, _ := ctx.Deadline()
		if !deadline.Equal(parentDeadline) {
			test.Fatal("model call extended the caller deadline")
		}
		<-requestCtx.Done()
		return nil, requestCtx.Err()
	})
	_, _, err := invokeTool(ctx, OverMessages(client), ToolRequest{System: "system", User: "user", ToolName: "test_tool", ToolDescription: "test"})
	if !errors.Is(err, context.DeadlineExceeded) {
		test.Fatalf("stalled call did not return its deadline error: %v", err)
	}
}

// The tool schema forces evidence to be an array, but models
// occasionally emit a bare string anyway, and a rigid decode burns the
// whole retry budget on it. StringList tolerates both shapes.
func TestStringList_ToleratesBareString(t *testing.T) {
	var fs FinalSummary
	blob := `{"evidence":"just one thing","narrative":"n","embedding_text":"e","context_hint":"c","tags":{}}`
	if err := json.Unmarshal([]byte(blob), &fs); err != nil {
		t.Fatalf("bare-string evidence must decode: %v", err)
	}
	if len(fs.Evidence) != 1 || fs.Evidence[0] != "just one thing" {
		t.Fatalf("want wrapped single element, got %v", fs.Evidence)
	}

	var cs ChunkSummary
	blob = `{"evidence":["a","b"],"summary":"s"}`
	if err := json.Unmarshal([]byte(blob), &cs); err != nil {
		t.Fatalf("array evidence must decode: %v", err)
	}
	if len(cs.Evidence) != 2 {
		t.Fatalf("want 2 elements, got %v", cs.Evidence)
	}

	// Garbage still errors.
	if err := json.Unmarshal([]byte(`{"evidence":42,"summary":"s"}`), &cs); err == nil {
		t.Fatal("numeric evidence must fail decode")
	}
}

type stubMessage struct {
	out *anthropic.Message
}

func (client *stubMessage) CreateMessage(context.Context, anthropic.MessageNewParams) (*anthropic.Message, error) {
	return client.out, nil
}

func toolUseOutput(toolName string, inputJSON string, stop anthropic.StopReason) *anthropic.Message {
	return &anthropic.Message{
		StopReason: stop,
		Content: []anthropic.ContentBlockUnion{{
			Type: "tool_use", Name: toolName, Input: json.RawMessage(inputJSON),
		}},
	}
}

// A reduce response truncated at max_tokens decodes as an empty tool
// document, and persisting it stores an empty narrative as a
// success. Both defenses must fire: the
// stop-reason check and the empty-narrative check.
func TestReduceCall_RejectsDegenerateOutput(t *testing.T) {
	bundle := EmbeddedPromptBundle
	opts := CallOptions{ModelID: "test-model", MaxTokens: 2048}
	inputs := []ChunkSummary{{Summary: "chunk one summary"}}

	t.Run("truncated at max_tokens", func(t *testing.T) {
		client := &stubMessage{out: toolUseOutput("record_session_summary",
			`{}`, anthropic.StopReasonMaxTokens)}
		_, _, err := ReduceCall(t.Context(), OverMessages(client), bundle, opts, inputs)
		if err == nil || !strings.Contains(err.Error(), "max_tokens") {
			t.Fatalf("want max_tokens truncation error, got %v", err)
		}
	})

	t.Run("empty narrative", func(t *testing.T) {
		client := &stubMessage{out: toolUseOutput("record_session_summary",
			`{"evidence":["a","b","c"],"narrative":"","embedding_text":"keywords","context_hint":"hint","tags":{}}`,
			anthropic.StopReasonToolUse)}
		_, _, err := ReduceCall(t.Context(), OverMessages(client), bundle, opts, inputs)
		if err == nil || !strings.Contains(err.Error(), "empty summary") {
			t.Fatalf("want empty-summary error, got %v", err)
		}
	})

	t.Run("whitespace narrative", func(t *testing.T) {
		client := &stubMessage{out: toolUseOutput("record_session_summary",
			`{"evidence":["a","b","c"],"narrative":"  \n\t","embedding_text":"keywords","context_hint":"hint","tags":{}}`,
			anthropic.StopReasonToolUse)}
		_, _, err := ReduceCall(t.Context(), OverMessages(client), bundle, opts, inputs)
		if err == nil || !strings.Contains(err.Error(), "empty summary") {
			t.Fatalf("want whitespace-summary error, got %v", err)
		}
	})

	t.Run("valid output passes", func(t *testing.T) {
		client := &stubMessage{out: toolUseOutput("record_session_summary",
			`{"evidence":["a","b","c"],"narrative":"a real narrative","embedding_text":"keywords","context_hint":"hint","tags":{}}`,
			anthropic.StopReasonToolUse)}
		fs, _, err := ReduceCall(t.Context(), OverMessages(client), bundle, opts, inputs)
		if err != nil {
			t.Fatalf("valid output must pass: %v", err)
		}
		if fs.Narrative != "a real narrative" {
			t.Fatalf("narrative round-trip broken: %q", fs.Narrative)
		}
	})

	t.Run("narrative without embedding text passes", func(t *testing.T) {
		client := &stubMessage{out: toolUseOutput("record_session_summary",
			`{"evidence":["a","b","c"],"narrative":"a usable narrative","embedding_text":"","context_hint":"hint","tags":{}}`,
			anthropic.StopReasonToolUse)}
		fs, _, err := ReduceCall(t.Context(), OverMessages(client), bundle, opts, inputs)
		if err != nil {
			t.Fatalf("narrative fallback must pass: %v", err)
		}
		if fs.Narrative != "a usable narrative" || fs.EmbeddingText != "" {
			t.Fatalf("unexpected fallback output: narrative=%q embedding=%q", fs.Narrative, fs.EmbeddingText)
		}
	})
}

func TestInvokeMessageValidatesToolOutput(t *testing.T) {
	t.Parallel()
	for _, scenario := range []struct {
		name    string
		message *anthropic.Message
		want    string
	}{
		{name: "missing response", want: "empty response"},
		{name: "truncated", message: toolUseOutput("test_tool", `{}`, anthropic.StopReasonMaxTokens), want: "max_tokens"},
		{name: "end turn", message: toolUseOutput("test_tool", `{}`, anthropic.StopReasonEndTurn), want: "stop reason"},
		{name: "wrong tool", message: toolUseOutput("other_tool", `{}`, anthropic.StopReasonToolUse), want: "unexpected tool_use"},
		{name: "null", message: toolUseOutput("test_tool", `null`, anthropic.StopReasonToolUse), want: "JSON object"},
		{name: "array", message: toolUseOutput("test_tool", `[]`, anthropic.StopReasonToolUse), want: "JSON object"},
		{name: "string", message: toolUseOutput("test_tool", `"input"`, anthropic.StopReasonToolUse), want: "JSON object"},
		{name: "missing input", message: toolUseOutput("test_tool", ``, anthropic.StopReasonToolUse), want: "JSON object"},
		{name: "missing tool", message: &anthropic.Message{StopReason: anthropic.StopReasonToolUse}, want: "no tool_use"},
		{name: "duplicate tools", message: &anthropic.Message{
			StopReason: anthropic.StopReasonToolUse,
			Content: []anthropic.ContentBlockUnion{
				{Type: "tool_use", Name: "test_tool", Input: json.RawMessage(`{}`)},
				{Type: "tool_use", Name: "test_tool", Input: json.RawMessage(`{}`)},
			},
		}, want: "multiple tool_use"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Parallel()
			client := &stubMessage{out: scenario.message}
			if scenario.message != nil {
				scenario.message.Usage = anthropic.Usage{InputTokens: 2, CacheReadInputTokens: 3, CacheCreationInputTokens: 5, OutputTokens: 7}
			}
			_, usage, err := invokeTool(t.Context(), OverMessages(client), ToolRequest{System: "system", User: "user", ToolName: "test_tool", ToolDescription: "test"})
			if err == nil || !strings.Contains(err.Error(), scenario.want) {
				t.Fatalf("want %q error, got %v", scenario.want, err)
			}
			if scenario.message != nil && (usage.InputTokens != 10 || usage.OutputTokens != 7) {
				t.Fatalf("failed request lost usage: %+v", usage)
			}
		})
	}
}
