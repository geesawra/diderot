package sessionsummary

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go/v2"

	"github.com/bluesky-social/diderot/internal/events"
)

// MessageClient sends provider-native Messages requests through the model gateway.
type MessageClient interface {
	CreateMessage(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error)
}

// ToolRequest is what a map or reduce stage asks a model: instructions, one
// user message, and the one tool the model must answer by calling.
type ToolRequest struct {
	ModelID   string
	MaxTokens int64
	System    string
	User      string
	// ToolName, ToolDescription and ToolSchema are the tool: its arguments
	// are the stage's answer.
	ToolName        string
	ToolDescription string
	ToolSchema      map[string]any
}

// ToolClient sends a ToolRequest on the one wire protocol it speaks, forcing
// the tool call by that protocol's own means, and returns the arguments of
// the model's one call to the tool, as the model wrote them. An answer
// without exactly that call is an error: a client never reads an answer out
// of prose. What the arguments must hold is checked once, by the stages, for
// every protocol alike.
type ToolClient interface {
	CallTool(ctx context.Context, request ToolRequest) (json.RawMessage, Usage, error)
}

// ChunkSummary is what the map step returns for a single chunk.
type ChunkSummary struct {
	Evidence StringList `json:"evidence"`
	Summary  string     `json:"summary"`
	// Label is the entry's chunk index in the reduce input: "4", or "1-3"
	// for a session summary standing in for chunks 1 to 3. Empty means its
	// position in the input.
	Label string `json:"-"`
}

// StringList is a []string that tolerates a model emitting a single
// JSON string where the schema asked for an array. The tool schema
// forces the array shape, but models occasionally emit
// "evidence": "one thing" on certain transcripts anyway, and do so on
// every retry. Silently wrapping beats burning the retry budget on a
// formatting quirk.
type StringList []string

func (s *StringList) UnmarshalJSON(data []byte) error {
	var arr []string
	if err := json.Unmarshal(data, &arr); err == nil {
		*s = arr
		return nil
	}
	var one string
	if err := json.Unmarshal(data, &one); err == nil {
		if one == "" {
			*s = nil
		} else {
			*s = []string{one}
		}
		return nil
	}
	return fmt.Errorf("evidence: want array of strings or string, got %s", string(data[:min(len(data), 40)]))
}

// FinalSummary is the reduce step output — the payload diderot
// ultimately ingests.
//
// Narrative is the human-facing prose. EmbeddingText and
// ContextHint exist to feed the embedding model (a last-token-pooling
// one rewards this) something better than a paragraph of English:
// dense keyword text for the chunk body and a short subject line for
// the contextual prefix.
type FinalSummary struct {
	Evidence      StringList `json:"evidence"`
	Narrative     string     `json:"narrative"`
	EmbeddingText string     `json:"embedding_text"`
	ContextHint   string     `json:"context_hint"`
	Tags          TagSet     `json:"tags"`
}

// TagSet is the typed tag slots. Flattened at DB-write time into
// the session_summaries.tags text[] with prefixes so diderot can
// filter by prefix (lang:go, activity:debug, project:diderot).
type TagSet struct {
	Projects   []string `json:"projects,omitempty"`
	Languages  []string `json:"languages,omitempty"`
	Activities []string `json:"activities,omitempty"`
	Freeform   []string `json:"freeform,omitempty"`
}

// Flatten turns a TagSet into the flat prefixed list stored in the DB.
// `diderot-ingest` style downstream can unflatten by prefix.
func (t TagSet) Flatten() []string {
	out := make([]string, 0,
		len(t.Projects)+len(t.Languages)+len(t.Activities)+len(t.Freeform))
	for _, p := range t.Projects {
		out = append(out, "project:"+p)
	}
	for _, l := range t.Languages {
		out = append(out, "lang:"+l)
	}
	for _, a := range t.Activities {
		out = append(out, "activity:"+a)
	}
	// Freeform tags carry no prefix — diderot filter semantics
	// expect bare tokens for uncategorized tags.
	out = append(out, t.Freeform...)
	return out
}

// --- Tool schemas -------------------------------------------------------

// chunkSummaryToolSchema is the JSON Schema (subset) for the map step.
func chunkSummaryToolSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"evidence": map[string]any{
				"type":        "array",
				"description": "3 to 5 verbatim strings from the chunk: file paths, commands, error messages, user directives. Do not paraphrase.",
				"items":       map[string]any{"type": "string"},
				"minItems":    3,
				"maxItems":    5,
			},
			"summary": map[string]any{
				"type":        "string",
				"description": "A 600–900 character summary of this chunk only. Lead with what the user was trying to do; name at least 2 concrete proper nouns; note decisions or pivots.",
			},
		},
		"required": []string{"evidence", "summary"},
	}
}

// sessionSummaryToolSchema is the reduce step's output shape. Enums
// are inlined from the prompt bundle so the schema and the prompt
// text stay in sync.
func sessionSummaryToolSchema(bundle PromptBundle) map[string]any {
	langEnum := make([]any, len(bundle.LanguageEnum))
	for i, v := range bundle.LanguageEnum {
		langEnum[i] = v
	}
	actEnum := make([]any, len(bundle.ActivityEnum))
	for i, v := range bundle.ActivityEnum {
		actEnum[i] = v
	}
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"evidence": map[string]any{
				"type":        "array",
				"description": "3 to 5 verbatim strings that recur across chunks and ground the narrative.",
				"items":       map[string]any{"type": "string"},
				"minItems":    3,
				"maxItems":    5,
			},
			"narrative": map[string]any{
				"type":        "string",
				"description": "A 900–1800 character narrative of the whole session: goal, pivots, outcome. Names concrete proper nouns. Weights early and mid session equally with the tail.",
			},
			"embedding_text": map[string]any{
				"type":        "string",
				"description": "A 250–500 character keyword-dense version of the summary, optimized for last-token-pooling embedding. Front-load proper nouns; drop filler. End with a 'Topic:' line listing the 3–5 most important nouns.",
			},
			"context_hint": map[string]any{
				"type":        "string",
				"description": "One short line (<150 chars) naming the session's subject. Format: '<repo>: <subsystem>; <one or two key nouns>'. Used as a contextual prefix prepended to each chunk during embedding.",
			},
			"tags": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"projects": map[string]any{
						"type":        "array",
						"description": "Repo or project names drawn from the evidence. Canonical form (\"diderot\" not \"the diderot project\").",
						"items":       map[string]any{"type": "string"},
					},
					"languages": map[string]any{
						"type":        "array",
						"description": "Programming languages worked with in the session. Closed vocabulary.",
						"items":       map[string]any{"type": "string", "enum": langEnum},
					},
					"activities": map[string]any{
						"type":        "array",
						"description": "Kinds of work being done. Closed vocabulary.",
						"items":       map[string]any{"type": "string", "enum": actEnum},
					},
					"freeform": map[string]any{
						"type":        "array",
						"description": "Up to 3 short tags for concepts not covered by the enums: features, subsystems, third-party services, infra components.",
						"items":       map[string]any{"type": "string"},
						"maxItems":    3,
					},
				},
				"required": []string{},
			},
		},
		"required": []string{"evidence", "narrative", "embedding_text", "context_hint", "tags"},
	}
}

// CallOptions configures a map or reduce request.
// No temperature: the Claude 5 family rejects the parameter
// (`temperature` is deprecated for this model → 400), and the
// tool-forced summarization calls never depended on it.
type CallOptions struct {
	ModelID   string
	MaxTokens int64
}

// MapCall runs one per-chunk summary via tool use. `chunkJSONL` is
// the (redacted) raw JSONL for this chunk; index/total let the prompt
// refer to chunk position.
func MapCall(ctx context.Context, client ToolClient, bundle PromptBundle, opts CallOptions, chunkJSONL []byte, index, total int) (ChunkSummary, Usage, error) {
	userText := fmt.Sprintf(
		"<chunk index=%q of=%q>\n%s\n</chunk>",
		fmt.Sprintf("%d", index),
		fmt.Sprintf("%d", total),
		chunkJSONL)

	resp, usage, err := invokeTool(ctx, client, ToolRequest{
		ModelID: opts.ModelID, MaxTokens: opts.MaxTokens, System: bundle.MapSystem, User: userText,
		ToolName:        "record_chunk_summary",
		ToolDescription: "Record the chunk summary. Call this exactly once, then stop.",
		ToolSchema:      chunkSummaryToolSchema(),
	})
	if err != nil {
		return ChunkSummary{}, usage, err
	}

	var cs ChunkSummary
	if err := json.Unmarshal(resp, &cs); err != nil {
		return ChunkSummary{}, usage, fmt.Errorf("decode chunk summary: %w", err)
	}
	return cs, usage, nil
}

// ReduceCall runs the final merge via tool use. `chunkSummaries` is
// the ordered slice of map-step outputs.
func ReduceCall(ctx context.Context, client ToolClient, bundle PromptBundle, opts CallOptions, chunkSummaries []ChunkSummary) (FinalSummary, Usage, error) {
	userBody, err := renderReduceInput(chunkSummaries)
	if err != nil {
		return FinalSummary{}, Usage{}, err
	}

	resp, usage, err := invokeTool(ctx, client, ToolRequest{
		ModelID: opts.ModelID, MaxTokens: opts.MaxTokens, System: bundle.ReduceSystem, User: userBody,
		ToolName:        "record_session_summary",
		ToolDescription: "Record the final session summary. Call this exactly once, then stop.",
		ToolSchema:      sessionSummaryToolSchema(bundle),
	})
	if err != nil {
		return FinalSummary{}, usage, err
	}

	var fs FinalSummary
	if err := json.Unmarshal(resp, &fs); err != nil {
		return FinalSummary{}, usage, fmt.Errorf("decode final summary: %w", err)
	}
	repairSwallowedFields(&fs)
	// A structurally-valid tool call can still carry empty fields: a
	// degenerate reduce of a few dozen output tokens would persist
	// narrative='' as success, and the event publish then fails on
	// the empty body.
	// An empty narrative is never a valid summary, so fail and let the
	// caller retry. EmbeddingText is an optimization rather than the
	// searchable source of truth: events.DBSink already falls back to
	// chunking the narrative when it is empty. Some otherwise-complete
	// reducer responses omit this field, and rejecting those would park
	// a usable session forever.
	if strings.TrimSpace(fs.Narrative) == "" {
		return FinalSummary{}, usage, fmt.Errorf(
			"reduce returned empty summary (narrative %d chars)", len(fs.Narrative))
	}
	return fs, usage, nil
}

// Usage is the total input/output token count, including cached input tokens.
// Stored on session_summaries for cost auditing.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// invokeTool makes the stage's forced tool call and returns the tool's
// arguments, which must be a JSON object whatever protocol carried them.
func invokeTool(ctx context.Context, client ToolClient, request ToolRequest) ([]byte, Usage, error) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()

	arguments, usage, err := client.CallTool(ctx, request)
	if err != nil {
		return nil, usage, err
	}
	var object map[string]any
	if err := json.Unmarshal(arguments, &object); err != nil || object == nil {
		return nil, usage, fmt.Errorf("tool call arguments must be a JSON object")
	}
	// The summaries decoded from this are written to Postgres text
	// columns, and "\u0000" in a model's JSON decodes to a NUL byte.
	return events.SanitizeJSON(arguments), usage, nil
}

// OverMessages is the ToolClient that speaks Anthropic Messages through
// client, forcing the tool with tool_choice.
func OverMessages(client MessageClient) ToolClient { return messagesTools{client: client} }

type messagesTools struct{ client MessageClient }

func (m messagesTools) CallTool(ctx context.Context, request ToolRequest) (json.RawMessage, Usage, error) {
	input := anthropic.MessageNewParams{
		Model:     anthropic.Model(request.ModelID),
		MaxTokens: request.MaxTokens,
		System: []anthropic.TextBlockParam{{
			Text:         request.System,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(request.User)),
		},
		Tools: []anthropic.ToolUnionParam{{OfTool: &anthropic.ToolParam{
			Name:        request.ToolName,
			Description: anthropic.String(request.ToolDescription),
			InputSchema: anthropic.ToolInputSchemaParam{ExtraFields: request.ToolSchema},
		}}},
		ToolChoice: anthropic.ToolChoiceUnionParam{
			OfTool: &anthropic.ToolChoiceToolParam{
				Name:                   request.ToolName,
				DisableParallelToolUse: anthropic.Bool(true),
			},
		},
	}

	resp, err := m.client.CreateMessage(ctx, input)
	if err != nil {
		return nil, Usage{}, fmt.Errorf("messages: %w", err)
	}
	if resp == nil {
		return nil, Usage{}, fmt.Errorf("messages: empty response")
	}
	usage := Usage{
		InputTokens:  int(resp.Usage.InputTokens + resp.Usage.CacheReadInputTokens + resp.Usage.CacheCreationInputTokens),
		OutputTokens: int(resp.Usage.OutputTokens),
	}
	if resp.StopReason == anthropic.StopReasonMaxTokens {
		return nil, usage, fmt.Errorf("response truncated at max_tokens (%d)", request.MaxTokens)
	}
	if resp.StopReason != anthropic.StopReasonToolUse {
		return nil, usage, fmt.Errorf("unexpected stop reason: %q", resp.StopReason)
	}
	var toolInput json.RawMessage
	found := false
	for _, block := range resp.Content {
		if block.Type != "tool_use" {
			continue
		}
		if block.Name != request.ToolName {
			return nil, usage, fmt.Errorf("unexpected tool_use %q, want %q", block.Name, request.ToolName)
		}
		if found {
			return nil, usage, fmt.Errorf("multiple tool_use blocks in response")
		}
		toolInput, found = block.Input, true
	}
	if !found {
		return nil, usage, fmt.Errorf("no tool_use block in response")
	}
	return toolInput, usage, nil
}
