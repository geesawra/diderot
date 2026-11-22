package sessionsummary

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The shapes below are the ones stored summaries show, with invented content.
func TestRepairSwallowedFields(t *testing.T) {
	const narrative = "The person moved the nightly export onto the job queue and fixed a timezone bug."
	const embedding = "billing-api export worker, internal/exportworker/worker.go, TestExportAcrossMidnight. Topic: export queue"
	const hint = "billing-api: export worker; timezone bug"
	for _, tc := range []struct {
		name string
		in   FinalSummary
		want FinalSummary
	}{
		{
			name: "embedding text after the narrative's closing tag",
			in: FinalSummary{
				Narrative:   narrative + "</narrative>\n<parameter name=\"embedding_text\">" + embedding,
				ContextHint: hint, Tags: TagSet{Projects: []string{"billing-api"}},
			},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint, Tags: TagSet{Projects: []string{"billing-api"}}},
		},
		{
			name: "every later field swallowed, each closed",
			in: FinalSummary{Narrative: narrative + "</narrative>\n<parameter name=\"embedding_text\">" + embedding +
				"</parameter>\n<parameter name=\"context_hint\">" + hint +
				"</parameter>\n<parameter name=\"tags\">{\"projects\": [\"billing-api\"], \"languages\": [\"go\"], \"activities\": [\"debug\"], \"freeform\": []}"},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint,
				Tags: TagSet{Projects: []string{"billing-api"}, Languages: []string{"go"}, Activities: []string{"debug"}}},
		},
		{
			name: "narrative closed as a parameter",
			in:   FinalSummary{Narrative: narrative + "</parameter>\n<parameter name=\"embedding_text\">" + embedding, ContextHint: hint},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint},
		},
		{
			name: "narrative closed as a summary",
			in:   FinalSummary{Narrative: narrative + "</summary>\n<parameter name=\"embedding_text\">" + embedding, ContextHint: hint},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint},
		},
		{
			name: "evidence swallowed while the other fields arrived",
			in: FinalSummary{Narrative: narrative + "</narrative>\n<parameter name=\"evidence\">[\"worker.go\", \"expected 2026-03-09\"]",
				EmbeddingText: embedding, ContextHint: hint},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint,
				Evidence: StringList{"worker.go", "expected 2026-03-09"}},
		},
		{
			name: "element form with CDATA evidence and per-list tags",
			in: FinalSummary{Narrative: narrative + "</narrative>\n<evidence>\n<![CDATA[\n\"worker.go\"\n]]>\n</evidence>\n<tags>\n" +
				"<projects>[\"billing-api\"]</projects>\n<languages>[\"go\"]</languages>\n<activities>[\"debug\"]</activities>\n<freeform>[]</freeform>\n</tags>\n",
				EmbeddingText: embedding, ContextHint: hint},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint,
				Evidence: StringList{"worker.go"}, Tags: TagSet{Projects: []string{"billing-api"}, Languages: []string{"go"}, Activities: []string{"debug"}}},
		},
		{
			name: "element form with JSON tags",
			in: FinalSummary{Narrative: narrative + "\n</narrative>\n<tags>{\"projects\": [\"billing-api\"]}</tags>",
				EmbeddingText: embedding, ContextHint: hint},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint, Tags: TagSet{Projects: []string{"billing-api"}}},
		},
		{
			name: "narrative written after evidence inside the narrative field",
			in: FinalSummary{Narrative: "worker.go; expected 2026-03-09</evidence>\n<narrative>" + narrative + "</narrative>\n<embedding_text>" + embedding +
				"</embedding_text>\n<context_hint>" + hint + "</context_hint>\n<tags>{\"projects\": [\"billing-api\"]}</tags>"},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint, Tags: TagSet{Projects: []string{"billing-api"}}},
		},
		{
			name: "stray closing tag",
			in:   FinalSummary{Narrative: narrative + "</narrative>", EmbeddingText: embedding, ContextHint: hint},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint},
		},
		{
			name: "embedding text swallowed the context hint",
			in:   FinalSummary{Narrative: narrative, EmbeddingText: embedding + "</embedding_text>\n<parameter name=\"context_hint\">" + hint},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint},
		},
		{
			name: "the model's own field wins over a swallowed one",
			in:   FinalSummary{Narrative: narrative + "</narrative>\n<parameter name=\"context_hint\">older hint", EmbeddingText: embedding, ContextHint: hint},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint},
		},
		{
			name: "angle brackets that are content stay",
			in: FinalSummary{Narrative: "Documented the s3://<bucket>/<repo>/<slug> layout and the <evidence> element of the reduce input.",
				EmbeddingText: embedding, ContextHint: hint},
			want: FinalSummary{Narrative: "Documented the s3://<bucket>/<repo>/<slug> layout and the <evidence> element of the reduce input.",
				EmbeddingText: embedding, ContextHint: hint},
		},
		{
			name: "parameter markup that is content stays",
			in: FinalSummary{Narrative: `Documented the <parameter name="tags"> syntax and updated the parser.`,
				EmbeddingText: embedding, ContextHint: hint},
			want: FinalSummary{Narrative: `Documented the <parameter name="tags"> syntax and updated the parser.`,
				EmbeddingText: embedding, ContextHint: hint},
		},
		{
			name: "placeholder tags dropped",
			in: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint,
				Tags: TagSet{Projects: []string{"<UNKNOWN>", "billing-api"}}},
			want: FinalSummary{Narrative: narrative, EmbeddingText: embedding, ContextHint: hint, Tags: TagSet{Projects: []string{"billing-api"}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.in
			repairSwallowedFields(&got)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestRepairStoredSummary(t *testing.T) {
	stored := StoredSummary{
		Narrative:     "Fixed the export.</narrative>\n<parameter name=\"tags\">{\"languages\": [\"go\"]}",
		EmbeddingText: "export worker", ContextHint: "billing-api: export", Tags: []string{"lang:<UNKNOWN>"},
	}
	repaired, changed, complete := RepairStoredSummary(stored)
	assert.True(t, changed)
	assert.True(t, complete)
	assert.Equal(t, "Fixed the export.", repaired.Narrative)
	assert.Equal(t, []string{"lang:go"}, repaired.Tags, "tags recovered from the text replace placeholder-only stored tags")

	_, changed, complete = RepairStoredSummary(StoredSummary{Narrative: "Fixed the export.</narrative>", Tags: []string{"project:x"}})
	assert.True(t, changed)
	assert.False(t, complete, "a summary without its search text must be written again")

	clean := StoredSummary{Narrative: "Fixed the export.", EmbeddingText: "e", ContextHint: "h", Tags: []string{"project:x"}}
	_, changed, _ = RepairStoredSummary(clean)
	assert.False(t, changed, "a clean summary is left alone")
}

func TestReduceCallRepairsSwallowedFields(t *testing.T) {
	input, err := json.Marshal(map[string]any{
		"evidence": []string{"worker.go"},
		"narrative": "Moved the export onto the job queue.</narrative>\n<parameter name=\"embedding_text\">export worker queue. Topic: export" +
			"</parameter>\n<parameter name=\"context_hint\">billing-api: export worker",
		"tags": map[string]any{"projects": []string{"billing-api"}},
	})
	require.NoError(t, err)
	client := messageClientFunc(func(context.Context, anthropic.MessageNewParams) (*anthropic.Message, error) {
		return &anthropic.Message{
			StopReason: anthropic.StopReasonToolUse,
			Content:    []anthropic.ContentBlockUnion{{Type: "tool_use", Name: "record_session_summary", Input: input}},
		}, nil
	})
	got, _, err := ReduceCall(t.Context(), OverMessages(client), EmbeddedPromptBundle, CallOptions{ModelID: "reduce", MaxTokens: 100},
		[]ChunkSummary{{Summary: "chunk"}})
	require.NoError(t, err)
	assert.Equal(t, "Moved the export onto the job queue.", got.Narrative)
	assert.Equal(t, "export worker queue. Topic: export", got.EmbeddingText)
	assert.Equal(t, "billing-api: export worker", got.ContextHint)
}

type messageClientFunc func(context.Context, anthropic.MessageNewParams) (*anthropic.Message, error)

func (f messageClientFunc) CreateMessage(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
	return f(ctx, params)
}
