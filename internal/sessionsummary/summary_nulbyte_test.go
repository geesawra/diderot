package sessionsummary

import (
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/v2"
)

// A model's tool input is JSON, where "\u0000" decodes to a NUL byte.
// Everything a summary call returns is written to Postgres text columns,
// which reject it, so the calls return storable text.
func TestSummaryCallsReturnStorableText(t *testing.T) {
	bundle := EmbeddedPromptBundle
	opts := CallOptions{ModelID: "test-model", MaxTokens: 2048}

	t.Run("map", func(t *testing.T) {
		client := &stubMessage{out: toolUseOutput("record_chunk_summary",
			`{"evidence":["read a\u0000b","plain"],"summary":"wrote \u0000 to the file"}`,
			anthropic.StopReasonToolUse)}
		got, _, err := MapCall(t.Context(), OverMessages(client), bundle, opts, []byte(`{}`), 1, 1)
		if err != nil {
			t.Fatalf("MapCall: %v", err)
		}
		if got.Summary != "wrote  to the file" || len(got.Evidence) != 2 || got.Evidence[0] != "read ab" {
			t.Fatalf("got summary %q, evidence %q", got.Summary, got.Evidence)
		}
	})

	t.Run("reduce", func(t *testing.T) {
		client := &stubMessage{out: toolUseOutput("record_session_summary",
			`{"evidence":["a\u0000"],"narrative":"the \u0000 byte","embedding_text":"k\u0000w","context_hint":"h\u0000","tags":{}}`,
			anthropic.StopReasonToolUse)}
		got, _, err := ReduceCall(t.Context(), OverMessages(client), bundle, opts, []ChunkSummary{{Summary: "chunk"}})
		if err != nil {
			t.Fatalf("ReduceCall: %v", err)
		}
		for name, value := range map[string]string{
			"narrative": got.Narrative, "embedding_text": got.EmbeddingText,
			"context_hint": got.ContextHint, "evidence": strings.Join(got.Evidence, ""),
		} {
			if strings.ContainsRune(value, 0) {
				t.Errorf("%s still has a NUL byte: %q", name, value)
			}
		}
	})
}
