package sessioningest

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/stretchr/testify/require"

	"github.com/bluesky-social/diderot/internal/promptcatalog"
	"github.com/bluesky-social/diderot/internal/sessiondecode"
	"github.com/bluesky-social/diderot/internal/sessionsummary"
)

// toolHeavyDocument is a session of turns, each opened by the harness's
// 8 KiB of instructions, sent again on every call, and each running a command
// that returns 4 KiB of output: the shape of most of a long session. Turn i's
// command is "make check-<i>", and every fifth fails with an error at the end
// of its output.
func toolHeavyDocument(id Identity, turns int) *sessionDocument {
	doc := docFor(id, "sha256:v1")
	doc.Transcript, doc.RecordTimes, doc.RecordCount = nil, nil, 0
	add := func(record sessiondecode.Record) {
		record.Index = doc.RecordCount
		record.ID = fmt.Sprintf("rec-%d", doc.RecordCount)
		record.Timestamp = "2026-11-18T10:00:00Z"
		record.Model, record.ProviderModel, record.InvocationID = "agent-model", "provider/agent-model", fmt.Sprintf("inv-%d", doc.RecordCount)
		line, err := json.Marshal(record)
		if err != nil {
			panic(err)
		}
		if doc.RecordCount > 0 {
			doc.Transcript = append(doc.Transcript, '\n')
		}
		doc.Transcript = append(doc.Transcript, line...)
		doc.RecordTimes = append(doc.RecordTimes, time.Date(2026, 11, 18, 10, 0, doc.RecordCount, 0, time.UTC))
		doc.RecordCount++
	}
	text := func(value string) json.RawMessage {
		raw, _ := json.Marshal(value)
		return raw
	}
	instructions := strings.Repeat("You are a coding agent working in this repository. ", 160)
	for turn := range turns {
		call := fmt.Sprintf("call-%d", turn)
		add(sessiondecode.Record{Role: "system", Content: text(instructions)})
		add(sessiondecode.Record{Role: "user", Content: text(fmt.Sprintf("Run check %d and fix what fails.", turn))})
		add(sessiondecode.Record{Role: "assistant", Content: text("Running the check."), ToolCalls: []sessiondecode.ToolCall{
			{ID: call, Name: "Bash", Args: fmt.Sprintf(`{"command":"make check-%d"}`, turn)},
		}})
		output := strings.Repeat(fmt.Sprintf("ok   package/%d/module\n", turn), 200)
		failed := turn%5 == 0
		if failed {
			output += fmt.Sprintf("FAIL check-%d: expected 200, got 503\n", turn)
		}
		add(sessiondecode.Record{Role: "tool", ToolCallID: call, IsError: failed, Content: text(output)})
	}
	return doc
}

type mapRecorder struct {
	mu        sync.Mutex
	mapInputs []string
}

func (m *mapRecorder) CreateMessage(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
	if input.ToolChoice.OfTool.Name == "record_chunk_summary" {
		m.mu.Lock()
		m.mapInputs = append(m.mapInputs, input.Messages[0].Content[0].OfText.Text)
		m.mu.Unlock()
	}
	return (cycleMessages{}).CreateMessage(ctx, input)
}

func (m *mapRecorder) sent() (bytes int, all string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, input := range m.mapInputs {
		bytes += len(input)
	}
	return bytes, strings.Join(m.mapInputs, "\n")
}

func TestMapReadsTheLeanRendering(t *testing.T) {
	store, _ := newTestStore(t)
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "lean-cost"}
	seedSummaryCandidate(t, store, id, "sha256:v1", 1)
	doc := toolHeavyDocument(id, 40)
	model := &mapRecorder{}
	summarizer := NewSessionsummarySummarizer(store, &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}, sessionsummary.OverMessages(model), "map-model", "reduce-model", 0, 4)
	result, err := summarizer.Summarize(t.Context(), doc, id, "sha256:v1", 1)
	require.NoError(t, err)

	sent, all := model.sent()
	t.Logf("transcript %d bytes, sent to the map model %d bytes (%.0f%%)", len(doc.Transcript), sent, 100*float64(sent)/float64(len(doc.Transcript)))
	require.Less(t, sent, len(doc.Transcript)/2, "repeated instructions and record metadata must not reach the map model")
	for turn := range 40 {
		require.Contains(t, all, fmt.Sprintf("make check-%d", turn), "every command reaches the map model")
		require.Contains(t, all, strings.Repeat(fmt.Sprintf(`ok   package/%d/module\n`, turn), 200), "every output reaches it whole")
		if turn%5 == 0 {
			require.Contains(t, all, fmt.Sprintf("FAIL check-%d: expected 200, got 503", turn), "every failure reaches the map model")
		}
	}
	require.NotContains(t, all, "invocation_id")

	chunks := sessionsummary.ChunkTranscript(doc.Transcript, 0)
	require.Len(t, result.Chunks, len(chunks))
	for index, chunk := range chunks {
		require.Equal(t, shaHex(chunk.Bytes), result.Chunks[index].ContentHash, "a chunk is still identified by its full records")
	}
}

// Chunks mapped before the lean rendering existed were mapped from their full
// records under the same map identity. Their next pass reuses them, maps only
// the open and new chunks, and maps those from the lean rendering.
func TestChunksMappedBeforeLeanRenderingAreReused(t *testing.T) {
	store, _ := newTestStore(t)
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "lean-reuse"}
	release := &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}
	bundle, err := sessionsummary.PromptBundleFromCatalog(release.bundle.Render(promptcatalog.Values()))
	require.NoError(t, err)
	mapOpts := sessionsummary.CallOptions{ModelID: "map-model", MaxTokens: 1024}
	reduceOpts := sessionsummary.CallOptions{ModelID: "reduce-model", MaxTokens: 4096}

	before := toolHeavyDocument(id, 20)
	chunks := sessionsummary.ChunkTranscript(before.Transcript, 0)
	require.GreaterOrEqual(t, len(chunks), 4)
	require.NoError(t, store.UpsertSessionSummary(t.Context(), SessionSummaryDoc{
		Identity: id, Revision: "sha256:before", RewriteEpoch: 1,
		InputDigest: summaryInputDigest(bundle, "session_summary", mapOpts, reduceOpts, chunks),
		Narrative:   "summary written from full records", EmbeddingText: "earlier embedding",
		ContextHint: "earlier hint", ModelID: "map-model", RecordCount: before.RecordCount,
	}))
	for _, chunk := range chunks {
		require.NoError(t, store.UpsertSummaryRecord(t.Context(), SummaryRecord{
			Identity: id, RecordStart: chunk.RecordStart, RecordEnd: chunk.RecordEnd,
			ContentHash: shaHex(chunk.Bytes), MapDigest: summaryMapInputDigest(bundle, "session_summary", mapOpts),
			Evidence: []string{"make check"}, Summary: fmt.Sprintf("full-record summary %d", chunk.RecordStart), ModelID: "map-model",
		}))
	}

	seedSummaryCandidate(t, store, id, "sha256:after", 1)
	after := toolHeavyDocument(id, 30)
	model := &mapRecorder{}
	summarizer := NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(model), "map-model", "reduce-model", 0, 1)
	result, err := summarizer.Summarize(t.Context(), after, id, "sha256:after", 1)
	require.NoError(t, err)

	afterChunks := sessionsummary.ChunkTranscript(after.Transcript, 0)
	unchanged := 0
	for index, chunk := range chunks {
		if afterChunks[index].RecordEnd == chunk.RecordEnd {
			unchanged++
			require.Equal(t, fmt.Sprintf("full-record summary %d", chunk.RecordStart), result.Chunks[index].Summary,
				"a chunk mapped from its full records keeps its summary")
		}
	}
	require.GreaterOrEqual(t, unchanged, len(chunks)-1)
	require.Len(t, model.mapInputs, len(afterChunks)-unchanged, "only the open chunk and the new ones are mapped")
	_, all := model.sent()
	require.NotContains(t, all, "invocation_id", "new chunks are mapped from the lean rendering")
	require.NotContains(t, all, "make check-0\"", "a closed chunk's records are not sent again")
	require.Equal(t, len(afterChunks)-unchanged+1, result.ReduceEntries, "the reduce folds the earlier summary")
}
