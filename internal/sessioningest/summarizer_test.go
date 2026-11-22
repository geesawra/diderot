package sessioningest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/anthropics/anthropic-sdk-go/v2"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/bluesky-social/diderot/internal/promptcatalog"
	"github.com/bluesky-social/diderot/internal/sessionsummary"
)

func stubToolOutput(toolName, inputJSON string) *anthropic.Message {
	return &anthropic.Message{
		StopReason: anthropic.StopReasonToolUse,
		Content: []anthropic.ContentBlockUnion{{
			Type: "tool_use", Name: toolName, Input: json.RawMessage(inputJSON),
		}},
		Usage: anthropic.Usage{InputTokens: 10, OutputTokens: 5},
	}
}

type cycleMessages struct{}

func (cycleMessages) CreateMessage(_ context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
	switch params.ToolChoice.OfTool.Name {
	case "record_chunk_summary":
		return stubToolOutput("record_chunk_summary", `{"evidence":["file.go","run tests"],"summary":"fixed the bug in this chunk"}`), nil
	case "record_session_summary":
		return stubToolOutput("record_session_summary", `{"evidence":["file.go"],"narrative":"Session summary narrative","embedding_text":"embedding tokens","context_hint":"repo: task","tags":{"projects":["x"]}}`), nil
	}
	return nil, errors.New("unexpected summary tool")
}

type countingCycleMessages struct {
	mu          sync.Mutex
	calls       int
	mapCalls    int
	reduceCalls int
}

type rejectingMessages struct{ calls atomic.Int32 }

func (client *rejectingMessages) CreateMessage(context.Context, anthropic.MessageNewParams) (*anthropic.Message, error) {
	client.calls.Add(1)
	return nil, errors.New("inference must not run")
}

func (client *countingCycleMessages) CreateMessage(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
	client.mu.Lock()
	client.calls++
	switch params.ToolChoice.OfTool.Name {
	case "record_chunk_summary":
		client.mapCalls++
	case "record_session_summary":
		client.reduceCalls++
	}
	client.mu.Unlock()
	return (cycleMessages{}).CreateMessage(ctx, params)
}

type staticPromptRelease struct{ bundle promptcatalog.Bundle }

func seedSummaryCandidate(t *testing.T, store *Store, id Identity, revision string, rewriteEpoch int64) {
	t.Helper()
	head := change(1, id.SessionID, id.Owner, id.Client)
	head.Revision, head.RewriteEpoch = revision, rewriteEpoch
	_, err := store.UpsertChanges(context.Background(), []SessionChange{head}, admitAll)
	if err != nil {
		t.Fatal(err)
	}
}

func (s *staticPromptRelease) Current(context.Context) (*promptcatalog.Release, error) {
	return &promptcatalog.Release{ID: 1, Bundle: s.bundle}, nil
}

func seedLegacySummary(t *testing.T, store *Store, doc *sessionDocument, id Identity, mutate func(*SessionSummaryDoc, []SummaryRecord)) {
	t.Helper()
	chunks := (&sessionsummarySummarizer{}).chunkRecords(doc, 0)
	records := make([]SummaryRecord, 0, len(chunks))
	for _, chunk := range chunks {
		records = append(records, SummaryRecord{
			Identity: id, RecordStart: chunk.RecordStart, RecordEnd: chunk.RecordEnd,
			ContentHash: shaHex(chunk.Bytes), Evidence: []string{"legacy"},
			Summary: "legacy chunk summary", ModelID: legacySummaryMapModel,
		})
	}
	summary := SessionSummaryDoc{
		Identity: id, Revision: "sha256:legacy", RewriteEpoch: 1,
		Narrative: "legacy session summary", EmbeddingText: "legacy embedding",
		ContextHint: "legacy context", Tags: []string{"legacy"},
		ModelID: legacySummaryMapModel, InputTokens: 111, OutputTokens: 22,
		RecordCount: doc.RecordCount,
	}
	if mutate != nil {
		mutate(&summary, records)
	}
	if err := store.UpsertSessionSummary(context.Background(), summary); err != nil {
		t.Fatal(err)
	}
	for _, record := range records {
		if err := store.UpsertSummaryRecord(context.Background(), record); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSessionsummarySummarizer_MapReduceOverRecords(t *testing.T) {
	sisStore, pool := newTestStore(t)
	ctx := context.Background()
	if _, err := pool.Exec(ctx, `TRUNCATE prompt_release_activations, prompt_active_release, prompt_release_entries, prompt_releases RESTART IDENTITY CASCADE`); err != nil {
		t.Fatal(err)
	}
	promptStore := promptcatalog.NewStore(pool)
	if _, err := promptStore.EnsureSeed(ctx, promptcatalog.MustEmbedded(), "test:sessioningest"); err != nil {
		t.Fatal(err)
	}

	id := Identity{Owner: "a@x", Client: "codex", SessionID: "s1"}
	doc := docFor(id, "sha256:v1")
	summ := NewSessionsummarySummarizer(sisStore, promptStore, sessionsummary.OverMessages(cycleMessages{}), "map-model", "reduce-model", 0, 0)
	seedSummaryCandidate(t, sisStore, id, "sha256:v1", 1)
	result, err := summ.Summarize(ctx, doc, id, "sha256:v1", 1)
	if err != nil {
		t.Fatal(err)
	}
	if result.Narrative != "Session summary narrative" {
		t.Errorf("body = %q, want reduce narrative", result.Narrative)
	}
	if result.EmbeddingText != "embedding tokens" {
		t.Errorf("embedding = %q", result.EmbeddingText)
	}
	if len(result.Chunks) == 0 {
		t.Fatal("expected record-range chunks")
	}
}

func TestSessionsummarySummarizer_ReusesExactInputDigest(t *testing.T) {
	sisStore, pool := newTestStore(t)
	ctx := context.Background()
	if _, err := pool.Exec(ctx, `TRUNCATE prompt_release_activations, prompt_active_release, prompt_release_entries, prompt_releases RESTART IDENTITY CASCADE`); err != nil {
		t.Fatal(err)
	}
	promptStore := promptcatalog.NewStore(pool)
	if _, err := promptStore.EnsureSeed(ctx, promptcatalog.MustEmbedded(), "test:sessioningest-reuse"); err != nil {
		t.Fatal(err)
	}

	id := Identity{Owner: "a@x", Client: "codex", SessionID: "reuse"}
	client := &countingCycleMessages{}
	summarizer := NewSessionsummarySummarizer(sisStore, promptStore, sessionsummary.OverMessages(client),
		"map-model", "reduce-model", 0, 0)

	seedSummaryCandidate(t, sisStore, id, "sha256:v1", 1)
	first, err := summarizer.Summarize(ctx, docFor(id, "sha256:v1"), id, "sha256:v1", 1)
	if err != nil {
		t.Fatal(err)
	}
	if first.Reused || first.InputDigest == "" {
		t.Fatalf("first summary = %+v, want generated summary with digest", first)
	}
	firstCalls := client.calls
	if firstCalls == 0 {
		t.Fatal("first summary made no inference calls")
	}
	if err := sisStore.UpsertSessionSummary(ctx, SessionSummaryDoc{
		Identity: id, Revision: "sha256:v1", RewriteEpoch: 1,
		InputDigest: first.InputDigest, Narrative: first.Narrative,
		EmbeddingText: first.EmbeddingText, ContextHint: first.ContextHint,
		Tags: first.Tags, ModelID: first.ModelID, InputTokens: first.InputTokens,
		OutputTokens: first.OutputTokens, RecordCount: 1,
	}); err != nil {
		t.Fatal(err)
	}
	for _, chunk := range first.Chunks {
		if err := sisStore.UpsertSummaryRecord(ctx, chunk); err != nil {
			t.Fatal(err)
		}
	}

	// The revision/envelope changed, but the exact record bytes, prompt,
	// models, token limits, and chunking inputs did not.
	seedSummaryCandidate(t, sisStore, id, "sha256:v2", 2)
	second, err := summarizer.Summarize(ctx, docFor(id, "sha256:v2"), id, "sha256:v2", 2)
	if err != nil {
		t.Fatal(err)
	}
	if !second.Reused {
		t.Fatal("projection-only revision did not reuse its summary")
	}
	if client.calls != firstCalls {
		t.Fatalf("reuse made %d additional inference calls", client.calls-firstCalls)
	}
	if second.Narrative != first.Narrative || len(second.Chunks) != len(first.Chunks) {
		t.Fatal("reused summary content drifted")
	}
	if _, err := pool.Exec(ctx, `UPDATE session_ingest_summaries
		SET narrative='   '
		WHERE owner=$1 AND client=$2 AND session_id=$3`, id.Owner, id.Client, id.SessionID); err != nil {
		t.Fatal(err)
	}
	beforeEmpty := client.calls
	empty, err := summarizer.Summarize(ctx, docFor(id, "sha256:v2"), id, "sha256:v2", 2)
	if err != nil {
		t.Fatal(err)
	}
	if empty.Reused || client.calls == beforeEmpty {
		t.Fatal("blank summary cache did not fail closed to inference")
	}
	if _, err := pool.Exec(ctx, `UPDATE session_ingest_summaries
		SET narrative=$1
		WHERE owner=$2 AND client=$3 AND session_id=$4`, first.Narrative, id.Owner, id.Client, id.SessionID); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, `UPDATE session_ingest_chunk_summaries
		SET content_hash='sha256:corrupt'
		WHERE owner=$1 AND client=$2 AND session_id=$3`, id.Owner, id.Client, id.SessionID); err != nil {
		t.Fatal(err)
	}
	beforeCorrupt := client.calls
	corrupt, err := summarizer.Summarize(ctx, docFor(id, "sha256:v2"), id, "sha256:v2", 2)
	if err != nil {
		t.Fatal(err)
	}
	if corrupt.Reused || client.calls == beforeCorrupt {
		t.Fatal("corrupt chunk cache did not fail closed to inference")
	}

	changed := docFor(id, "sha256:v3")
	changed.Transcript = []byte(`{"index":0,"id":"rec0","role":"user","content":"changed","timestamp":"2026-07-24T18:00:00Z"}`)
	seedSummaryCandidate(t, sisStore, id, "sha256:v3", 3)
	third, err := summarizer.Summarize(ctx, changed, id, "sha256:v3", 3)
	if err != nil {
		t.Fatal(err)
	}
	if third.Reused {
		t.Fatal("changed record bytes reused an old summary")
	}
	if client.calls == firstCalls {
		t.Fatal("changed record bytes made no inference calls")
	}
}

// TestAdoptLegacySummaryStampsExactRows: a summary stored before input
// digests, whose chunks cover the same records with the same content under
// the same model, is returned as stored and stamped with both digests.
func TestAdoptLegacySummaryStampsExactRows(t *testing.T) {
	store, pool := newTestStore(t)
	ctx := context.Background()
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "legacy-adopt"}
	doc := docFor(id, "sha256:v1")
	seedLegacySummary(t, store, doc, id, nil)
	chunks := (&sessionsummarySummarizer{}).chunkRecords(doc, 0)

	result, ok, err := store.AdoptLegacySummary(ctx, id, "sha256:input", "sha256:map", doc.RecordCount, chunks, legacySummaryMapModel)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || result.Narrative != "legacy session summary" || result.InputDigest != "sha256:input" {
		t.Fatalf("adopted result = %+v, ok=%t", result, ok)
	}
	var summaryDigest string
	if err := pool.QueryRow(ctx, `SELECT input_digest FROM session_ingest_summaries
		WHERE owner=$1 AND client=$2 AND session_id=$3`, id.Owner, id.Client, id.SessionID).Scan(&summaryDigest); err != nil {
		t.Fatal(err)
	}
	if summaryDigest != "sha256:input" {
		t.Fatalf("stored summary digest = %q", summaryDigest)
	}
	var otherMapDigests int
	if err := pool.QueryRow(ctx, `SELECT count(*) FROM session_ingest_chunk_summaries
		WHERE owner=$1 AND client=$2 AND session_id=$3 AND map_input_digest<>'sha256:map'`, id.Owner, id.Client, id.SessionID).Scan(&otherMapDigests); err != nil {
		t.Fatal(err)
	}
	if otherMapDigests != 0 {
		t.Fatalf("%d adopted chunks were not stamped with the map digest", otherMapDigests)
	}

	// Adopting again under the same digests returns the same summary; under
	// other digests the row is someone else's and is left alone.
	if _, ok, err := store.AdoptLegacySummary(ctx, id, "sha256:input", "sha256:map", doc.RecordCount, chunks, legacySummaryMapModel); err != nil || !ok {
		t.Fatalf("second adoption: ok=%t err=%v", ok, err)
	}
	if _, ok, err := store.AdoptLegacySummary(ctx, id, "sha256:other", "sha256:map", doc.RecordCount, chunks, legacySummaryMapModel); err != nil || ok {
		t.Fatalf("adoption under another digest: ok=%t err=%v", ok, err)
	}
}

// TestAdoptLegacySummaryRefusesRowsItCannotProve: stored rows whose content,
// coverage or model differ from what is being summarized are not adopted and
// keep their empty digests.
func TestAdoptLegacySummaryRefusesRowsItCannotProve(t *testing.T) {
	tests := []struct {
		name       string
		mutateRows func(*SessionSummaryDoc, []SummaryRecord)
		mapModel   string
	}{
		{name: "content hash", mutateRows: func(_ *SessionSummaryDoc, chunks []SummaryRecord) {
			chunks[0].ContentHash = "sha256:wrong"
		}},
		{name: "range coverage", mutateRows: func(_ *SessionSummaryDoc, chunks []SummaryRecord) {
			chunks[0].RecordEnd++
		}},
		{name: "map model", mapModel: "different-map-model"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store, pool := newTestStore(t)
			ctx := context.Background()
			id := Identity{Owner: "a@x", Client: "codex", SessionID: "legacy-refused-" + strings.ReplaceAll(tt.name, " ", "-")}
			doc := docFor(id, "sha256:v1")
			seedLegacySummary(t, store, doc, id, tt.mutateRows)
			mapModel := legacySummaryMapModel
			if tt.mapModel != "" {
				mapModel = tt.mapModel
			}
			chunks := (&sessionsummarySummarizer{}).chunkRecords(doc, 0)
			_, ok, err := store.AdoptLegacySummary(ctx, id, "sha256:input", "sha256:map", doc.RecordCount, chunks, mapModel)
			if err != nil {
				t.Fatal(err)
			}
			if ok {
				t.Fatal("a row that does not match was adopted")
			}
			var digest string
			if err := pool.QueryRow(ctx, `SELECT input_digest FROM session_ingest_summaries
				WHERE owner=$1 AND client=$2 AND session_id=$3`, id.Owner, id.Client, id.SessionID).Scan(&digest); err != nil {
				t.Fatal(err)
			}
			if digest != "" {
				t.Fatalf("refused row was stamped with %q", digest)
			}
		})
	}
}

func TestSessionsummarySummarizer_LegacyAdoptionFailsClosed(t *testing.T) {
	tests := []struct {
		name       string
		mutateRows func(*SessionSummaryDoc, []SummaryRecord)
		mutateRun  func(*staticPromptRelease, *string, *string)
	}{
		{name: "content hash", mutateRows: func(_ *SessionSummaryDoc, chunks []SummaryRecord) {
			chunks[0].ContentHash = "sha256:wrong"
		}},
		{name: "range coverage", mutateRows: func(_ *SessionSummaryDoc, chunks []SummaryRecord) {
			chunks[0].RecordEnd++
		}},
		{name: "map prompt", mutateRun: func(release *staticPromptRelease, _, _ *string) {
			for i := range release.bundle.Prompts {
				if release.bundle.Prompts[i].Key == promptcatalog.SummarySessionMap {
					release.bundle.Prompts[i].Body += "\nChanged."
				}
			}
		}},
		{name: "map model", mutateRun: func(_ *staticPromptRelease, mapModel, _ *string) {
			*mapModel = "different-map-model"
		}},
		{name: "reduce model", mutateRun: func(_ *staticPromptRelease, _, reduceModel *string) {
			*reduceModel = "different-reduce-model"
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store, pool := newTestStore(t)
			ctx := context.Background()
			id := Identity{Owner: "a@x", Client: "codex", SessionID: "legacy-mismatch-" + strings.ReplaceAll(tt.name, " ", "-")}
			doc := docFor(id, "sha256:v1")
			seedLegacySummary(t, store, doc, id, tt.mutateRows)
			release := &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}
			mapModel, reduceModel := legacySummaryMapModel, legacySummaryReduceModel
			if tt.mutateRun != nil {
				tt.mutateRun(release, &mapModel, &reduceModel)
			}
			client := &countingCycleMessages{}
			summarizer := NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(client), mapModel, reduceModel, 0, 0)
			seedSummaryCandidate(t, store, id, "sha256:v1", 1)
			result, err := summarizer.Summarize(ctx, doc, id, "sha256:v1", 1)
			if err != nil {
				t.Fatal(err)
			}
			if result.Reused || client.calls == 0 {
				t.Fatalf("uncertain legacy cache was adopted: reused=%t calls=%d", result.Reused, client.calls)
			}
			var digest string
			if err := pool.QueryRow(ctx, `SELECT input_digest FROM session_ingest_summaries
				WHERE owner=$1 AND client=$2 AND session_id=$3`, id.Owner, id.Client, id.SessionID).Scan(&digest); err != nil {
				t.Fatal(err)
			}
			if digest != "" {
				t.Fatalf("rejected legacy row was stamped with %q", digest)
			}
		})
	}
}

// TestAdoptLegacySummaryConcurrently: two adopters of the same rows both get
// the summary; the second finds it already stamped with its digests.
func TestAdoptLegacySummaryConcurrently(t *testing.T) {
	store, _ := newTestStore(t)
	ctx := context.Background()
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "legacy-concurrent"}
	doc := docFor(id, "sha256:v1")
	seedLegacySummary(t, store, doc, id, nil)
	chunks := (&sessionsummarySummarizer{}).chunkRecords(doc, 0)

	start := make(chan struct{})
	errs := make(chan error, 2)
	var wg sync.WaitGroup
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, ok, err := store.AdoptLegacySummary(ctx, id, "sha256:input", "sha256:map", doc.RecordCount, chunks, legacySummaryMapModel)
			if err == nil && !ok {
				err = errors.New("concurrent adopter did not get the summary")
			}
			errs <- err
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestSummaryMapInputDigestIncludesGenerationSemantics(test *testing.T) {
	mapOpts := sessionsummary.CallOptions{ModelID: "map", MaxTokens: 1024}
	bundle := sessionsummary.PromptBundle{MapSystem: "map prompt", ReduceSystem: "reduce prompt"}
	base := summaryMapInputDigest(bundle, "session_summary", mapOpts)
	changedMap := bundle
	changedMap.MapSystem = "changed"
	for name, digest := range map[string]string{
		"map prompt": summaryMapInputDigest(changedMap, "session_summary", mapOpts),
		"kind":       summaryMapInputDigest(bundle, "other", mapOpts),
		"map model": summaryMapInputDigest(bundle, "session_summary",
			sessionsummary.CallOptions{ModelID: "other", MaxTokens: 1024}),
		"map limit": summaryMapInputDigest(bundle, "session_summary",
			sessionsummary.CallOptions{ModelID: "map", MaxTokens: 2048}),
	} {
		if digest == base {
			test.Errorf("%s change did not invalidate map digest", name)
		}
	}
	changedReduce := bundle
	changedReduce.ReduceSystem = "changed"
	changedReduce.LanguageEnum = []string{"go"}
	changedReduce.ActivityEnum = []string{"test"}
	if summaryMapInputDigest(changedReduce, "session_summary", mapOpts) != base {
		test.Fatal("reduce-only change invalidated map digest")
	}
}

func TestSummaryInputDigestIncludesGenerationSemantics(t *testing.T) {
	chunks := []sessionsummary.Chunk{{RecordStart: 0, RecordEnd: 1, Bytes: []byte("record")}}
	mapOpts := sessionsummary.CallOptions{ModelID: "map", MaxTokens: 1024}
	reduceOpts := sessionsummary.CallOptions{ModelID: "reduce", MaxTokens: 4096}
	bundle := sessionsummary.PromptBundle{
		MapSystem: "map prompt", ReduceSystem: "reduce prompt",
		LanguageEnum: []string{"go"}, ActivityEnum: []string{"test"},
	}
	base := summaryInputDigest(bundle, "session_summary", mapOpts, reduceOpts, chunks)
	changedMap := bundle
	changedMap.MapSystem = "changed"
	changedReduce := bundle
	changedReduce.ReduceSystem = "changed"
	changedEnum := bundle
	changedEnum.LanguageEnum = []string{"go", "rust"}
	tests := map[string]string{
		"map prompt":    summaryInputDigest(changedMap, "session_summary", mapOpts, reduceOpts, chunks),
		"reduce prompt": summaryInputDigest(changedReduce, "session_summary", mapOpts, reduceOpts, chunks),
		"enum":          summaryInputDigest(changedEnum, "session_summary", mapOpts, reduceOpts, chunks),
		"kind":          summaryInputDigest(bundle, "other", mapOpts, reduceOpts, chunks),
		"map model": summaryInputDigest(bundle, "session_summary",
			sessionsummary.CallOptions{ModelID: "other", MaxTokens: 1024}, reduceOpts, chunks),
		"reduce limit": summaryInputDigest(bundle, "session_summary", mapOpts,
			sessionsummary.CallOptions{ModelID: "reduce", MaxTokens: 2048}, chunks),
		"record bytes": summaryInputDigest(bundle, "session_summary", mapOpts, reduceOpts,
			[]sessionsummary.Chunk{{RecordStart: 0, RecordEnd: 1, Bytes: []byte("changed")}}),
	}
	for name, got := range tests {
		if got == base {
			t.Errorf("%s change did not invalidate digest", name)
		}
	}
	left := bundle
	left.MapSystem, left.ReduceSystem = "a", "b\nc"
	right := bundle
	right.MapSystem, right.ReduceSystem = "a\nb", "c"
	if summaryInputDigest(left, "session_summary", mapOpts, reduceOpts, chunks) ==
		summaryInputDigest(right, "session_summary", mapOpts, reduceOpts, chunks) {
		t.Fatal("prompt field boundaries are ambiguous")
	}
}

func TestSessionsummarySummarizer_ReduceOnlyChangeReusesMapChunks(t *testing.T) {
	store, _ := newTestStore(t)
	ctx := context.Background()
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "reduce-only"}
	release := &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}
	client := &countingCycleMessages{}
	summarizer := NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(client), "map-model", "reduce-model", 0, 0)

	seedSummaryCandidate(t, store, id, "sha256:v1", 1)
	first, err := summarizer.Summarize(ctx, docFor(id, "sha256:v1"), id, "sha256:v1", 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, chunk := range first.Chunks {
		if chunk.MapDigest == "" {
			t.Fatal("generated map chunk is missing its input digest")
		}
		if err := store.UpsertSummaryRecord(ctx, chunk); err != nil {
			t.Fatal(err)
		}
	}
	mapCalls, reduceCalls := client.mapCalls, client.reduceCalls
	for i := range release.bundle.Prompts {
		if release.bundle.Prompts[i].Key == promptcatalog.SummarySessionReduce {
			release.bundle.Prompts[i].Body += "\nUse a different reduce rendering."
			break
		}
	}

	second, err := summarizer.Summarize(ctx, docFor(id, "sha256:v2"), id, "sha256:v2", 2)
	if err != nil {
		t.Fatal(err)
	}
	if second.Reused {
		t.Fatal("reduce-only change incorrectly reused the final summary")
	}
	if client.mapCalls != mapCalls {
		t.Fatalf("reduce-only change made %d new map calls", client.mapCalls-mapCalls)
	}
	if client.reduceCalls != reduceCalls+1 {
		t.Fatalf("reduce-only change made %d new reduce calls, want 1", client.reduceCalls-reduceCalls)
	}
}

func TestParseSessionDocument_BoundsOversizedLine(t *testing.T) {
	// Poison-session regression: one record line far larger than the
	// model context (e.g. a tool_result dump from a big file read or
	// shell output). Unbounded, such a line survives
	// chunking intact, produces a chunk that overflows the model context,
	// and fails the map call with a deterministic "prompt is too long"
	// 400. Bounded, the line must be shrunk below MaxLineBytes
	// with the visible truncation marker kept in the payload, every
	// chunk must stay under a safe byte ceiling, and the line-based
	// record ranges must stay contiguous across all records.
	const chunkBytes = 4096
	small := func(i int) []byte {
		return []byte(fmt.Sprintf(
			`{"index":%d,"id":"rec%d","role":"user","content":"hello","timestamp":"2026-07-24T18:00:00Z"}`,
			i, i))
	}
	poisonRecord := fmt.Sprintf(
		`{"index":1,"id":"rec1","role":"assistant","content":%q,"timestamp":"2026-07-24T18:00:01Z"}`,
		strings.Repeat("A", 200*1024))

	records := bytes.Join([][]byte{small(0), []byte(poisonRecord), small(2)}, []byte("\n"))
	body := []byte(fmt.Sprintf(`{"type":"session","schema":"bsky.session.v1","session_id":"ses_1","revision":"sha256:abc","record_count":3,"user_turn_count":2,"content_bytes":%d,"complete":true,"owner":"a@x","owner_kind":"human","source":"codex","session_role":"primary","started_at":"2026-07-24T18:00:00Z","updated_at":"2026-07-24T18:42:00Z"}`+"\n%s", len(records)+1, records))
	doc, err := parseSessionDocument(body, Identity{}, "sha256:abc", 256<<20)
	if err != nil {
		t.Fatal(err)
	}

	chunks := (&sessionsummarySummarizer{}).chunkRecords(doc, chunkBytes)
	if len(chunks) == 0 {
		t.Fatal("expected chunks for poisoned transcript")
	}
	// Line-based record ranges: contiguous coverage of [0, RecordCount).
	if chunks[0].RecordStart != 0 {
		t.Errorf("first chunk RecordStart = %d, want 0", chunks[0].RecordStart)
	}
	if last := chunks[len(chunks)-1]; last.RecordEnd != doc.RecordCount {
		t.Errorf("last chunk RecordEnd = %d, want %d", last.RecordEnd, doc.RecordCount)
	}
	var joined []byte
	for i, ch := range chunks {
		if i > 0 && ch.RecordStart != chunks[i-1].RecordEnd {
			t.Fatalf("chunk %d RecordStart %d != previous RecordEnd %d (record ranges must stay contiguous)",
				i, ch.RecordStart, chunks[i-1].RecordEnd)
		}
		if len(ch.Bytes) == 0 {
			t.Fatalf("chunk %d has empty bytes", i)
		}
		if len(ch.Bytes) > sessionsummary.MaxLineBytes+chunkBytes {
			t.Errorf("chunk %d size %d exceeds safe ceiling %d",
				i, len(ch.Bytes), sessionsummary.MaxLineBytes+chunkBytes)
		}
		joined = append(joined, ch.Bytes...)
		joined = append(joined, '\n')
	}
	// The oversized record must have been bounded in place, not passed
	// through as a ~200KB blob, and the visible truncation marker must
	// survive into the chunk payload.
	if len(poisonRecord) < 200*1024 {
		t.Fatal("test setup: poison record unexpectedly small")
	}
	if !bytes.Contains(joined, []byte("[... truncated:")) {
		t.Error("oversized line was not bounded: truncation marker missing from chunk payloads")
	}
}
