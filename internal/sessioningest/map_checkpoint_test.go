package sessioningest

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"github.com/anthropics/anthropic-sdk-go/v2"
	"testing"

	"github.com/bluesky-social/diderot/internal/promptcatalog"
	"github.com/bluesky-social/diderot/internal/sessionsummary"
	"github.com/stretchr/testify/require"
)

func TestMapCacheReusesUnchangedChunks(test *testing.T) {
	for _, scenario := range []struct {
		name         string
		failReduce   bool
		republish    bool
		changeMiddle bool
	}{
		{name: "published append"},
		{name: "checkpoint append", failReduce: true},
		{name: "cached republication append", republish: true},
		{name: "changed middle chunk", changeMiddle: true},
	} {
		test.Run(scenario.name, func(test *testing.T) {
			store, pool := newTestStore(test)
			id := Identity{Owner: "a@x", Client: "codex", SessionID: "map-cache-reuse"}
			seedSummaryCandidate(test, store, id, "sha256:v1", 1)
			calls := make(map[int]int)
			var reduces int
			failed := errors.New("reduce failed")
			client := summaryMessageFunc(func(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
				if string(input.Model) == "reduce-model" {
					reduces++
					if scenario.failReduce && reduces == 1 {
						return nil, failed
					}
				} else {
					index, err := summaryChunkIndex(input)
					if err != nil {
						return nil, err
					}
					calls[index]++
				}
				return (cycleMessages{}).CreateMessage(ctx, input)
			})
			release := &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}
			summarizer := NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 1)
			doc := summaryChunkDocument(id, 3)
			publish := func(result SummaryResult) {
				test.Helper()
				chunks := append([]SummaryRecord(nil), result.Chunks...)
				for chunkIndex := range chunks {
					chunks[chunkIndex].OccurredAt = doc.RecordTimes[0]
				}
				tx, err := pool.Begin(test.Context())
				require.NoError(test, err)
				defer func() { _ = tx.Rollback(test.Context()) }()
				require.NoError(test, store.ReplaceSummaryTx(test.Context(), tx, SessionSummaryDoc{
					Identity: id, Revision: "sha256:v1", RewriteEpoch: 1,
					InputDigest: result.InputDigest, Narrative: result.Narrative,
					EmbeddingText: result.EmbeddingText, ContextHint: result.ContextHint,
					Tags: result.Tags, ModelID: result.ModelID, InputTokens: result.InputTokens,
					OutputTokens: result.OutputTokens, RecordCount: doc.RecordCount,
				}, chunks))
				require.NoError(test, tx.Commit(test.Context()))
			}
			first, err := summarizer.Summarize(test.Context(), doc, id, "sha256:v1", 1)
			if scenario.failReduce {
				require.ErrorIs(test, err, failed)
			} else {
				require.NoError(test, err)
				publish(first)
				if scenario.republish {
					cached, err := summarizer.Summarize(test.Context(), doc, id, "sha256:v1", 1)
					require.NoError(test, err)
					require.True(test, cached.Reused)
					require.Equal(test, first.Chunks, cached.Chunks)
					publish(cached)
				}
			}
			require.Equal(test, map[int]int{1: 1, 2: 1, 3: 1}, calls)
			seedSummaryCandidate(test, store, id, "sha256:v2", 2)
			updated := summaryChunkDocument(id, 4)
			expected := map[int]int{1: 1, 2: 1, 3: 1, 4: 1}
			if scenario.changeMiddle {
				updated = summaryChunkDocument(id, 3)
				updated.Transcript = bytes.Replace(updated.Transcript, []byte("chunk-2"), []byte("changed chunk"), 1)
				expected = map[int]int{1: 1, 2: 2, 3: 1}
			}
			restarted := NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 1)
			result, err := restarted.Summarize(test.Context(), updated, id, "sha256:v2", 2)
			require.NoError(test, err)
			require.False(test, result.Reused)
			require.Len(test, result.Chunks, updated.RecordCount)
			require.Equal(test, expected, calls)
			require.Equal(test, 2, reduces)
			if !scenario.failReduce {
				require.NotEqual(test, first.InputDigest, result.InputDigest)
				for index, chunk := range first.Chunks {
					if !scenario.changeMiddle || index != 1 {
						require.Equal(test, chunk, result.Chunks[index])
					}
				}
			}
		})
	}
}

func TestMapCheckpointsResumeAfterReduceFailure(t *testing.T) {
	store, pool := newTestStore(t)
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "checkpoint-reduce"}
	seedSummaryCandidate(t, store, id, "sha256:v1", 1)
	var maps, reduces int
	failed := errors.New("reduce failed")
	client := summaryMessageFunc(func(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
		if string(input.Model) == "reduce-model" {
			reduces++
			if reduces == 1 {
				return nil, failed
			}
		} else {
			maps++
		}
		return (cycleMessages{}).CreateMessage(ctx, input)
	})
	release := &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}
	doc := summaryChunkDocument(id, 3)
	summarizer := NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 1)
	_, err := summarizer.Summarize(t.Context(), doc, id, "sha256:v1", 1)
	require.ErrorIs(t, err, failed)
	require.Equal(t, 3, maps)
	var checkpoints int
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT count(*) FROM session_ingest_map_checkpoints`).Scan(&checkpoints))
	require.Equal(t, 3, checkpoints)
	var published int
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT count(*) FROM session_ingest_chunk_summaries WHERE session_id=$1`, id.SessionID).Scan(&published))
	require.Zero(t, published)
	restarted := NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 1)
	result, err := restarted.Summarize(t.Context(), doc, id, "sha256:v1", 1)
	require.NoError(t, err)
	require.Len(t, result.Chunks, 3)
	require.Equal(t, 3, maps)
	require.Equal(t, 2, reduces)
}

func TestMapCheckpointsResumeOnlyMissingChunks(t *testing.T) {
	store, _ := newTestStore(t)
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "checkpoint-map"}
	seedSummaryCandidate(t, store, id, "sha256:v1", 1)
	failed := errors.New("one chunk failed")
	calls := make(map[int]int)
	client := summaryMessageFunc(func(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
		if string(input.Model) == "map-model" {
			index, err := summaryChunkIndex(input)
			if err != nil {
				return nil, err
			}
			calls[index]++
			if index == 2 && calls[index] == 1 {
				return nil, failed
			}
		}
		return (cycleMessages{}).CreateMessage(ctx, input)
	})
	release := &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}
	doc := summaryChunkDocument(id, 3)
	summarizer := NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 1)
	_, err := summarizer.Summarize(t.Context(), doc, id, "sha256:v1", 1)
	require.ErrorIs(t, err, failed)
	_, err = summarizer.Summarize(t.Context(), doc, id, "sha256:v1", 1)
	require.NoError(t, err)
	require.Equal(t, 1, calls[1])
	require.Equal(t, 2, calls[2])
	require.Equal(t, 1, calls[3])
}

func TestMapCheckpointsDoNotRequeuePublishedEmbeddings(t *testing.T) {
	harness := newChunkHarness(t)
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "checkpoint-isolation"}
	published := SummaryRecord{Identity: id, RecordStart: 0, RecordEnd: 1, ContentHash: "old-hash", Summary: "published summary", ModelID: "map-model"}
	harness.seedCandidate(t, id, "sha256:old", 1, 1)
	harness.publish(t, id, "sha256:old", 1, []SummaryRecord{published})
	_, err := harness.ep.ReconcileChunkEvents(t.Context(), 100)
	require.NoError(t, err)
	var before string
	require.NoError(t, harness.pool.QueryRow(t.Context(), `SELECT jsonb_agg(to_jsonb(c) ORDER BY c.id)::text FROM event_chunks c`).Scan(&before))
	harness.seedCandidate(t, id, "sha256:new", 2, 1)
	checkpoint := published
	checkpoint.Summary, checkpoint.ContentHash, checkpoint.MapDigest = "unpublished summary", "new-hash", "new-digest"
	require.NoError(t, harness.store.CheckpointMapChunk(t.Context(), checkpoint, "sha256:new", 2))
	selected, err := harness.store.SelectChunkEventsToPublish(t.Context(), 100)
	require.NoError(t, err)
	require.Empty(t, selected)
	var after string
	require.NoError(t, harness.pool.QueryRow(t.Context(), `SELECT jsonb_agg(to_jsonb(c) ORDER BY c.id)::text FROM event_chunks c`).Scan(&after))
	require.Equal(t, before, after)
	chunks, err := harness.store.ListSummaryRecords(t.Context(), id)
	require.NoError(t, err)
	require.Equal(t, "published summary", chunks[0].Summary)
	harness.publish(t, id, "sha256:new", 2, []SummaryRecord{checkpoint})
	var remaining int
	require.NoError(t, harness.pool.QueryRow(t.Context(), `SELECT count(*) FROM session_ingest_map_checkpoints`).Scan(&remaining))
	require.Zero(t, remaining)
}

func TestMapCheckpointRejectsStaleDeletedExcludedAndForeignWork(t *testing.T) {
	for _, state := range []string{"revision", "epoch", "deleted", "excluded", "owner", "client", "missing"} {
		t.Run(state, func(t *testing.T) {
			store, pool := newTestStore(t)
			id := Identity{Owner: "a@x", Client: "codex", SessionID: "checkpoint-guard"}
			seedSummaryCandidate(t, store, id, "sha256:v1", 1)
			chunk := SummaryRecord{Identity: id, RecordStart: 0, RecordEnd: 1, ContentHash: "hash", MapDigest: "digest", Summary: "summary", ModelID: "map-model"}
			revision, epoch := "sha256:v1", int64(1)
			switch state {
			case "revision":
				revision = "sha256:stale"
			case "epoch":
				epoch++
			case "owner":
				chunk.Owner = "other@x"
			case "client":
				chunk.Client = "other"
			case "missing":
				chunk.SessionID = "missing"
			case "deleted", "excluded":
				column, value := "deleted", "true"
				if state == "excluded" {
					column, value = "excluded_at", "now()"
				}
				_, err := pool.Exec(t.Context(), fmt.Sprintf("UPDATE session_ingest_candidates SET %s=%s", column, value))
				require.NoError(t, err)
			}
			require.ErrorIs(t, store.CheckpointMapChunk(t.Context(), chunk, revision, epoch), errCASStale)
			var count int
			require.NoError(t, pool.QueryRow(t.Context(), `SELECT count(*) FROM session_ingest_map_checkpoints`).Scan(&count))
			require.Zero(t, count)
		})
	}
}

func TestMapCheckpointExclusionClearsProgress(t *testing.T) {
	store, pool := newTestStore(t)
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "checkpoint-excluded"}
	head := change(1, id.SessionID, id.Owner, id.Client)
	_, err := store.UpsertChanges(t.Context(), []SessionChange{head}, admitAll)
	require.NoError(t, err)
	chunk := SummaryRecord{Identity: id, RecordStart: 0, RecordEnd: 1, ContentHash: "hash", MapDigest: "digest", Summary: "summary", ModelID: "map-model"}
	require.NoError(t, store.CheckpointMapChunk(t.Context(), chunk, head.Revision, head.RewriteEpoch))
	head.ChangeSeq++
	head.Deleted = true
	_, err = store.UpsertChanges(t.Context(), []SessionChange{head}, admitAll)
	require.NoError(t, err)
	var count int
	require.NoError(t, pool.QueryRow(t.Context(), `SELECT count(*) FROM session_ingest_map_checkpoints`).Scan(&count))
	require.Zero(t, count)
	require.ErrorIs(t, store.CheckpointMapChunk(t.Context(), chunk, head.Revision, head.RewriteEpoch), errCASStale)
}

func TestSummaryCacheGenerationChangesOnlyMapIdentity(t *testing.T) {
	require.Equal(t, "sessioningest-summary-input-v1", summaryInputVersion)
	require.Equal(t, "sessioningest-map-input-v2", mapInputVersion)
}
