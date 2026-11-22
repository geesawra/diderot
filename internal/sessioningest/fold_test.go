package sessioningest

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/stretchr/testify/require"

	"github.com/bluesky-social/diderot/internal/promptcatalog"
	"github.com/bluesky-social/diderot/internal/sessionsummary"
)

// growingRecordBytes and growingChunkBytes put exactly ten records in each
// chunk: ten 99-byte lines and their nine separators fill 999 bytes.
const (
	growingRecordBytes = 99
	growingChunkBytes  = 1000
)

// growingDocument is a transcript of count fixed-size records; record i
// carries the marker rec-<i>.
func growingDocument(id Identity, count int) *sessionDocument {
	doc := docFor(id, "sha256:v1")
	doc.RecordCount = count
	doc.Transcript = nil
	doc.RecordTimes = nil
	for index := range count {
		if index > 0 {
			doc.Transcript = append(doc.Transcript, '\n')
		}
		line := fmt.Appendf(nil, `{"index":%d,"role":"user","content":"rec-%d`, index, index)
		line = append(line, bytes.Repeat([]byte("x"), growingRecordBytes-len(line)-2)...)
		line = append(line, `"}`...)
		doc.Transcript = append(doc.Transcript, line...)
		doc.RecordTimes = append(doc.RecordTimes, time.Date(2026, 7, 24, 18, 0, index, 0, time.UTC))
	}
	return doc
}

var recordMarker = regexp.MustCompile(`rec-(\d+)x`)

// recordingModel is a fake provider that answers map calls with a summary
// naming the records the chunk held, answers reduce calls with a narrative,
// and records every request it was sent.
type recordingModel struct {
	mu       sync.Mutex
	maps     []string // user text of each map call
	reduces  []string // user text of each reduce call
	bytes    int      // user text bytes across all calls
	failOnce map[string]bool
	// narratives numbers reduce outputs across resets.
	narratives int
}

func (m *recordingModel) CreateMessage(_ context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
	text := input.Messages[0].Content[0].OfText.Text
	m.mu.Lock()
	defer m.mu.Unlock()
	m.bytes += len(text)
	if input.ToolChoice.OfTool.Name == "record_session_summary" {
		m.reduces = append(m.reduces, text)
		m.narratives++
		return stubToolOutput("record_session_summary", fmt.Sprintf(
			`{"evidence":["e"],"narrative":"narrative %d","embedding_text":"embedding %d","context_hint":"hint %d","tags":{"projects":["p"]}}`,
			m.narratives, m.narratives, m.narratives)), nil
	}
	markers := recordMarker.FindAllStringSubmatch(text, -1)
	first, last := markers[0][1], markers[len(markers)-1][1]
	if m.failOnce[first] {
		delete(m.failOnce, first)
		return nil, errors.New("map failed")
	}
	m.maps = append(m.maps, first+"-"+last)
	return stubToolOutput("record_chunk_summary", fmt.Sprintf(
		`{"evidence":["rec-%s"],"summary":"summary of rec-%s to rec-%s"}`, first, first, last)), nil
}

func (m *recordingModel) reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.maps, m.reduces, m.bytes = nil, nil, 0
}

type growingSession struct {
	test       *testing.T
	store      *Store
	id         Identity
	model      *recordingModel
	release    *staticPromptRelease
	summarizer Summarizer
	pass       int
}

func newGrowingSession(test *testing.T, name string) *growingSession {
	store, _ := newTestStore(test)
	model := &recordingModel{failOnce: map[string]bool{}}
	release := &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}
	return &growingSession{
		test: test, store: store, id: Identity{Owner: "a@x", Client: "codex", SessionID: name},
		model: model, release: release,
		summarizer: NewSessionsummarySummarizer(store, release, sessionsummary.OverMessages(model), "map-model", "reduce-model", growingChunkBytes, 1),
	}
}

// summarize runs one pass over the session's first records records as a new
// revision, without publishing it.
func (g *growingSession) summarize(records int) (SummaryResult, *sessionDocument, string, error) {
	g.test.Helper()
	g.pass++
	revision := fmt.Sprintf("sha256:v%d", g.pass)
	seedSummaryCandidate(g.test, g.store, g.id, revision, 1)
	doc := growingDocument(g.id, records)
	result, err := g.summarizer.Summarize(g.test.Context(), doc, g.id, revision, 1)
	return result, doc, revision, err
}

// publish stores a pass's result the way PublishSession does.
func (g *growingSession) publish(result SummaryResult, doc *sessionDocument, revision string) {
	g.test.Helper()
	chunks := append([]SummaryRecord(nil), result.Chunks...)
	for index := range chunks {
		chunks[index].OccurredAt = doc.RecordTimes[chunks[index].RecordEnd-1]
	}
	tx, err := g.store.pool.Begin(g.test.Context())
	require.NoError(g.test, err)
	defer func() { _ = tx.Rollback(g.test.Context()) }()
	require.NoError(g.test, g.store.ReplaceSummaryTx(g.test.Context(), tx, SessionSummaryDoc{
		Identity: g.id, Revision: revision, RewriteEpoch: 1,
		InputDigest: result.InputDigest, Narrative: result.Narrative,
		EmbeddingText: result.EmbeddingText, ContextHint: result.ContextHint,
		Tags: result.Tags, ModelID: result.ModelID, RecordCount: doc.RecordCount,
	}, chunks))
	require.NoError(g.test, tx.Commit(g.test.Context()))
}

// grow summarizes and publishes the session at records records.
func (g *growingSession) grow(records int) SummaryResult {
	g.test.Helper()
	result, doc, revision, err := g.summarize(records)
	require.NoError(g.test, err)
	g.publish(result, doc, revision)
	return result
}

func TestGrowingSessionSummarizesOnlyTheOpenChunk(test *testing.T) {
	session := newGrowingSession(test, "growing-open-chunk")

	// 25 records: chunks [0,10) [10,20) [20,25).
	first := session.grow(25)
	require.Equal(test, []string{"0-9", "10-19", "20-24"}, session.model.maps)
	require.Len(test, session.model.reduces, 1)
	require.Equal(test, 3, first.ReduceEntries)
	session.model.reset()

	// 40 records: [20,25) grew to [20,30), and [30,40) is new.
	second := session.grow(40)
	require.Equal(test, []string{"20-29", "30-39"}, session.model.maps,
		"only the grown last chunk and the new chunk are mapped")
	require.Len(test, session.model.reduces, 1)
	reduce := session.model.reduces[0]
	require.Contains(test, reduce, `<chunk index="1-3">`)
	require.Contains(test, reduce, "<summary>narrative 1</summary>", "the earlier session summary stands in for its chunks")
	require.Contains(test, reduce, "hint 1")
	require.Contains(test, reduce, "project:p", "the earlier tags reach the reduce that replaces them")
	require.Contains(test, reduce, `<chunk index="3">`)
	require.Contains(test, reduce, "summary of rec-20 to rec-29")
	require.Contains(test, reduce, `<chunk index="4">`)
	require.NotContains(test, reduce, "summary of rec-0 to rec-9", "a chunk summary the earlier reduce read was sent again")
	require.NotContains(test, reduce, "summary of rec-10 to rec-19", "a chunk summary the earlier reduce read was sent again")
	require.NotContains(test, reduce, "summary of rec-20 to rec-24", "the superseded summary of the open chunk was sent")
	require.Equal(test, 3, second.ReduceEntries)
	require.Len(test, second.Chunks, 4, "every chunk is still published for retrieval")
	require.Equal(test, first.Chunks[:2], second.Chunks[:2], "closed chunks keep their summaries")
	session.model.reset()

	// 50 records: [30,40) was closed exactly at the last pass, so only the
	// new chunk [40,50) is read.
	session.grow(50)
	require.Equal(test, []string{"40-49"}, session.model.maps)
	reduce = session.model.reduces[0]
	require.Contains(test, reduce, `<chunk index="1-4">`)
	require.Contains(test, reduce, "<summary>narrative 2</summary>")
	require.NotContains(test, reduce, `<chunk index="4">`)
	require.Contains(test, reduce, `<chunk index="5">`)
}

func TestGrowingSessionCostIsLinearInLength(test *testing.T) {
	cost := func(passes int) (total, largestReduce int) {
		session := newGrowingSession(test, fmt.Sprintf("growing-linear-%d", passes))
		for pass := 1; pass <= passes; pass++ {
			session.model.mu.Lock()
			before := session.model.bytes
			session.model.mu.Unlock()
			// Each pass appends 15 records: one and a half chunks, so the
			// open chunk is regrown every other pass.
			session.grow(pass * 15)
			total += session.model.bytes - before
			last := session.model.reduces[len(session.model.reduces)-1]
			largestReduce = max(largestReduce, len(last))
		}
		return total, largestReduce
	}
	short, shortReduce := cost(10)
	long, longReduce := cost(100)
	ratio := float64(long) / float64(short)
	test.Logf("bytes sent: 10 passes %d, 100 passes %d (x%.1f); largest reduce input %d vs %d", short, long, ratio, shortReduce, longReduce)
	require.Less(test, ratio, 12.0, "a session ten times longer must cost about ten times more")
	require.Greater(test, ratio, 8.0)
	require.Less(test, longReduce, shortReduce+200, "a reduce's input must not grow with the session")
}

func TestGrowingSessionFinalPassCompletesTheSummary(test *testing.T) {
	session := newGrowingSession(test, "growing-final")
	session.grow(35)
	session.model.reset()
	// The session stops at 37 records: the last pass maps only the open
	// chunk and the published summary covers every record.
	final := session.grow(37)
	require.Equal(test, []string{"30-36"}, session.model.maps)
	require.Equal(test, 37, final.Chunks[len(final.Chunks)-1].RecordEnd)
	published, err := session.store.ListSummaryRecords(test.Context(), session.id)
	require.NoError(test, err)
	require.Len(test, published, 4)
	require.Equal(test, 37, published[3].RecordEnd)
	earlier, ok, err := session.store.LoadEarlierSummary(test.Context(), session.id)
	require.NoError(test, err)
	require.True(test, ok)
	require.Equal(test, 37, earlier.RecordCount)
	require.Equal(test, final.Narrative, earlier.Narrative)
}

func TestGrowingSessionMapFailureRetriesOnlyThatChunk(test *testing.T) {
	session := newGrowingSession(test, "growing-retry")
	session.grow(25)
	session.model.reset()
	session.model.failOnce["30"] = true
	_, _, _, err := session.summarize(40)
	require.Error(test, err)
	require.Empty(test, session.model.reduces)
	mappedBefore := append([]string(nil), session.model.maps...)
	require.Contains(test, mappedBefore, "20-29")
	session.model.reset()

	result, doc, revision, err := session.summarize(40)
	require.NoError(test, err)
	require.Equal(test, []string{"30-39"}, session.model.maps, "the retry maps only the chunk that failed")
	require.Len(test, session.model.reduces, 1)
	require.Contains(test, session.model.reduces[0], `<chunk index="1-3">`)
	session.publish(result, doc, revision)
}

func TestFoldingFallsBackToEveryChunk(test *testing.T) {
	for _, scenario := range []struct {
		name   string
		change func(*growingSession, *sessionDocument)
		maps   []string
	}{
		{
			name: "earlier content changed",
			change: func(_ *growingSession, doc *sessionDocument) {
				doc.Transcript = bytes.Replace(doc.Transcript, []byte(`"content":"rec-5x`), []byte(`"content":"rec-5y`), 1)
			},
			maps: []string{"0-9", "20-29", "30-39"},
		},
		{
			name: "reduce prompt changed",
			change: func(session *growingSession, _ *sessionDocument) {
				bundle, err := session.release.bundle.WithBody(promptcatalog.SummarySessionReduce, "A different reduce prompt.")
				require.NoError(session.test, err)
				session.release.bundle = bundle
			},
			maps: []string{"20-29", "30-39"},
		},
		{
			name: "session rewound",
			change: func(session *growingSession, doc *sessionDocument) {
				*doc = *growingDocument(session.id, 18)
			},
			maps: []string{"10-17"},
		},
	} {
		test.Run(scenario.name, func(test *testing.T) {
			session := newGrowingSession(test, "growing-fallback")
			session.grow(25)
			session.model.reset()
			session.pass++
			revision := fmt.Sprintf("sha256:v%d", session.pass)
			seedSummaryCandidate(test, session.store, session.id, revision, 1)
			doc := growingDocument(session.id, 40)
			scenario.change(session, doc)
			result, err := session.summarizer.Summarize(test.Context(), doc, session.id, revision, 1)
			require.NoError(test, err)
			require.Equal(test, scenario.maps, session.model.maps)
			require.Len(test, session.model.reduces, 1)
			reduce := session.model.reduces[0]
			require.NotContains(test, reduce, "narrative 1", "an earlier summary that no longer describes the session was folded in")
			require.Contains(test, reduce, `<chunk index="1">`)
			require.Equal(test, len(result.Chunks), result.ReduceEntries)
		})
	}
}

// A summary published before this change came from one reduce over every
// chunk, under the same input digest. The first pass after deploy folds it
// like any other; a row whose inputs cannot be proven reduces in full once.
func TestFoldingAfterDeployUsesSummariesPublishedBeforeIt(test *testing.T) {
	test.Run("digest-bearing summary", func(test *testing.T) {
		session := newGrowingSession(test, "deploy-digest")
		doc := growingDocument(session.id, 25)
		bundle, err := sessionsummary.PromptBundleFromCatalog(session.release.bundle.Render(promptcatalog.Values()))
		require.NoError(test, err)
		mapOpts := sessionsummary.CallOptions{ModelID: "map-model", MaxTokens: 1024}
		reduceOpts := sessionsummary.CallOptions{ModelID: "reduce-model", MaxTokens: 4096}
		chunks := sessionsummary.ChunkTranscript(doc.Transcript, growingChunkBytes)
		mapDigest := summaryMapInputDigest(bundle, "session_summary", mapOpts)
		require.NoError(test, session.store.UpsertSessionSummary(test.Context(), SessionSummaryDoc{
			Identity: session.id, Revision: "sha256:before", RewriteEpoch: 1,
			InputDigest: summaryInputDigest(bundle, "session_summary", mapOpts, reduceOpts, chunks),
			Narrative:   "summary written before the deploy", EmbeddingText: "before embedding",
			ContextHint: "before hint", ModelID: "map-model", RecordCount: 25,
		}))
		for _, chunk := range chunks {
			require.NoError(test, session.store.UpsertSummaryRecord(test.Context(), SummaryRecord{
				Identity: session.id, RecordStart: chunk.RecordStart, RecordEnd: chunk.RecordEnd,
				ContentHash: shaHex(chunk.Bytes), MapDigest: mapDigest, Evidence: []string{"e"},
				Summary: fmt.Sprintf("old summary %d", chunk.RecordStart), ModelID: "map-model",
			}))
		}
		session.pass = 1
		session.grow(40)
		require.Equal(test, []string{"20-29", "30-39"}, session.model.maps)
		reduce := session.model.reduces[0]
		require.Contains(test, reduce, "summary written before the deploy")
		require.NotContains(test, reduce, "old summary 0")
		require.NotContains(test, reduce, "old summary 10")
	})
	test.Run("summary without a digest", func(test *testing.T) {
		session := newGrowingSession(test, "deploy-legacy")
		session.grow(25)
		_, err := session.store.pool.Exec(test.Context(), `UPDATE session_ingest_summaries SET input_digest=''`)
		require.NoError(test, err)
		session.model.reset()
		result := session.grow(40)
		require.Equal(test, []string{"20-29", "30-39"}, session.model.maps)
		require.NotContains(test, session.model.reduces[0], "narrative 1")
		require.Equal(test, 4, result.ReduceEntries)
		session.model.reset()
		session.grow(50)
		require.Contains(test, session.model.reduces[0], `<chunk index="1-4">`, "the next pass folds again")
	})
}

// An earlier summary of one chunk is labelled as a range all the same, and a
// summary stored with its fields inside its narrative is folded in repaired.
func TestFoldingLabelsAndRepairsTheEarlierSummary(test *testing.T) {
	session := newGrowingSession(test, "growing-one-chunk")
	session.grow(5)
	_, err := session.store.pool.Exec(test.Context(), `
		UPDATE session_ingest_summaries
		   SET narrative = narrative || '</narrative>' || chr(10) || '<parameter name="embedding_text">swallowed embedding',
		       embedding_text = NULL`)
	require.NoError(test, err)
	session.model.reset()
	session.grow(15)
	reduce := session.model.reduces[0]
	require.Contains(test, reduce, `<chunk index="1-1">`)
	require.Contains(test, reduce, "<summary>narrative 1</summary>", "the earlier narrative is folded in without the markup")
	require.Contains(test, reduce, "swallowed embedding", "and the search text it held goes along as evidence")
	require.NotContains(test, reduce, "</narrative>")
}

func TestRecordPrefixMatchesEarlierChunking(test *testing.T) {
	full := growingDocument(Identity{Owner: "a@x", Client: "codex", SessionID: "prefix"}, 37).Transcript
	earlier := growingDocument(Identity{Owner: "a@x", Client: "codex", SessionID: "prefix"}, 25).Transcript
	prefix, ok := sessionsummary.RecordPrefix(full, 25)
	require.True(test, ok)
	require.Equal(test, earlier, prefix)
	require.Equal(test, sessionsummary.ChunkTranscript(earlier, growingChunkBytes), sessionsummary.ChunkTranscript(prefix, growingChunkBytes))
	require.True(test, strings.HasSuffix(string(prefix), `"}`))
}
