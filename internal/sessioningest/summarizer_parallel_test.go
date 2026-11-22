package sessioningest

import (
	"context"
	"errors"
	"fmt"
	"github.com/anthropics/anthropic-sdk-go/v2"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/bluesky-social/diderot/internal/promptcatalog"
	"github.com/bluesky-social/diderot/internal/sessionsummary"
)

type summaryMessageFunc func(context.Context, anthropic.MessageNewParams) (*anthropic.Message, error)

func (client summaryMessageFunc) CreateMessage(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
	return client(ctx, input)
}

func summaryChunkDocument(id Identity, count int) *sessionDocument {
	doc := docFor(id, "sha256:v1")
	doc.RecordCount = count
	doc.Transcript = nil
	for index := range count {
		if index > 0 {
			doc.Transcript = append(doc.Transcript, '\n')
		}
		doc.Transcript = fmt.Appendf(doc.Transcript, `{"index":%d,"role":"user","content":"chunk-%d"}`, index, index+1)
	}
	return doc
}

func summaryChunkIndex(input anthropic.MessageNewParams) (int, error) {
	text := input.Messages[0].Content[0].OfText.Text
	var index, total string
	if _, err := fmt.Sscanf(text, `<chunk index=%q of=%q>`, &index, &total); err != nil {
		return 0, err
	}
	return strconv.Atoi(index)
}

func TestSessionsummarySummarizer_ParallelChunksKeepOrder(t *testing.T) {
	store, _ := newTestStore(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	entered := make(chan int, 3)
	finished := make(chan int, 3)
	release := []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
	reduceInput := make(chan string, 1)
	client := summaryMessageFunc(func(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
		if string(input.Model) == "reduce-model" {
			reduceInput <- input.Messages[0].Content[0].OfText.Text
			return (cycleMessages{}).CreateMessage(ctx, input)
		}
		index, err := summaryChunkIndex(input)
		if err != nil {
			return nil, err
		}
		entered <- index
		select {
		case <-release[index-1]:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		finished <- index
		return stubToolOutput("record_chunk_summary", fmt.Sprintf(`{"evidence":["chunk-%d"],"summary":"chunk-%d"}`, index, index)), nil
	})
	summarizer := NewSessionsummarySummarizer(store, &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 3)
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "parallel-order"}
	seedSummaryCandidate(t, store, id, "sha256:v1", 1)
	var result SummaryResult
	var resultErr error
	done := make(chan struct{})
	go func() {
		result, resultErr = summarizer.Summarize(ctx, summaryChunkDocument(id, 3), id, "sha256:v1", 1)
		close(done)
	}()
	for range 3 {
		select {
		case <-entered:
		case <-ctx.Done():
			t.Fatal("chunks did not run concurrently")
		}
	}
	for _, index := range []int{3, 2, 1} {
		close(release[index-1])
		select {
		case got := <-finished:
			require.Equal(t, index, got)
		case <-ctx.Done():
			t.Fatal("released chunk did not complete")
		}
	}
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatal("summary did not complete")
	}
	require.NoError(t, resultErr)
	require.Len(t, result.Chunks, 3)
	require.EqualValues(t, 40, result.InputTokens)
	require.EqualValues(t, 20, result.OutputTokens)
	reduced := <-reduceInput
	previous := -1
	for index, chunk := range result.Chunks {
		require.Equal(t, index, chunk.RecordStart)
		require.Equal(t, index+1, chunk.RecordEnd)
		require.Equal(t, fmt.Sprintf("chunk-%d", index+1), chunk.Summary)
		position := strings.Index(reduced, "<summary>"+chunk.Summary+"</summary>")
		require.Greater(t, position, previous)
		previous = position
	}
}

func TestSessionsummarySummarizer_ConcurrencySharedAcrossSessions(t *testing.T) {
	store, _ := newTestStore(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var active, peak, calls atomic.Int32
	entered := make(chan struct{}, 20)
	release := make(chan struct{})
	client := summaryMessageFunc(func(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
		current := active.Add(1)
		defer active.Add(-1)
		for previous := peak.Load(); current > previous; previous = peak.Load() {
			if peak.CompareAndSwap(previous, current) {
				break
			}
		}
		calls.Add(1)
		entered <- struct{}{}
		select {
		case <-release:
			return (cycleMessages{}).CreateMessage(ctx, input)
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
	summarizer := NewSessionsummarySummarizer(store, &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 2)
	done := make(chan error, 2)
	for _, sessionID := range []string{"parallel-one", "parallel-two"} {
		id := Identity{Owner: "a@x", Client: "codex", SessionID: sessionID}
		seedSummaryCandidate(t, store, id, "sha256:v1", 1)
		go func() {
			_, err := summarizer.Summarize(ctx, summaryChunkDocument(id, 6), id, "sha256:v1", 1)
			done <- err
		}()
	}
	for range 2 {
		select {
		case <-entered:
		case <-ctx.Done():
			t.Fatal("model concurrency was not reached")
		}
	}
	select {
	case <-entered:
		t.Fatal("shared model-call limit exceeded")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	for range 2 {
		select {
		case err := <-done:
			require.NoError(t, err)
		case <-ctx.Done():
			t.Fatal("summaries did not complete")
		}
	}
	require.EqualValues(t, 2, peak.Load())
	require.EqualValues(t, 14, calls.Load())
}

func TestSessionsummarySummarizer_MapFailureCancelsSiblings(t *testing.T) {
	store, _ := newTestStore(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	failed := errors.New("map failed")
	entered := make(chan struct{}, 3)
	release := make(chan struct{})
	var canceled, reductions atomic.Int32
	client := summaryMessageFunc(func(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
		if string(input.Model) == "reduce-model" {
			reductions.Add(1)
			return nil, errors.New("unexpected reduction")
		}
		index, err := summaryChunkIndex(input)
		if err != nil {
			return nil, err
		}
		entered <- struct{}{}
		if index == 1 {
			select {
			case <-release:
				return nil, failed
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		<-ctx.Done()
		canceled.Add(1)
		return nil, ctx.Err()
	})
	summarizer := NewSessionsummarySummarizer(store, &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 3).(*sessionsummarySummarizer)
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "parallel-failure"}
	seedSummaryCandidate(t, store, id, "sha256:v1", 1)
	done := make(chan error, 1)
	go func() {
		_, err := summarizer.Summarize(ctx, summaryChunkDocument(id, 6), id, "sha256:v1", 1)
		done <- err
	}()
	for range 3 {
		select {
		case <-entered:
		case <-ctx.Done():
			t.Fatal("map siblings did not start")
		}
	}
	close(release)
	select {
	case err := <-done:
		require.ErrorIs(t, err, failed)
	case <-ctx.Done():
		t.Fatal("map failure did not cancel siblings")
	}
	require.EqualValues(t, 2, canceled.Load())
	require.Zero(t, reductions.Load())
	require.True(t, summarizer.modelCalls.TryAcquire(3), "failed work leaked model slots")
	summarizer.modelCalls.Release(3)
}

func TestSessionsummarySummarizer_CanceledWaiterDoesNotInfer(t *testing.T) {
	store, _ := newTestStore(t)
	client := &rejectingMessages{}
	summarizer := NewSessionsummarySummarizer(store, &staticPromptRelease{bundle: promptcatalog.MustEmbedded()}, sessionsummary.OverMessages(client), "map-model", "reduce-model", 1, 1).(*sessionsummarySummarizer)
	require.NoError(t, summarizer.modelCalls.Acquire(context.Background(), 1))
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	id := Identity{Owner: "a@x", Client: "codex", SessionID: "parallel-canceled"}
	_, err := summarizer.Summarize(ctx, summaryChunkDocument(id, 3), id, "sha256:v1", 1)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Zero(t, client.calls.Load())
	summarizer.modelCalls.Release(1)
	require.True(t, summarizer.modelCalls.TryAcquire(1), "canceled waiter leaked a model slot")
	summarizer.modelCalls.Release(1)
}
