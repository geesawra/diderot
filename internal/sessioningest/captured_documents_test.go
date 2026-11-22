package sessioningest

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"github.com/anthropics/anthropic-sdk-go/v2"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/bluesky-social/diderot/internal/promptcatalog"
	"github.com/bluesky-social/diderot/internal/sessiondecode"
	"github.com/bluesky-social/diderot/internal/sessionsummary"
)

func capturedDocumentPaths(test *testing.T) []string {
	test.Helper()
	paths := filepath.SplitList(os.Getenv("DIDEROT_SESSION_DOC_FIXTURES"))
	if len(paths) == 0 {
		test.Skip("set DIDEROT_SESSION_DOC_FIXTURES to private cleaned session .ndjson.gz files")
	}
	return paths
}

func readCapturedDocument(test *testing.T, filename string) *sessionDocument {
	test.Helper()
	body := readCapturedDocumentBytes(test, filename)
	var envelope sessiondecode.BskyEnvelope
	if err := json.Unmarshal(bytes.SplitN(body, []byte("\n"), 2)[0], &envelope); err != nil {
		test.Fatal(err)
	}
	document, err := parseSessionDocument(body, Identity{Owner: envelope.Owner, Client: envelope.Source, SessionID: envelope.SessionID}, envelope.Revision, 256<<20)
	if err != nil {
		test.Fatal(err)
	}
	return document
}

func readCapturedDocumentBytes(test *testing.T, filename string) []byte {
	test.Helper()
	file, err := os.Open(filename)
	if err != nil {
		test.Fatal(err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			test.Error(err)
		}
	}()
	reader, err := gzip.NewReader(file)
	if err != nil {
		test.Fatal(err)
	}
	defer func() {
		if err := reader.Close(); err != nil {
			test.Error(err)
		}
	}()
	body, err := io.ReadAll(io.LimitReader(reader, (256<<20)+1))
	if err != nil {
		test.Fatal(err)
	}
	if len(body) > 256<<20 {
		test.Fatal("captured document exceeds production cap")
	}
	return body
}

func TestCapturedSessionDocumentChunks(test *testing.T) {
	for _, filename := range capturedDocumentPaths(test) {
		test.Run(filepath.Base(filename), func(test *testing.T) {
			document := readCapturedDocument(test, filename)
			assertLegacySessionChunks(test, readCapturedDocumentBytes(test, filename), document)
			chunks := (&sessionsummarySummarizer{}).chunkRecords(document, 0)
			next, errorNotes, largestChunk := 0, 0, 0
			for _, chunk := range chunks {
				if chunk.RecordStart != next || chunk.RecordEnd <= next || chunk.RecordEnd > document.RecordCount {
					test.Fatalf("chunk range [%d,%d) breaks coverage at record %d", chunk.RecordStart, chunk.RecordEnd, next)
				}
				lines := bytes.Split(bytes.TrimSpace(chunk.Bytes), []byte("\n"))
				if len(lines) != chunk.RecordEnd-chunk.RecordStart {
					test.Fatal("chunk range does not match its record payload")
				}
				for _, line := range lines {
					var record sessiondecode.Record
					if err := json.Unmarshal(line, &record); err != nil {
						test.Fatal(err)
					}
					if record.Phase == "outcome" || record.Phase == "request-incomplete" || record.Phase == "response-error" || record.Phase == "normalization" {
						errorNotes++
					}
				}
				largestChunk = max(largestChunk, len(chunk.Bytes))
				next = chunk.RecordEnd
			}
			if next != document.RecordCount {
				test.Fatal("missing session records")
			}
			test.Logf("records=%d chunks=%d error_notes=%d largest_chunk_bytes=%d", document.RecordCount, len(chunks), errorNotes, largestChunk)
		})
	}
}

type capturedSummaryClient struct {
	mutex       sync.Mutex
	mapInputs   []string
	reduceCalls int
}

func (client *capturedSummaryClient) CreateMessage(ctx context.Context, input anthropic.MessageNewParams) (*anthropic.Message, error) {
	client.mutex.Lock()
	defer client.mutex.Unlock()
	switch input.ToolChoice.OfTool.Name {
	case "record_chunk_summary":
		client.mapInputs = append(client.mapInputs, input.Messages[0].Content[0].OfText.Text)
	case "record_session_summary":
		client.reduceCalls++
	}
	return (cycleMessages{}).CreateMessage(ctx, input)
}

func TestCapturedSessionDocumentSummarization(test *testing.T) {
	for _, filename := range capturedDocumentPaths(test) {
		test.Run(filepath.Base(filename), func(test *testing.T) {
			document := readCapturedDocument(test, filename)
			store, pool := newTestStore(test)
			prompts := promptcatalog.NewStore(pool)
			if _, err := prompts.EnsureSeed(test.Context(), promptcatalog.MustEmbedded(), "test:captured-session"); err != nil {
				test.Fatal(err)
			}
			client := &capturedSummaryClient{}
			summarizer := NewSessionsummarySummarizer(store, prompts, sessionsummary.OverMessages(client), "map-model", "reduce-model", 0, 0)
			identity := Identity{Owner: document.Bsky.Owner, Client: document.Bsky.Source, SessionID: document.Bsky.SessionID}
			seedSummaryCandidate(test, store, identity, document.Bsky.Revision, document.Bsky.RewriteEpoch)
			result, err := summarizer.Summarize(test.Context(), document, identity, document.Bsky.Revision, document.Bsky.RewriteEpoch)
			if err != nil {
				test.Fatal(err)
			}
			chunks := (&sessionsummarySummarizer{}).chunkRecords(document, 0)
			if len(result.Chunks) != len(chunks) || len(client.mapInputs) != len(chunks) || client.reduceCalls != 1 {
				test.Fatalf("map/reduce coverage mismatch: chunks=%d results=%d maps=%d reduces=%d", len(chunks), len(result.Chunks), len(client.mapInputs), client.reduceCalls)
			}
			for index, chunk := range chunks {
				expected := fmt.Sprintf("<chunk index=\"%d\" of=\"%d\">\n%s\n</chunk>", index+1, len(chunks), chunk.Bytes)
				found := false
				for _, input := range client.mapInputs {
					found = found || input == expected
				}
				if !found {
					test.Fatalf("records [%d,%d) did not reach a map request", chunk.RecordStart, chunk.RecordEnd)
				}
			}
			test.Logf("records=%d maps=%d reduces=%d; model calls are stubbed", document.RecordCount, len(client.mapInputs), client.reduceCalls)
		})
	}
}
