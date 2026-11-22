package sessionsummary

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/bluesky-social/diderot/internal/llmclient"
)

const gatewaySummaryMaxResponseBytes int64 = 4 << 20

// GatewayClientConfig configures the native Messages transport.
type GatewayClientConfig struct {
	// BaseURL is the Anthropic Messages endpoint, with or without /v1.
	BaseURL string
	APIKey  string
	// Auth selects bearer (a model gateway, the default) or x-api-key (the
	// public Anthropic API).
	Auth       llmclient.AuthStyle
	HTTPClient *http.Client
	// Headers are extra headers every request carries: the provider's.
	Headers http.Header
}

// GatewayClient sends native Anthropic Messages requests.
type GatewayClient struct {
	client *llmclient.Client
}

// NewGatewayClient validates cfg without making a network request.
func NewGatewayClient(cfg GatewayClientConfig) (*GatewayClient, error) {
	baseURL := strings.TrimSpace(cfg.BaseURL)
	baseURL = strings.TrimRight(baseURL, "/")
	if baseURL != "" && !strings.HasSuffix(baseURL, "/v1") {
		baseURL += "/v1"
	}
	client, err := llmclient.New(llmclient.Config{
		BaseURL: baseURL, APIKey: cfg.APIKey, Auth: cfg.Auth, HTTPClient: cfg.HTTPClient,
		Headers: withHeader(cfg.Headers, "Anthropic-Version", "2023-06-01"),
	})
	if err != nil {
		return nil, fmt.Errorf("sessionsummary gateway: %w", err)
	}
	return &GatewayClient{client: client}, nil
}

func (g *GatewayClient) CreateMessage(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
	raw, err := g.client.PostJSON(ctx, "/messages", params, gatewaySummaryMaxResponseBytes)
	if err != nil {
		return nil, fmt.Errorf("sessionsummary gateway: request: %w", err)
	}

	var decoded anthropic.Message
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("sessionsummary gateway: decode response: %w", err)
	}
	return &decoded, nil
}

// CallTool makes the client a ToolClient on Messages.
func (g *GatewayClient) CallTool(ctx context.Context, request ToolRequest) (json.RawMessage, Usage, error) {
	return OverMessages(g).CallTool(ctx, request)
}

var (
	_ MessageClient = (*GatewayClient)(nil)
	_ ToolClient    = (*GatewayClient)(nil)
)

// withHeader is headers with name set to value.
func withHeader(headers http.Header, name, value string) http.Header {
	out := headers.Clone()
	if out == nil {
		out = http.Header{}
	}
	out.Set(name, value)
	return out
}
