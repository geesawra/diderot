package ask

import (
	"context"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go/v2"
	"github.com/cloudwego/eino-ext/components/model/agenticclaude"
	"github.com/cloudwego/eino-ext/components/model/agenticopenai"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	openairesponses "github.com/openai/openai-go/v3/responses"

	"github.com/bluesky-social/diderot/internal/llmclient"
	"github.com/bluesky-social/diderot/internal/modelregistry"
)

// ModelConfig is the Diderot-owned construction boundary for an Ask model.
// Model is the registry entry; its first agentic protocol selects the Eino
// adapter, wire protocol, and provider-native reasoning encoding. BaseURL,
// APIKey, and Auth are the endpoint configured for that protocol.
type ModelConfig struct {
	BaseURL string
	APIKey  string
	// Auth applies to the Messages protocol only; Responses always sends a
	// bearer token. Empty means bearer.
	Auth            llmclient.AuthStyle
	Model           modelregistry.Model
	Timeout         time.Duration
	MaxTokens       int
	HTTPClient      *http.Client
	ReasoningEffort string
	ServiceTier     string
}

// ConfiguredModel is an AgenticModel plus the resolved protocol information
// needed by protocol-adjacent callers such as the readiness probe.
type ConfiguredModel struct {
	inner    model.AgenticModel
	entry    modelregistry.Model
	protocol modelregistry.Protocol
}

var _ model.AgenticModel = (*ConfiguredModel)(nil)

func (m *ConfiguredModel) Generate(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.AgenticMessage, error) {
	return m.inner.Generate(ctx, input, m.withReasoningEffort(ctx, opts)...)
}

func (m *ConfiguredModel) Stream(ctx context.Context, input []*schema.AgenticMessage, opts ...model.Option) (*schema.StreamReader[*schema.AgenticMessage], error) {
	return m.inner.Stream(ctx, input, m.withReasoningEffort(ctx, opts)...)
}

type reasoningEffortContextKey struct{}

// WithReasoningEffort sets the reasoning effort of every Ask model call made
// with ctx, the root's and its observability child's. AskService sets it
// from its limits as each run starts; empty keeps the model's default.
func WithReasoningEffort(ctx context.Context, effort string) context.Context {
	return context.WithValue(ctx, reasoningEffortContextKey{}, effort)
}

// withReasoningEffort puts ctx's effort ahead of opts, so a request-scoped
// override among them still wins.
func (m *ConfiguredModel) withReasoningEffort(ctx context.Context, opts []model.Option) []model.Option {
	effort, _ := ctx.Value(reasoningEffortContextKey{}).(string)
	if effort == "" {
		return opts
	}
	var option model.Option
	switch m.protocol {
	case modelregistry.ProtocolResponses:
		option = agenticopenai.WithResponsesReasoning(&openairesponses.ReasoningParam{Effort: openairesponses.ReasoningEffort(effort)})
	case modelregistry.ProtocolMessages:
		// The Messages client takes thinking only at construction; per call
		// it is a body field like the effort itself.
		thinking, fields := claudeReasoning(effort)
		fields = maps.Clone(fields)
		if fields == nil {
			fields = map[string]any{}
		}
		fields["thinking"] = *thinking
		option = agenticclaude.WithExtraFields(fields)
	default:
		return opts
	}
	return append([]model.Option{option}, opts...)
}

func (m *ConfiguredModel) ModelID() string { return m.entry.ID }

// Model returns the registry entry the model was built from.
func (m *ConfiguredModel) Model() modelregistry.Model { return m.entry }

func (m *ConfiguredModel) Protocol() modelregistry.Protocol { return m.protocol }

// WithCustomHeaders returns the adapter-specific Eino option for request
// headers. Eino does not define a provider-neutral custom-header option.
func (m *ConfiguredModel) WithCustomHeaders(headers map[string]string) model.Option {
	if m.protocol == modelregistry.ProtocolMessages {
		return agenticclaude.WithCustomHeaders(headers)
	}
	return agenticopenai.WithCustomHeaders(headers)
}

// ValidateModelSelection checks that an Ask model has an agentic protocol and
// accepts the configured effort and service tier. Empty effort and tier mean
// the provider default and are valid for every model.
func ValidateModelSelection(entry modelregistry.Model, effort, serviceTier string) error {
	if _, ok := entry.AgenticProtocol(); !ok {
		return fmt.Errorf("model %q lists no agentic protocol (%s or %s)", entry.ID,
			modelregistry.ProtocolResponses, modelregistry.ProtocolMessages)
	}
	if err := entry.ValidateReasoningEffort(effort); err != nil {
		return err
	}
	return entry.ValidateServiceTier(serviceTier)
}

// NewModel constructs the provider-native Eino client selected by the
// registry entry's first agentic protocol. Construction performs no network
// activity.
func NewModel(ctx context.Context, cfg ModelConfig) (*ConfiguredModel, error) {
	if err := validateModelConfig(cfg); err != nil {
		return nil, err
	}
	protocol, _ := cfg.Model.AgenticProtocol()

	var (
		inner model.AgenticModel
		err   error
	)
	switch protocol {
	case modelregistry.ProtocolResponses:
		if cfg.Auth != "" && cfg.Auth != llmclient.AuthBearer {
			err = fmt.Errorf("the %s protocol only supports bearer auth", protocol)
			break
		}
		inner, err = NewResponsesModel(ctx, ResponsesConfig{
			BaseURL:              cfg.BaseURL,
			APIKey:               cfg.APIKey,
			ModelID:              cfg.Model.ID,
			Timeout:              cfg.Timeout,
			MaxTokens:            cfg.MaxTokens,
			MaxRetries:           responsesDefaultMaxRetries,
			HTTPClient:           cfg.HTTPClient,
			ReasoningEffort:      cfg.ReasoningEffort,
			ServiceTier:          cfg.ServiceTier,
			PromptCacheRetention: cfg.Model.PromptCacheRetention,
		})
	case modelregistry.ProtocolMessages:
		inner, err = newClaudeModel(ctx, cfg)
	default:
		err = fmt.Errorf("unsupported model protocol %q", protocol)
	}
	if err != nil {
		return nil, fmt.Errorf("model %q (%s): %w", cfg.Model.ID, protocol, err)
	}
	return &ConfiguredModel{inner: inner, entry: cfg.Model, protocol: protocol}, nil
}

func validateModelConfig(cfg ModelConfig) error {
	if cfg.Model.ID == "" {
		return fmt.Errorf("model: registry entry is required")
	}
	if err := ValidateModelSelection(cfg.Model, cfg.ReasoningEffort, cfg.ServiceTier); err != nil {
		return err
	}
	if strings.TrimSpace(cfg.BaseURL) == "" {
		return fmt.Errorf("model: BaseURL is required")
	}
	if strings.TrimSpace(cfg.APIKey) == "" {
		return fmt.Errorf("model: APIKey is required")
	}
	if cfg.Timeout < 0 {
		return fmt.Errorf("model: Timeout must be positive")
	}
	if cfg.MaxTokens <= 0 {
		return fmt.Errorf("model: MaxTokens must be positive")
	}
	if _, err := validatedModelBaseURL(cfg.BaseURL); err != nil {
		return err
	}
	return nil
}

func validatedModelBaseURL(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("model: invalid BaseURL")
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("model: BaseURL must use http or https")
	}
	if u.User != nil {
		return nil, fmt.Errorf("model: BaseURL must not contain userinfo")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return nil, fmt.Errorf("model: BaseURL must not contain a query or fragment")
	}
	return u, nil
}

func newClaudeModel(ctx context.Context, cfg ModelConfig) (*agenticclaude.Model, error) {
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = responsesDefaultTimeout
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}
	httpCopy := *httpClient
	httpCopy.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}

	thinking, extraFields := claudeReasoning(cfg.ReasoningEffort)
	baseURL, err := claudeBaseURL(cfg.BaseURL)
	if err != nil {
		return nil, err
	}
	auth, err := llmclient.ParseAuthStyle(string(cfg.Auth))
	if err != nil {
		return nil, fmt.Errorf("messages: %w", err)
	}
	claudeConfig := &agenticclaude.Config{
		BaseURL:        baseURL,
		Model:          cfg.Model.ID,
		MaxTokens:      cfg.MaxTokens,
		HTTPClient:     &httpCopy,
		RequestTimeout: timeout,
		Thinking:       thinking,
		ExtraFields:    extraFields,
	}
	// The Anthropic SDK sends APIKey as x-api-key and AuthToken as a bearer.
	if auth == llmclient.AuthXAPIKey {
		claudeConfig.APIKey = cfg.APIKey
	} else {
		claudeConfig.AuthToken = cfg.APIKey
	}
	mdl, err := agenticclaude.New(ctx, claudeConfig)
	if err != nil {
		return nil, fmt.Errorf("messages: build model: %w", err)
	}
	return mdl, nil
}

func claudeReasoning(effort string) (*anthropic.ThinkingConfigParamUnion, map[string]any) {
	switch effort {
	case "":
		return nil, nil
	case "none":
		disabled := anthropic.NewThinkingConfigDisabledParam()
		return &anthropic.ThinkingConfigParamUnion{OfDisabled: &disabled}, nil
	default:
		adaptive := anthropic.ThinkingConfigAdaptiveParam{}
		return &anthropic.ThinkingConfigParamUnion{OfAdaptive: &adaptive}, map[string]any{
			"output_config": map[string]any{"effort": effort},
		}
	}
}

// The OpenAI client accepts a base URL ending in /v1, while Anthropic's SDK
// appends v1/messages itself. Normalize the configured Messages base URL,
// with or without /v1, to the origin/path prefix the native client expects.
func claudeBaseURL(raw string) (string, error) {
	u, err := validatedModelBaseURL(raw)
	if err != nil {
		return "", err
	}
	path := strings.TrimRight(u.Path, "/")
	path = strings.TrimSuffix(path, "/v1")
	if path == "" {
		path = "/"
	} else {
		path += "/"
	}
	u.Path = path
	u.RawPath = ""
	return u.String(), nil
}
