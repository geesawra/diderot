package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"slices"
	"sync"

	"github.com/anthropics/anthropic-sdk-go/v2"

	"github.com/bluesky-social/diderot/internal/config"
	"github.com/bluesky-social/diderot/internal/llmclient"
	"github.com/bluesky-social/diderot/internal/modelregistry"
	"github.com/bluesky-social/diderot/internal/sessioningest"
	"github.com/bluesky-social/diderot/internal/sessionsummary"
	"github.com/bluesky-social/diderot/internal/settings"
	"github.com/bluesky-social/diderot/internal/setup"
)

// modelRole is one enabled inference role as its model test calls it: key is
// its Models page key (empty for embeddings), name its title.
type modelRole struct {
	key      string
	name     string
	model    boundModel
	protocol modelregistry.Protocol
	// dimensions is what the embedding model's vectors must have.
	dimensions int
}

// enabledModelRoles lists every role models gives a model, and embeddings.
func enabledModelRoles(models *inferenceModels) []modelRole {
	var enabled []modelRole
	for _, role := range modelregistry.Roles {
		model := *models.role(role)
		if model.ID == "" {
			continue
		}
		protocol, _ := model.RoleProtocol(role)
		enabled = append(enabled, modelRole{key: string(role), name: roleInfo[role].title, model: model, protocol: protocol})
	}
	if models.embed.ID != "" {
		enabled = append(enabled, modelRole{
			name: "Embeddings", model: models.embed, protocol: modelregistry.ProtocolEmbeddings,
			dimensions: models.embedding.Dimensions,
		})
	}
	return enabled
}

// setupFacts is what the setup checklist shows from configuration.
func setupFacts(cfg *config.Config, catalog *modelCatalog, sa *serverAuth, summaries sessioningest.SummaryControl) setup.Facts {
	facts := setup.Facts{
		Access: setup.Access{
			AuthMode: cfg.AuthMode, BrowserSignIn: browserSignInMethods(cfg, sa),
			AdminGroups: cfg.AdminGroups, SelfServiceGroups: cfg.SelfServiceGroups,
			PolicyPath: cfg.AuthorizationPolicyPath,
		},
	}
	if cfg.SessionPush.Enabled {
		facts.SessionPush = &setup.SessionPush{NoPusher: noSessionPusher(cfg, sa), Summaries: summaries}
	}
	return facts
}

// noSessionPusher says why nobody can push agent sessions, or is empty when
// someone may: pushing takes a group in DIDEROT_SELF_SERVICE_GROUPS, which
// may deliberately name none, and in static mode only the token file grants
// groups.
func noSessionPusher(cfg *config.Config, sa *serverAuth) string {
	switch {
	case len(cfg.SelfServiceGroups) == 0:
		return "Nobody can push agent sessions: DIDEROT_SELF_SERVICE_GROUPS names no group."
	case sa == nil || sa.static == nil || sa.static.AnyInGroups(cfg.SelfServiceGroups):
		return ""
	}
	return "Nobody can push agent sessions: no token in the token file has a group from DIDEROT_SELF_SERVICE_GROUPS."
}

// browserSignInMethods names how a browser can sign in, for the Access page.
func browserSignInMethods(cfg *config.Config, sa *serverAuth) []string {
	if sa == nil {
		return nil
	}
	if sa.mode == config.AuthModeNone {
		return []string{"none: every browser is " + cfg.NoneEmail}
	}
	var methods []string
	switch {
	case sa.browserMode == config.BrowserAuthOIDC:
		methods = append(methods, "OIDC sign-in ("+cfg.OIDCIssuer+")")
	case len(cfg.TrustedProxyCIDRs) > 0:
		methods = append(methods, "trusted proxy header "+cfg.TrustedProxyEmailHeader)
	}
	if sa.localSignIn {
		methods = append(methods, "the bootstrap admin, while it is enabled")
		if cfg.AuthMode == config.AuthModeStatic {
			methods = append(methods, "a static API token")
		}
	}
	return methods
}

// newModelTester makes one tiny request to the model each enabled role uses
// under the saved choices, or only the named roles'. Each outcome is a short
// classification: no provider text or credential reaches the result.
func newModelTester(cfg *config.Config, catalog *modelCatalog) setup.ModelTester {
	return func(ctx context.Context, saved settings.Models, only ...string) []settings.RoleTest {
		roles := enabledModelRoles(catalog.resolve(saved))
		if len(only) > 0 {
			roles = slices.DeleteFunc(roles, func(role modelRole) bool { return !slices.Contains(only, role.key) })
		}
		results := make([]settings.RoleTest, len(roles))
		var wg sync.WaitGroup
		for i, role := range roles {
			wg.Add(1)
			go func() {
				defer wg.Done()
				err := probeModel(ctx, role)
				results[i] = settings.RoleTest{Role: role.name, Model: role.model.refOrEmpty(), Passed: err == nil}
				if err != nil {
					results[i].Error = modelTestError(ctx, err)
				}
			}()
		}
		wg.Wait()
		return results
	}
}

const modelTestPrompt = "Reply with OK."

// modelTestTokens is the smallest output limit every protocol accepts
// (Responses rejects fewer than 16).
const modelTestTokens = 16

// errEmbeddingDimensions is a model returning a vector the collection
// cannot hold.
var errEmbeddingDimensions = errors.New("embedding dimension mismatch")

func probeModel(ctx context.Context, role modelRole) error {
	endpoint := role.model.endpoint
	switch role.protocol {
	case modelregistry.ProtocolMessages:
		client, err := sessionsummary.NewGatewayClient(sessionsummary.GatewayClientConfig{
			BaseURL: endpoint.BaseURL, APIKey: endpoint.APIKey, Auth: endpoint.Auth, Headers: endpoint.Headers,
		})
		if err != nil {
			return err
		}
		_, err = client.CreateMessage(ctx, anthropic.MessageNewParams{
			Model: anthropic.Model(role.model.ID), MaxTokens: modelTestTokens,
			Messages: []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(modelTestPrompt))},
		})
		return err
	case modelregistry.ProtocolResponses:
		body := map[string]any{"model": role.model.ID, "input": modelTestPrompt, "max_output_tokens": modelTestTokens}
		if len(role.model.ReasoningEfforts) > 0 {
			body["reasoning"] = map[string]string{"effort": role.model.ReasoningEfforts[0]}
		}
		_, err := postModelTest(ctx, endpoint, "/responses", body)
		return err
	case modelregistry.ProtocolChat:
		body := struct {
			Model    string              `json:"model"`
			Messages []map[string]string `json:"messages"`
			llmclient.ChatTokenLimit
		}{
			Model:          role.model.ID,
			Messages:       []map[string]string{{"role": "user", "content": modelTestPrompt}},
			ChatTokenLimit: role.model.ChatTokenParam.Limit(modelTestTokens + role.model.ChatReasoningTokenAllowance),
		}
		payload, err := llmclient.WithExtraFields(body, role.model.ChatFields())
		if err != nil {
			return err
		}
		_, err = postModelTest(ctx, endpoint, "/chat/completions", payload)
		return err
	case modelregistry.ProtocolEmbeddings:
		body := map[string]any{"model": role.model.ID, "input": []string{modelTestPrompt}}
		raw, err := postModelTest(ctx, endpoint, "/embeddings", body)
		if err != nil {
			return err
		}
		var decoded struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		}
		if err := json.Unmarshal(raw, &decoded); err != nil || len(decoded.Data) != 1 {
			return errors.New("unexpected embeddings response")
		}
		if got := len(decoded.Data[0].Embedding); got != role.dimensions {
			return fmt.Errorf("%w: the model returned %d dimensions; its collection holds %d", errEmbeddingDimensions, got, role.dimensions)
		}
		return nil
	default:
		return fmt.Errorf("no test for protocol %q", role.protocol)
	}
}

func postModelTest(ctx context.Context, endpoint llmclient.Endpoint, path string, body any) ([]byte, error) {
	client, err := llmclient.New(llmclient.Config{BaseURL: endpoint.BaseURL, APIKey: endpoint.APIKey, Auth: endpoint.Auth, Headers: endpoint.Headers})
	if err != nil {
		return nil, err
	}
	return client.PostJSON(ctx, path, body, 1<<20)
}

// modelTestError classifies a failed probe without echoing the provider's
// response body, the endpoint URL, or a credential.
func modelTestError(ctx context.Context, err error) string {
	var status *llmclient.HTTPError
	var urlErr *url.Error
	var netErr net.Error
	switch {
	case errors.As(err, &status):
		return fmt.Sprintf("HTTP %d %s", status.StatusCode, http.StatusText(status.StatusCode))
	case errors.Is(err, errEmbeddingDimensions):
		return err.Error()
	case errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil:
		return fmt.Sprintf("no answer within %s", setup.ModelTestTimeout)
	case errors.As(err, &urlErr), errors.As(err, &netErr):
		return "could not reach the provider"
	default:
		return "unexpected response"
	}
}
