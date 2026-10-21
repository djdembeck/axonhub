package orchestrator

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/internal/authz"
	"github.com/looplj/axonhub/internal/contexts"
	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/ent/channel"
	"github.com/looplj/axonhub/internal/ent/enttest"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/pipeline/stream"
	"github.com/looplj/axonhub/llm/transformer/nanogpt"
	"github.com/looplj/axonhub/llm/transformer/openrouter"
	"github.com/looplj/axonhub/llm/transformer/typesafe"
)

// TestChatCompletionOrchestrator_Process_SystemOne_ProviderChannels verifies that
// provider channels (OpenRouter, NanoGPT) exposing a custom typesafe/systemone
// endpoint route the request through the typesafe outbound at the endpoint's own
// base_url + path, while the channel's default endpoints keep using the channel
// base_url.
func TestChatCompletionOrchestrator_Process_SystemOne_ProviderChannels(t *testing.T) {
	tests := []struct {
		name            string
		channelType     channel.Type
		channelBaseURL  string
		endpointBaseURL string
		endpointPath    string
		wantUpstreamURL string
		wantPrimaryType any
	}{
		{
			name:            "openrouter",
			channelType:     channel.TypeOpenrouter,
			channelBaseURL:  "https://openrouter.ai/api/v1",
			endpointBaseURL: "https://openrouter.ai",
			endpointPath:    "/api/alpha/decisions",
			wantUpstreamURL: "https://openrouter.ai/api/alpha/decisions",
			wantPrimaryType: &openrouter.OutboundTransformer{},
		},
		{
			name:            "nanogpt",
			channelType:     channel.TypeNanogpt,
			channelBaseURL:  "https://nano-gpt.com/api/v1",
			endpointBaseURL: "https://nano-gpt.com",
			endpointPath:    "/api/v1/decisions",
			wantUpstreamURL: "https://nano-gpt.com/api/v1/decisions",
			wantPrimaryType: &nanogpt.OutboundTransformer{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ctx = authz.WithTestBypass(ctx)

			client := enttest.NewEntClient(t, "sqlite3", "file:ent?mode=memory&_fk=0")
			defer client.Close()

			ctx = ent.NewContext(ctx, client)

			project := createTestProject(t, ctx, client)
			ctx = contexts.WithProjectID(ctx, project.ID)

			model := "typesafe/jev-1.13"
			ch, err := client.Channel.Create().
				SetType(tt.channelType).
				SetName("Test " + tt.name + " SystemOne Channel").
				SetBaseURL(tt.channelBaseURL).
				SetCredentials(objects.ChannelCredentials{APIKey: "sk-" + tt.name + "-test-key"}).
				SetSupportedModels([]string{model}).
				SetDefaultTestModel(model).
				SetEndpoints([]objects.ChannelEndpoint{{
					APIFormat: llm.APIFormatTypeSafeSystemOne.String(),
					BaseURL:   tt.endpointBaseURL,
					Path:      tt.endpointPath,
				}}).
				Save(ctx)
			require.NoError(t, err)

			channelService, requestService, systemService, usageLogService := setupTestServices(t, client)

			// Build the channel through the real biz service path so the
			// typesafe outbound is constructed by channel_llm.go's systemone case.
			bizChannel, err := channelService.GetChannel(ctx, ch.ID)
			require.NoError(t, err)

			// The biz-built channel carries a real HTTP client, which the
			// pipeline's CustomizeExecutor prefers over the factory executor.
			// Clear it so requests flow through the offline mock executor,
			// matching the hand-built channels used by the other systemone
			// orchestrator tests.
			bizChannel.HTTPClient = nil

			// (a) The systemone endpoint's outbound is the typesafe transformer,
			// and the channel's primary outbound stays the provider's transformer.
			systemOneOutbound, ok := bizChannel.Outbounds[llm.APIFormatTypeSafeSystemOne.String()]
			require.True(t, ok, "channel %s must expose a typesafe/systemone outbound", tt.name)
			require.IsType(t, &typesafe.OutboundTransformer{}, systemOneOutbound)
			require.IsType(t, tt.wantPrimaryType, bizChannel.Outbound)

			// (d) The channel's DEFAULT endpoint (openai/chat_completions) resolves
			// against the unchanged channel base_url, not the endpoint base_url.
			chatOutbound, err := biz.BuildOutboundByAPIFormat(bizChannel, llm.APIFormatOpenAIChatCompletion.String())
			require.NoError(t, err)
			chatReq, err := chatOutbound.TransformRequest(ctx, &llm.Request{
				Model:    model,
				Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("hello")}}},
			})
			require.NoError(t, err)
			require.Equal(t, tt.channelBaseURL+"/chat/completions", chatReq.URL)

			mockRespBody := []byte(`{
				"model": "` + model + `",
				"answers": {
					"urgent": {
						"type": "noul",
						"noul": 0.96
					}
				},
				"usage": {
					"input_tokens": 120,
					"output_tokens": 0
				}
			}`)

			executor := &mockExecutor{
				response: &httpclient.Response{
					StatusCode: 200,
					Body:       mockRespBody,
					Headers:    http.Header{"Content-Type": []string{"application/json"}},
				},
			}

			channelSelector := &staticChannelSelector{candidates: channelsToTestCandidates([]*biz.Channel{bizChannel}, model)}

			orch := &ChatCompletionOrchestrator{
				channelSelector:       channelSelector,
				Inbound:               typesafe.NewSystemOneInboundTransformer(),
				RequestService:        requestService,
				ChannelService:        channelService,
				PromptProvider:        &stubPromptProvider{},
				SystemService:         systemService,
				UsageLogService:       usageLogService,
				PipelineFactory:       pipeline.NewFactory(executor),
				ModelMapper:           NewModelMapper(),
				channelLimiterManager: NewChannelLimiterManager(),
				Middlewares: []pipeline.Middleware{
					stream.EnsureUsage(),
				},
			}

			reqBody := []byte(`{
				"model": "` + model + `",
				"state": "The server CPU usage is 98% and memory is saturated.",
				"questions": {
					"urgent": {
						"type": "noul",
						"instructions": "Is this situation critical and requires immediate pager duty alert?"
					}
				}
			}`)

			httpReq := &httpclient.Request{
				Method: http.MethodPost,
				Body:   reqBody,
				Headers: http.Header{
					"Content-Type": []string{"application/json"},
				},
			}

			result, err := orch.Process(ctx, httpReq)
			require.NoError(t, err)
			require.NotNil(t, result.ChatCompletion)
			require.Equal(t, http.StatusOK, result.ChatCompletion.StatusCode)

			// (a) The upstream HTTP request goes to exactly base_url + path.
			require.True(t, executor.requestCalled)
			require.Equal(t, tt.wantUpstreamURL, executor.lastRequest.URL)

			// (b) The upstream body is the verbatim System One body.
			var upstreamBody map[string]any
			err = json.Unmarshal(executor.lastRequest.Body, &upstreamBody)
			require.NoError(t, err)
			require.Equal(t, model, upstreamBody["model"])
			require.Equal(t, "The server CPU usage is 98% and memory is saturated.", upstreamBody["state"])
			questions, ok := upstreamBody["questions"].(map[string]any)
			require.True(t, ok)
			urgent, ok := questions["urgent"].(map[string]any)
			require.True(t, ok)
			require.Equal(t, "noul", urgent["type"])
			require.Equal(t, "Is this situation critical and requires immediate pager duty alert?", urgent["instructions"])

			// (c) The provider response parses; usage prompt/completion tokens are surfaced.
			var respBody map[string]any
			err = json.Unmarshal(result.ChatCompletion.Body, &respBody)
			require.NoError(t, err)
			require.Equal(t, model, respBody["model"])
			answers, ok := respBody["answers"].(map[string]any)
			require.True(t, ok)
			urgentAnswer, ok := answers["urgent"].(map[string]any)
			require.True(t, ok)
			require.Equal(t, "noul", urgentAnswer["type"])
			require.Equal(t, 0.96, urgentAnswer["noul"])
			usage, ok := respBody["usage"].(map[string]any)
			require.True(t, ok)
			require.Equal(t, float64(120), usage["input_tokens"])
			require.Equal(t, float64(0), usage["output_tokens"])

			execution, err := client.RequestExecution.Query().Only(ctx)
			require.NoError(t, err)
			require.Equal(t, model, execution.ModelID)
			require.Equal(t, model, execution.UpstreamModelID)
		})
	}
}
