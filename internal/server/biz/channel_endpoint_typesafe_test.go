package biz

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/internal/ent/channel"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/llm"
)

func TestChannelEndpoint_TypesafeStrictness(t *testing.T) {
	systemOneEndpoints := []objects.ChannelEndpoint{
		{APIFormat: llm.APIFormatTypeSafeSystemOne.String()},
	}

	// 1. Typesafe channel with typesafe/systemone endpoint should pass
	require.NoError(t, validateEndpointsForChannelType(channel.TypeTypesafe, systemOneEndpoints))

	// 2. Typesafe channel with openai/chat_completions endpoint should FAIL
	invalidEndpoints := []objects.ChannelEndpoint{
		{APIFormat: llm.APIFormatOpenAIChatCompletion.String()},
	}
	err := validateEndpointsForChannelType(channel.TypeTypesafe, invalidEndpoints)
	require.Error(t, err)
	require.Contains(t, err.Error(), "only supports api_format")

	// 3. Allowed non-typesafe channel types with a typesafe/systemone endpoint should pass
	for _, channelType := range []channel.Type{
		channel.TypeOpenai,
		channel.TypeCloudflare,
		channel.TypeOpenrouter,
		channel.TypeNanogpt,
		channel.TypeNanogptResponses,
	} {
		require.NoError(t, validateEndpointsForChannelType(channelType, systemOneEndpoints))
	}

	// 4. OpenAI and Cloudflare channels may mix chat completions with System One
	mixedEndpoints := []objects.ChannelEndpoint{
		{APIFormat: llm.APIFormatOpenAIChatCompletion.String()},
		{APIFormat: llm.APIFormatTypeSafeSystemOne.String()},
	}
	require.NoError(t, validateEndpointsForChannelType(channel.TypeOpenai, mixedEndpoints))
	require.NoError(t, validateEndpointsForChannelType(channel.TypeCloudflare, mixedEndpoints))

	// 5. Disallowed channel types keep rejecting System One, including mixed endpoints
	err = validateEndpointsForChannelType(channel.TypeAnthropic, systemOneEndpoints)
	require.Error(t, err)
	anthropicWithSystemOne := []objects.ChannelEndpoint{
		{APIFormat: llm.APIFormatAnthropicMessage.String()},
		{APIFormat: llm.APIFormatTypeSafeSystemOne.String()},
	}
	err = validateEndpointsForChannelType(channel.TypeAnthropic, anthropicWithSystemOne)
	require.Error(t, err)

	// 6. Default endpoints for TypeSafe channel must only be typesafe/systemone
	defaults := DefaultEndpointsForChannelType(channel.TypeTypesafe)
	require.Len(t, defaults, 1)
	require.Equal(t, llm.APIFormatTypeSafeSystemOne.String(), defaults[0].APIFormat)
}
