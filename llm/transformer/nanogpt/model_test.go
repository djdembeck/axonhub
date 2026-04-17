package nanogpt

import (
	"strings"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"

	"github.com/looplj/axonhub/llm/transformer/openai"
)

func TestResponse_ToOpenAIResponse(t *testing.T) {
	tests := []struct {
		name     string
		response Response
		wantLen  int
		validate func(*testing.T, *openai.Response)
	}{
		{
			name: "empty choices",
			response: Response{
				Response: openai.Response{
					ID:      "test-1",
					Model:   "test-model",
					Choices: []openai.Choice{},
				},
				Choices: []Choice{},
			},
			wantLen: 0,
		},
		{
			name: "single choice with reasoning",
			response: Response{
				Response: openai.Response{
					ID:      "test-2",
					Model:   "test-model",
					Choices: []openai.Choice{},
				},
				Choices: []Choice{
					{
						Message: &Message{
							Reasoning: lo.ToPtr("thinking..."),
						},
					},
				},
			},
			wantLen: 1,
			validate: func(t *testing.T, resp *openai.Response) {
				assert.Equal(t, "thinking...", *resp.Choices[0].Message.ReasoningContent)
			},
		},
		{
			name: "multiple choices",
			response: Response{
				Response: openai.Response{
					ID:      "test-3",
					Model:   "test-model",
					Choices: []openai.Choice{},
				},
				Choices: []Choice{
					{Message: &Message{Reasoning: lo.ToPtr("reason1")}},
					{Message: &Message{Reasoning: lo.ToPtr("reason2")}},
					{Message: &Message{Reasoning: lo.ToPtr("reason3")}},
				},
			},
			wantLen: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := tt.response.ToOpenAIResponse()
			assert.Len(t, resp.Choices, tt.wantLen)
			if tt.validate != nil {
				tt.validate(t, resp)
			}
		})
	}
}

func TestChoice_ToOpenAIChoice(t *testing.T) {
	tests := []struct {
		name     string
		choice   Choice
		validate func(*testing.T, openai.Choice)
	}{
		{
			name: "choice with message containing reasoning",
			choice: Choice{
				Message: &Message{
					Reasoning: lo.ToPtr("reasoning content"),
				},
			},
			validate: func(t *testing.T, c openai.Choice) {
				assert.NotNil(t, c.Message)
				assert.Equal(t, "reasoning content", *c.Message.ReasoningContent)
			},
		},
		{
			name: "choice with delta containing reasoning",
			choice: Choice{
				Delta: &Message{
					Reasoning: lo.ToPtr("streaming reasoning"),
				},
			},
			validate: func(t *testing.T, c openai.Choice) {
				assert.NotNil(t, c.Delta)
				assert.Equal(t, "streaming reasoning", *c.Delta.ReasoningContent)
			},
		},
		{
			name: "choice with both message and delta",
			choice: Choice{
				Message: &Message{
					Reasoning: lo.ToPtr("final reasoning"),
				},
				Delta: &Message{
					Reasoning: lo.ToPtr("partial reasoning"),
				},
			},
			validate: func(t *testing.T, c openai.Choice) {
				assert.NotNil(t, c.Message)
				assert.NotNil(t, c.Delta)
				assert.Equal(t, "final reasoning", *c.Message.ReasoningContent)
				assert.Equal(t, "partial reasoning", *c.Delta.ReasoningContent)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			choice := tt.choice.ToOpenAIChoice()
			if tt.validate != nil {
				tt.validate(t, choice)
			}
		})
	}
}

func TestMessage_ToOpenAIMessage(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		validate func(*testing.T, openai.Message)
	}{
		{
			name: "message with reasoning maps to reasoning_content",
			message: Message{
				Reasoning: lo.ToPtr("thinking..."),
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.NotNil(t, msg.ReasoningContent)
				assert.Equal(t, "thinking...", *msg.ReasoningContent)
			},
		},
		{
			name:    "message without reasoning",
			message: Message{},
			validate: func(t *testing.T, msg openai.Message) {
				assert.Nil(t, msg.ReasoningContent)
			},
		},
		{
			name: "leaked thinking in content cleared when content equals reasoning",
			message: Message{
				Reasoning: lo.ToPtr("I am thinking about this problem"),
				Message: openai.Message{
					Content: openai.MessageContent{Content: lo.ToPtr("I am thinking about this problem")},
				},
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.NotNil(t, msg.ReasoningContent)
				assert.Equal(t, "I am thinking about this problem", *msg.ReasoningContent)
				assert.Nil(t, msg.Content.Content, "leaked thinking should be cleared from content")
			},
		},
		{
			name: "leaked thinking in content cleared when content is prefix of reasoning",
			message: Message{
				Reasoning: lo.ToPtr("I am thinking about this problem in great detail and here is more context about what I need to do next"),
				Message: openai.Message{
					Content: openai.MessageContent{Content: lo.ToPtr("I am thinking about this problem in great detail and here is more")},
				},
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.NotNil(t, msg.ReasoningContent)
				assert.Nil(t, msg.Content.Content, "leaked thinking prefix should be cleared from content")
			},
		},
		{
			name: "leaked thinking with native tool_calls preserved",
			message: Message{
				Reasoning: lo.ToPtr("I need to read that test to understand the failure scenario, let me search for it:\n<function=grep>\n<parameter=pattern>func TestFoo</parameter>\n</function>"),
				Message: openai.Message{
					Content: openai.MessageContent{Content: lo.ToPtr("I need to read that test to understand the failure scenario, let me search for it:")},
					ToolCalls: []openai.ToolCall{{
						ID:   "call_abc123",
						Type: "function",
						Function: openai.FunctionCall{
							Name:      "grep",
							Arguments: `{"pattern":"func TestFoo"}`,
						},
					}},
				},
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.Nil(t, msg.Content.Content, "leaked thinking should be cleared")
				assert.Len(t, msg.ToolCalls, 1, "native tool_calls should be preserved")
				assert.Equal(t, "grep", msg.ToolCalls[0].Function.Name)
				assert.NotContains(t, *msg.ReasoningContent, "<function=", "XML should be stripped from reasoning")
			},
		},
		{
			name: "XML in content without native tool_calls uses XML-parsed fallback",
			message: Message{
				Message: openai.Message{
					Content: openai.MessageContent{Content: lo.ToPtr("<function=grep>\n<parameter=pattern>func TestFoo</parameter>\n</function>")},
				},
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.Len(t, msg.ToolCalls, 1, "XML-parsed tool call should be used as fallback")
				assert.Equal(t, "grep", msg.ToolCalls[0].Function.Name)
			},
		},
		{
			name: "XML in content with native tool_calls does not duplicate",
			message: Message{
				Message: openai.Message{
					Content: openai.MessageContent{Content: lo.ToPtr("Result: <function=grep>\n<parameter=pattern>foo</parameter>\n</function>")},
					ToolCalls: []openai.ToolCall{{
						ID:   "call_native",
						Type: "function",
						Function: openai.FunctionCall{
							Name:      "bash",
							Arguments: `{"command":"ls"}`,
						},
					}},
				},
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.Len(t, msg.ToolCalls, 1, "should not duplicate tool_calls from XML")
				assert.Equal(t, "bash", msg.ToolCalls[0].Function.Name, "native tool_call should win")
			},
		},
		{
			name: "XML stripped from reasoning_content",
			message: Message{
				Reasoning: lo.ToPtr("Let me think...\n<function=grep>\n<parameter=pattern>foo</parameter>\n</function>"),
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.NotNil(t, msg.ReasoningContent)
				assert.NotContains(t, *msg.ReasoningContent, "<function=", "XML should be stripped from reasoning")
				assert.Contains(t, *msg.ReasoningContent, "Let me think...", "non-XML reasoning should be preserved")
			},
		},
		{
			name: "content not leaked when not matching reasoning",
			message: Message{
				Reasoning: lo.ToPtr("I am thinking about this"),
				Message: openai.Message{
					Content: openai.MessageContent{Content: lo.ToPtr("Here is the actual response")},
				},
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.NotNil(t, msg.Content.Content, "legitimate content should be preserved")
				assert.Equal(t, "Here is the actual response", *msg.Content.Content)
			},
		},
		{
			name: "short content prefix of reasoning not cleared (below minLeakedThinkingPrefixLen)",
			message: Message{
				Reasoning: lo.ToPtr("The answer is 42 and here is my detailed reasoning about why"),
				Message: openai.Message{
					Content: openai.MessageContent{Content: lo.ToPtr("The answer")},
				},
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.NotNil(t, msg.Content.Content, "short coincidental prefix should not be cleared")
				assert.Equal(t, "The answer", *msg.Content.Content)
			},
		},
		{
			name: "reasoning_content set to nil when XML consumes all reasoning",
			message: Message{
				Reasoning: lo.ToPtr("<function=grep>\n<parameter=pattern>foo</parameter>\n</function>"),
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.Nil(t, msg.ReasoningContent, "reasoning should be nil when XML consumes all content")
				assert.Len(t, msg.ToolCalls, 1, "XML tool call should be extracted as fallback")
				assert.Equal(t, "grep", msg.ToolCalls[0].Function.Name)
			},
		},
		{
			name: "XML in reasoning_content extracted as fallback when no native tool_calls",
			message: Message{
				Reasoning: lo.ToPtr("Let me search for that:\n<function=grep>\n<parameter=pattern>foo</parameter>\n</function>"),
				Message: openai.Message{
					Content: openai.MessageContent{Content: lo.ToPtr("Let me search for that:")},
				},
			},
			validate: func(t *testing.T, msg openai.Message) {
				assert.Len(t, msg.ToolCalls, 1, "XML tool call from reasoning should be extracted as fallback")
				assert.Equal(t, "grep", msg.ToolCalls[0].Function.Name)
				assert.NotContains(t, *msg.ReasoningContent, "<function=", "XML should be stripped from reasoning")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := tt.message.ToOpenAIMessage()
			if tt.validate != nil {
				tt.validate(t, msg)
			}
		})
	}
}

func TestIsLeakedThinkingContent(t *testing.T) {
	tests := []struct {
		name      string
		content   string
		reasoning string
		want      bool
	}{
		{name: "exact match", content: "abc", reasoning: "abc", want: true},
		{name: "content is long prefix of reasoning", content: strings.Repeat("x", 50), reasoning: strings.Repeat("x", 50) + " and more", want: true},
		{name: "content is short prefix of reasoning", content: "The answer", reasoning: "The answer is 42", want: false},
		{name: "content is not a prefix", content: "Different response entirely", reasoning: "I am thinking about something else", want: false},
		{name: "empty content", content: "", reasoning: "some reasoning", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, isLeakedThinkingContent(tt.content, tt.reasoning))
		})
	}
}
