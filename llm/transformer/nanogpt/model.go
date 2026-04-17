package nanogpt

import (
	"strings"

	"github.com/looplj/axonhub/llm/transformer/openai"
)

// Response represents a NanoGPT chat completion response.
// It extends the OpenAI response format to handle NanoGPT-specific fields like reasoning.
type Response struct {
	openai.Response

	Choices []Choice `json:"choices"`
}

// ToOpenAIResponse converts the NanoGPT Response to an OpenAI Response.
func (r *Response) ToOpenAIResponse() *openai.Response {
	r.Response.Choices = make([]openai.Choice, 0, len(r.Choices))
	for _, choice := range r.Choices {
		r.Response.Choices = append(r.Response.Choices, choice.ToOpenAIChoice())
	}

	return &r.Response
}

// Choice represents a choice in the response.
// It extends the OpenAI Choice to handle NanoGPT-specific message fields.
type Choice struct {
	openai.Choice

	Message *Message `json:"message,omitempty"`
	Delta   *Message `json:"delta,omitempty"`
}

// ToOpenAIChoice converts the NanoGPT Choice to an OpenAI Choice.
func (c *Choice) ToOpenAIChoice() openai.Choice {
	if c.Message != nil {
		msg := c.Message.ToOpenAIMessage()
		c.Choice.Message = &msg
	}

	if c.Delta != nil {
		delta := c.Delta.ToOpenAIMessage()
		c.Choice.Delta = &delta
	}

	return c.Choice
}

// Message represents a message in the response.
// It extends the OpenAI Message to handle NanoGPT-specific fields like reasoning.
type Message struct {
	openai.Message

	// Reasoning is the reasoning content from NanoGPT models (e.g., zai-org/glm-4.7:thinking).
	Reasoning *string `json:"reasoning,omitempty"`
}

const minLeakedThinkingPrefixLen = 50

// ToOpenAIMessage converts the NanoGPT Message to an OpenAI Message.
func (m *Message) ToOpenAIMessage() openai.Message {
	m.mapReasoningToReasoningContent()
	m.clearLeakedThinkingContent()
	m.parseXMLToolCallsFromContent()
	m.extractAndStripXMLFromReasoningContent()

	return m.Message
}

func (m *Message) mapReasoningToReasoningContent() {
	if m.Reasoning != nil {
		m.ReasoningContent = m.Reasoning
	}
}

func isLeakedThinkingContent(content, reasoning string) bool {
	if content == reasoning {
		return true
	}
	if len(content) < minLeakedThinkingPrefixLen {
		return false
	}
	return strings.HasPrefix(reasoning, content)
}

func (m *Message) clearLeakedThinkingContent() {
	if m.Content.Content == nil || *m.Content.Content == "" || m.ReasoningContent == nil {
		return
	}

	content := *m.Content.Content
	reasoning := *m.ReasoningContent

	if isLeakedThinkingContent(content, reasoning) {
		m.Content = openai.MessageContent{Content: nil}
	}
}

func (m *Message) parseXMLToolCallsFromContent() {
	if m.Content.Content == nil || *m.Content.Content == "" {
		return
	}

	content := *m.Content.Content
	if !MaybeHasXMLToolCalls(content) {
		return
	}

	tools, remaining, err := ParseXMLToolCalls(content)
	if err != nil || len(tools) == 0 {
		return
	}

	if len(m.ToolCalls) > 0 {
		m.setContentOrClear(remaining)
	} else {
		m.ToolCalls = ToOpenAIToolCalls(tools)
		m.setContentOrClear(remaining)
	}
}

func (m *Message) extractAndStripXMLFromReasoningContent() {
	if m.ReasoningContent == nil || *m.ReasoningContent == "" {
		return
	}

	reasoning := *m.ReasoningContent
	if !MaybeHasXMLToolCalls(reasoning) {
		return
	}

	tools, remaining, err := ParseXMLToolCalls(reasoning)
	if err != nil {
		return
	}

	cleaned := strings.TrimSpace(remaining)
	m.ReasoningContent = stringPtrOrNil(cleaned)

	if len(tools) > 0 && len(m.ToolCalls) == 0 {
		m.ToolCalls = ToOpenAIToolCalls(tools)
	}
}

func (m *Message) setContentOrClear(remaining string) {
	if remaining != "" {
		m.Content = ToOpenAIMessageContent(remaining)
	} else {
		m.Content = openai.MessageContent{Content: nil}
	}
}

func stringPtrOrNil(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
