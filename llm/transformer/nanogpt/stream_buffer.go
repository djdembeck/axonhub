package nanogpt

import (
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/streams"
)

// xmlToolCallBufferStream wraps a stream of llm.Response and buffers content
// that may contain XML tool calls from NanoGPT models. When complete XML tool calls
// are detected in the accumulated content, they are emitted as native tool_calls
// instead of plain text, preventing raw XML from leaking to the client.
type xmlToolCallBufferStream struct {
	stream    streams.Stream[*llm.Response]
	buffer    strings.Builder
	buffering bool
	pending   []*llm.Response
	idx       int
	current   *llm.Response
	err       error
	finished  bool
}

func newXMLToolCallBufferStream(stream streams.Stream[*llm.Response]) *xmlToolCallBufferStream {
	return &xmlToolCallBufferStream{
		stream: stream,
	}
}

func (s *xmlToolCallBufferStream) Next() bool {
	// Advance past current pending item
	s.idx++

	// Drain pending responses
	if s.idx < len(s.pending) {
		s.current = s.pending[s.idx]
		return true
	}
	s.pending = nil
	s.idx = -1

	if s.finished || s.err != nil {
		return false
	}

	// If actively buffering, continue consuming upstream chunks
	if s.buffering {
		return s.drainBuffer()
	}

	// Normal (non-buffering) mode
	if !s.stream.Next() {
		s.err = s.stream.Err()
		s.finished = true
		return false
	}

	resp := s.stream.Current()
	content := extractContentFromResponse(resp)
	if content == "" {
		s.current = resp
		return true
	}

	if couldStartXMLToolCall(content) {
		before, after := splitAtXMLStart(content)
		if before != "" {
			s.pending = append(s.pending, textResponse(before))
		}
		if after != "" {
			s.buffer.WriteString(after)
			s.buffering = true
		}
		if len(s.pending) > 0 {
			s.idx = 0
			s.current = s.pending[0]
			return true
		}
		return s.drainBuffer()
	}

	s.current = resp
	return true
}

// drainBuffer consumes upstream chunks while buffering XML content.
// Returns true when a complete response (text flush or tool call) is ready to emit.
func (s *xmlToolCallBufferStream) drainBuffer() bool {
	for {
		bufContent := s.buffer.String()

		// Only attempt to parse COMPLETE XML tool calls (with closing </function>)
		// while the stream is still active. This prevents emitting a truncated
		// tool call before all parameters have arrived across streaming chunks.
		if strings.Contains(bufContent, "</function>") || strings.Contains(bufContent, "</Function>") {
			if MaybeHasXMLToolCalls(bufContent) {
				toolCalls, remaining, err := ParseXMLToolCalls(bufContent)
				if err == nil && len(toolCalls) > 0 {
					s.buffer.Reset()
					s.buffering = false
					s.emitToolCallResponse(toolCalls, remaining)
					s.idx = 0
					s.current = s.pending[0]
					return true
				}
			}
		}

		if !couldStartXMLToolCall(bufContent) {
			s.flushBufferAsText()
			s.buffering = false
			if len(s.pending) > 0 {
				s.idx = 0
				s.current = s.pending[0]
				return true
			}
			break
		}

		if !s.stream.Next() {
			s.err = s.stream.Err()
			s.finished = true

			// Stream ended — try parsing as truncated tool call before flushing as text
			if MaybeHasXMLToolCalls(bufContent) {
				toolCalls, remaining, err := ParseXMLToolCalls(bufContent)
				if err == nil && len(toolCalls) > 0 {
					s.buffer.Reset()
					s.buffering = false
					s.emitToolCallResponse(toolCalls, remaining)
					s.idx = 0
					s.current = s.pending[0]
					return true
				}
			}

			s.flushBufferAsText()
			if len(s.pending) > 0 {
				s.idx = 0
				s.current = s.pending[0]
				return true
			}
			return false
		}

		resp := s.stream.Current()
		s.accumulateChunk(resp)
	}

	if !s.stream.Next() {
		s.err = s.stream.Err()
		s.finished = true
		return false
	}

	resp := s.stream.Current()
	content := extractContentFromResponse(resp)
	if content == "" {
		s.current = resp
		return true
	}

	if couldStartXMLToolCall(content) {
		before, after := splitAtXMLStart(content)
		if before != "" {
			s.pending = append(s.pending, textResponse(before))
		}
		if after != "" {
			s.buffer.WriteString(after)
			s.buffering = true
		}
		if len(s.pending) > 0 {
			s.idx = 0
			s.current = s.pending[0]
			return true
		}
		return s.drainBuffer()
	}

	s.current = resp
	return true
}

func (s *xmlToolCallBufferStream) Current() *llm.Response {
	return s.current
}

func (s *xmlToolCallBufferStream) Err() error {
	if s.err != nil {
		return s.err
	}
	return s.stream.Err()
}

func (s *xmlToolCallBufferStream) Close() error {
	return s.stream.Close()
}

// accumulateChunk appends content from a streaming chunk to the buffer.
// Non-content fields (role, finish_reason, etc.) are tracked separately.
func (s *xmlToolCallBufferStream) accumulateChunk(resp *llm.Response) {
	content := extractContentFromResponse(resp)
	if content != "" {
		s.buffer.WriteString(content)
	}
}

// flushBufferAsText emits the accumulated buffer content as a text response.
func (s *xmlToolCallBufferStream) flushBufferAsText() {
	if s.buffer.Len() == 0 {
		return
	}
	content := s.buffer.String()
	s.buffer.Reset()
	s.pending = append(s.pending, textResponse(content))
}

// emitToolCallResponse creates a response with tool calls and optional remaining text.
func (s *xmlToolCallBufferStream) emitToolCallResponse(toolCalls []llm.ToolCall, remaining string) {
	if remaining != "" {
		s.pending = append(s.pending, textResponse(remaining))
	}
	s.pending = append(s.pending, toolCallResponse(toolCalls))
}

// couldStartXMLToolCall checks whether content could be the start of an XML tool call.
// This is intentionally conservative — false positives just delay emission slightly.
func couldStartXMLToolCall(content string) bool {
	for _, tag := range xmlStartTags {
		if strings.Contains(content, tag) {
			return true
		}
	}
	for _, tag := range xmlPartialStartTags {
		if strings.Contains(content, tag) {
			return true
		}
	}
	return false
}

// xmlStartTags lists the tags that indicate the start of an XML tool call.
var xmlStartTags = []string{"<function", "<Function", "<use_tool", "<Write", "<Read", "<Bash"}

// xmlPartialStartTags lists shorter prefixes for detecting XML tool calls split across streaming chunks.
var xmlPartialStartTags = []string{"<func", "<Func", "<use_", "<Writ", "<Read", "<Bash"}

// splitAtXMLStart splits content at the first XML tool call start tag.
// Returns (textBefore, xmlPart). If no XML start is found, returns (content, "").
func splitAtXMLStart(content string) (string, string) {
	firstIdx := len(content)
	allTags := append(append([]string{}, xmlStartTags...), xmlPartialStartTags...)
	for _, tag := range allTags {
		if idx := strings.Index(content, tag); idx != -1 && idx < firstIdx {
			firstIdx = idx
		}
	}
	if firstIdx == len(content) {
		return content, ""
	}
	return content[:firstIdx], content[firstIdx:]
}

// extractContentFromResponse extracts the text content from an llm.Response delta.
func extractContentFromResponse(resp *llm.Response) string {
	if resp == nil || len(resp.Choices) == 0 {
		return ""
	}
	delta := resp.Choices[0].Delta
	if delta == nil {
		return ""
	}
	if delta.Content.Content != nil {
		return *delta.Content.Content
	}
	return ""
}

// textResponse creates a streaming delta response with the given text content.
func textResponse(content string) *llm.Response {
	return &llm.Response{
		Choices: []llm.Choice{{
			Delta: &llm.Message{
				Content: llm.MessageContent{Content: &content},
			},
		}},
	}
}

// toolCallResponse creates a streaming delta response with tool calls.
func toolCallResponse(toolCalls []llm.ToolCall) *llm.Response {
	return &llm.Response{
		Choices: []llm.Choice{{
			Delta: &llm.Message{
				ToolCalls: toolCalls,
			},
		}},
	}
}
