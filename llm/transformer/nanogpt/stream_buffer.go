package nanogpt

import (
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/streams"
)

// maxBufferBytes limits how much content the buffer will accumulate before flushing
// as plain text. This prevents unbounded memory growth from false-positive
// couldStartXMLToolCall matches on normal text (e.g. "I'll <Read> the document").
const maxBufferBytes = 65536

// xmlToolCallBufferStream wraps a stream of llm.Response and buffers content
// that may contain XML tool calls from NanoGPT models. When complete XML tool calls
// are detected in the accumulated content, they are emitted as native tool_calls
// instead of plain text, preventing raw XML from leaking to the client.
//
// The stream operates in two modes:
//   - Normal mode: chunks are passed through unless they could start an XML tool call.
//   - Buffering mode: content is accumulated until either a complete tool call is
//     detected (emitted as native tool_calls), the content no longer looks like XML
//     (flushed as text), the buffer exceeds maxBufferBytes (flushed as text), or
//     the stream ends (truncated parse attempted, then flush).
//
// Note: drainBuffer may consume many upstream chunks before returning, as it
// accumulates content until a complete XML tool call arrives or buffering is aborted.
type xmlToolCallBufferStream struct {
	stream    streams.Stream[*llm.Response]
	buffer    strings.Builder
	buffering bool
	pending   []*llm.Response
	idx       int // -1 when no pending item is current
	current   *llm.Response
	err       error
	finished  bool

	// Metadata tracked during buffering so it isn't lost when
	// chunks are consumed by accumulateChunk.
	lastFinishReason *string
	lastUsage        *llm.Usage
	pendingToolCalls []llm.ToolCall
}

func newXMLToolCallBufferStream(stream streams.Stream[*llm.Response]) *xmlToolCallBufferStream {
	return &xmlToolCallBufferStream{
		stream: stream,
		idx:    -1,
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
		// Preserve any metadata from the chunk that triggered buffering
		s.captureMetadata(resp)
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

		// Safety valve: if the buffer exceeds the limit, flush as text.
		// This prevents unbounded memory growth from false-positive
		// couldStartXMLToolCall matches on normal text.
		if s.buffer.Len() > maxBufferBytes {
			s.flushBufferAsText()
			s.emitPendingMetadata()
			s.buffering = false
			if len(s.pending) > 0 {
				s.idx = 0
				s.current = s.pending[0]
				return true
			}
			// Continue to the next upstream chunk in normal mode
			break
		}

		// Attempt to parse complete XML tool calls.
		// Check for closing tags that indicate a complete tool call
		// for any of the formats ParseXMLToolCalls handles.
		if hasCompleteXMLTag(bufContent) {
			if MaybeHasXMLToolCalls(bufContent) {
				toolCalls, remaining, err := ParseXMLToolCalls(bufContent)
				if err == nil && len(toolCalls) > 0 {
					s.buffer.Reset()
					s.buffering = false
					s.emitToolCallResponse(toolCalls, remaining)
					s.emitPendingMetadata()
					s.idx = 0
					s.current = s.pending[0]
					return true
				}
			}
		}

		if !couldStartXMLToolCall(bufContent) {
			s.flushBufferAsText()
			s.emitPendingMetadata()
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
					s.emitPendingMetadata()
					s.idx = 0
					s.current = s.pending[0]
					return true
				}
			}

			s.flushBufferAsText()
			s.emitPendingMetadata()
			s.buffering = false
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

	// After flushing buffer or breaking out, continue in normal mode
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
		s.captureMetadata(resp)
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

// captureMetadata preserves non-content fields from a chunk that is about to be
// buffered. These fields (finish_reason, usage, existing tool_calls) would
// otherwise be lost since accumulateChunk only extracts text content.
func (s *xmlToolCallBufferStream) captureMetadata(resp *llm.Response) {
	if resp == nil || len(resp.Choices) == 0 {
		return
	}
	choice := resp.Choices[0]
	if choice.FinishReason != nil {
		s.lastFinishReason = choice.FinishReason
	}
	if choice.Delta != nil {
		if len(choice.Delta.ToolCalls) > 0 {
			s.pendingToolCalls = append(s.pendingToolCalls, choice.Delta.ToolCalls...)
		}
	}
	if resp.Usage != nil {
		s.lastUsage = resp.Usage
	}
}

// emitPendingMetadata creates response items for any metadata accumulated during
// buffering (usage stats, native tool_calls that arrived alongside content, etc.).
// finish_reason is attached to the last tool call response if appropriate.
func (s *xmlToolCallBufferStream) emitPendingMetadata() {
	// Emit any native tool_calls that arrived during buffering
	if len(s.pendingToolCalls) > 0 {
		tcResp := toolCallResponse(s.pendingToolCalls)
		// If we captured a finish_reason, attach it to the tool call response
		if s.lastFinishReason != nil {
			finishReason := *s.lastFinishReason
			tcResp.Choices[0].FinishReason = &finishReason
		}
		s.pending = append(s.pending, tcResp)
		s.pendingToolCalls = nil
		s.lastFinishReason = nil
	}

	// Emit usage information if captured
	if s.lastUsage != nil {
		usageResp := &llm.Response{
			Usage: s.lastUsage,
		}
		s.pending = append(s.pending, usageResp)
		s.lastUsage = nil
	}
}

// accumulateChunk appends content from a streaming chunk to the buffer.
// Non-content fields (role, finish_reason, usage, existing tool_calls) are
// tracked separately via captureMetadata to avoid data loss.
func (s *xmlToolCallBufferStream) accumulateChunk(resp *llm.Response) {
	s.captureMetadata(resp)
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
// If remaining content could itself start an XML tool call, it is re-buffered rather
// than emitted as plain text, preventing a second tool call from leaking as XML.
func (s *xmlToolCallBufferStream) emitToolCallResponse(toolCalls []llm.ToolCall, remaining string) {
	if remaining != "" {
		// Check if remaining content could start another XML tool call.
		// If so, re-buffer it instead of emitting as plain text.
		if couldStartXMLToolCall(remaining) {
			s.buffer.WriteString(remaining)
			s.buffering = true
		} else {
			s.pending = append(s.pending, textResponse(remaining))
		}
	}

	// Set finish_reason to "tool_calls" on the response carrying the tool calls,
	// matching the OpenAI streaming protocol convention.
	finishReason := "tool_calls"
	s.pending = append(s.pending, &llm.Response{
		Choices: []llm.Choice{{
			Delta: &llm.Message{
				ToolCalls: toolCalls,
			},
			FinishReason: &finishReason,
		}},
	})
}

// hasCompleteXMLTag checks whether the buffer content contains a closing tag
// for any of the XML formats that ParseXMLToolCalls handles. This is used to
// determine when to attempt parsing during streaming.
func hasCompleteXMLTag(content string) bool {
	// <function=NAME>...</function> format (case-insensitive check)
	lower := strings.ToLower(content)
	if strings.Contains(lower, "</function>") {
		return true
	}
	// <use_tool>...</use_tool> format
	if strings.Contains(lower, "</use_tool>") {
		return true
	}
	// <Write>...</Write>, <Read>...</Read> formats
	if strings.Contains(lower, "</write>") || strings.Contains(lower, "</read>") {
		return true
	}
	// <Bash>...</Bash> format
	if strings.Contains(lower, "</bash>") {
		return true
	}
	// Self-closing tags: />
	// Check for common tool tag self-closing patterns
	if strings.Contains(content, "/>") {
		return true
	}
	return false
}

// couldStartXMLToolCall checks whether content could be the start of an XML tool call.
// This is intentionally conservative — false positives just delay emission slightly
// (and are bounded by maxBufferBytes).
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
// Aligned with all XML formats handled by ParseXMLToolCalls:
//   - <function=...>, <Function=...> — function-param format
//   - <use_tool — use_tool format
//   - <Write, <Read — nested/attribute XML format
//   - <Bash — Bash command format
var xmlStartTags = []string{"<function", "<Function", "<use_tool", "<Write", "<Read", "<Bash"}

// xmlPartialStartTags lists shorter prefixes for detecting XML tool calls split
// across streaming chunks. These are substrings that could be the beginning of
// one of the xmlStartTags entries.
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
