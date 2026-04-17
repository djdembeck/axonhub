package nanogpt

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer/openai"
)

// Maximum content length to prevent ReDoS attacks
const maxXMLParseLength = 100000 // 100KB

// toolCallPattern matches XML-like tool calls with content: <Tag>content</Tag>
// Uses [^<] to match content safely without ReDoS backtracking
// Allows optional whitespace after tag name for formats like <Write_File>{...}</Write_File>
var toolCallPattern = regexp.MustCompile(`<([a-zA-Z_][a-zA-Z0-9_-]*)[\s]*([^>]*)>([^<]*)</([a-zA-Z_][a-zA-Z0-9_-]*)>`)

// selfClosingPattern matches self-closing XML tags: <Tag attr="val" />
// Allows optional space between tag name and attributes
var selfClosingPattern = regexp.MustCompile(`<([a-zA-Z_][a-zA-Z0-9_-]*)[\s]*([^>]*)/>`)

// attrPattern matches attributes like name="value" or name='value'
// Handles both single and double quotes, allows empty values
var attrPattern = regexp.MustCompile(`([a-zA-Z_][a-zA-Z0-9_-]*)[\s]*=[\s]*["']([^"']*)["']`)

// normalizeTagPattern matches tags without space before />
var normalizeTagPattern = regexp.MustCompile(`([^\s])/>`)
// nestedXMLPattern matches nested XML like <Write><file_path>X</file_path><content>Y</content></Write>
var nestedXMLPattern = regexp.MustCompile(`<(Write|Read)[^>]*>\s*<file_path>([^<]*)</file_path>\s*<content>([\s\S]*?)</content>\s*</(Write|Read)>`)
// mismatchTagPattern matches <Write>content</use_tool> type patterns
// Uses [^<] to match content safely without ReDoS backtracking
var mismatchTagPattern = regexp.MustCompile(`<(Write|Read|Write_FILE|Write_file|Read_FILE|Read_file)([^>]*)>([^<]*)</use_tool>`)

// unclosedPattern matches unclosed opening tags like <Write attr="..."\n</use_tool>
var unclosedPattern = regexp.MustCompile(`<(Write|Read|Write_FILE|Write_file|Read_FILE|Read_file)([^>]*)\n([\s\S]*?)</use_tool>`)

// functionParamPattern matches the <function=NAME>...</function> format
// used by some NanoGPT models that output tool calls as XML instead of native JSON.
// Allows hyphens and dots in function names (e.g. edit-file, search.web).
// Uses (?s).*? instead of [^<] because parameter values may contain < characters (e.g. HTML, code).
// Go's RE2 engine guarantees bounded execution, so lazy quantifiers don't cause ReDoS.
var functionParamPattern = regexp.MustCompile(`(?si)<function=([a-zA-Z_][a-zA-Z0-9_.-]*)>(.*?)</function>`)

// functionParamOpenPattern matches an opening <function=NAME> without requiring the closing tag.
// Used to detect truncated/incomplete function blocks during streaming.
var functionParamOpenPattern = regexp.MustCompile(`(?si)<function=([a-zA-Z_][a-zA-Z0-9_.-]*)>`)

// Note: parameter extraction is handled by the scanner-based parseFunctionParams
// rather than a regex, to correctly handle values containing < characters.
// MaybeHasXMLToolCalls is a fast pre-check to determine if content likely contains XML tool calls.
func MaybeHasXMLToolCalls(content string) bool {
	if len(content) > maxXMLParseLength {
		content = truncateToValidUTF8(content, maxXMLParseLength)
	}

	// Check for common tool-related patterns
	return strings.Contains(content, "<") && strings.Contains(content, ">") &&
		(toolCallPattern.MatchString(content) ||
			selfClosingPattern.MatchString(content) ||
			containsFunctionTag(content) ||
			strings.Contains(content, "use_tool") ||
			strings.Contains(content, "Write") ||
			strings.Contains(content, "Bash") ||
			strings.Contains(content, "Read"))
}

// ParseXMLToolCalls extracts tool calls from XML content using regex-based parsing.
// Handles various XML formats including:
// - <function=NAME><parameter=KEY>VALUE</parameter>...</function>
// - <use_tool name="X"><arg>value</arg></use_tool>
// - <Write file_path="X" content="Y"/>
// - <Write file_path="X">content</Write>
// - <Write> {"file_path": "X", "content": "Y"}</use_tool>
// Returns the parsed tool calls, any remaining content after tool calls, and any error encountered.
func ParseXMLToolCalls(content string) ([]llm.ToolCall, string, error) {
	// Fast check - if no XML tool tags, return as-is
	if !MaybeHasXMLToolCalls(content) {
		return nil, content, nil
	}

	// Enforce length limit during parsing in addition to the pre-check.
	// Truncate at a valid UTF-8 boundary to avoid splitting mid-character.
	if len(content) > maxXMLParseLength {
		content = truncateToValidUTF8(content, maxXMLParseLength)
	}

	// Normalize common malformed variations
	content = normalizeXML(content)

	var toolCalls []llm.ToolCall
	var remainingContent strings.Builder
	lastEnd := 0

	// Find all matches from both patterns
	type matchInfo struct {
		start        int
		end          int
		tagName      string
		attrs        string
		innerContent string
		closingTag   string // for validation
	}

	var matches []matchInfo

	// Handle <function=NAME>/<parameter=KEY>VALUE</parameter>...</function> format
	// Processed first as it's the most specific and self-contained pattern.
	for _, m := range functionParamPattern.FindAllStringSubmatchIndex(content, -1) {
		if len(m) >= 6 {
			funcName := strings.ToLower(content[m[2]:m[3]])
			paramsBody := content[m[4]:m[5]]

			args := parseFunctionParams(paramsBody)

			argsJSON, err := json.Marshal(args)
			if err != nil {
				argsJSON = []byte("{}")
			}

			id := generateToolCallID(funcName, string(argsJSON))

			toolCalls = append(toolCalls, llm.ToolCall{
				Index: len(toolCalls),
				ID:    id,
				Type:  "function",
				Function: llm.FunctionCall{
					Name:      funcName,
					Arguments: string(argsJSON),
				},
			})

			if lastEnd < m[0] {
				remainingContent.WriteString(content[lastEnd:m[0]])
			}
			lastEnd = m[1]
		}
	}

	// Handle truncated <function=NAME>... blocks without closing </function>.
	// During streaming the closing tag may not have arrived yet.
	// Only process if we didn't already match a complete function block at this position.
	// Check m[0] (start position) against lastEnd to avoid overlapping with
	// complete blocks already consumed above.
	for _, m := range functionParamOpenPattern.FindAllStringSubmatchIndex(content, -1) {
		if len(m) >= 4 && m[0] >= lastEnd {
			funcName := strings.ToLower(content[m[2]:m[3]])

			// Extract everything after the opening tag up to end of content
			paramsBody := content[m[1]:]

			args := parseFunctionParams(paramsBody)
			if len(args) == 0 {
				continue
			}

			argsJSON, err := json.Marshal(args)
			if err != nil {
				argsJSON = []byte("{}")
			}

			id := generateToolCallID(funcName, string(argsJSON))

			toolCalls = append(toolCalls, llm.ToolCall{
				Index: len(toolCalls),
				ID:    id,
				Type:  "function",
				Function: llm.FunctionCall{
					Name:      funcName,
					Arguments: string(argsJSON),
				},
			})

			if lastEnd < m[0] {
				remainingContent.WriteString(content[lastEnd:m[0]])
			}
			lastEnd = len(content)
		}
	}

	// Handle nested XML format: <Write><file_path>X</file_path><content>Y</content></Write>
	// This must be processed before other patterns to avoid matching inner elements
	for _, m := range nestedXMLPattern.FindAllStringSubmatchIndex(content, -1) {
		if len(m) >= 10 {
			// m[2]:m[3] = opening tag (Write/Read)
			// m[4]:m[5] = file_path value
			// m[6]:m[7] = content value
			// m[8]:m[9] = closing tag
			tagName := content[m[2]:m[3]]
			filePath := content[m[4]:m[5]]
			innerContent := content[m[6]:m[7]]
			closingTag := content[m[8]:m[9]]

			// Only process if opening and closing tags match
			if strings.EqualFold(tagName, closingTag) {
				// Create synthetic attributes - escape quotes to prevent XML injection
				filePathEscaped := strings.ReplaceAll(filePath, `"`, `\"`)
				innerContentEscaped := strings.ReplaceAll(innerContent, `"`, `\"`)
				attrs := fmt.Sprintf(`file_path="%s" content="%s"`, filePathEscaped, innerContentEscaped)
				matches = append(matches, matchInfo{
					start:        m[0],
					end:          m[1],
					tagName:      tagName,
					attrs:        attrs,
					innerContent: "", // Content is already in attrs
					closingTag:   closingTag,
				})
			}
		}
	}
	// Find opening/closing tag patterns
	for _, m := range toolCallPattern.FindAllStringSubmatchIndex(content, -1) {
		if len(m) >= 10 {
			openingTag := content[m[2]:m[3]]
			closingTag := content[m[8]:m[9]]
			// Only accept if opening and closing tags match
			if strings.EqualFold(openingTag, closingTag) {
				matches = append(matches, matchInfo{
					start:        m[0],
					end:          m[1],
					tagName:      openingTag,
					attrs:        content[m[4]:m[5]],
					innerContent: content[m[6]:m[7]],
					closingTag:   closingTag,
				})
			}
		}
	}

	// Find self-closing patterns
	for _, m := range selfClosingPattern.FindAllStringSubmatchIndex(content, -1) {
		if len(m) >= 6 {
			matches = append(matches, matchInfo{
				start:   m[0],
				end:     m[1],
				tagName: content[m[2]:m[3]],
				attrs:   content[m[4]:m[5]],
			})
		}
	}

	// Sort matches by start position using efficient O(n log n) sort
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].start < matches[j].start
	})

	for _, match := range matches {
		// Skip matches fully consumed by functionParam processing
		if match.end <= lastEnd {
			continue
		}
		// Partially overlapping match: adjust start to avoid re-consuming content
		if match.start < lastEnd {
			match.start = lastEnd
		}
		// Skip if not a valid tool tag
		toolName := extractToolName(match.tagName, match.attrs)
		if toolName == "" {
			// Not a recognized tool pattern, keep in remaining content
			if lastEnd < match.start {
				remainingContent.WriteString(content[lastEnd:match.start])
			}
			lastEnd = match.end
			continue
		}

		// Extract tool arguments
		args := extractToolArguments(toolName, match.attrs, match.innerContent)

		// Generate deterministic ID
		id := generateToolCallID(toolName, args)

		toolCalls = append(toolCalls, llm.ToolCall{
			Index: len(toolCalls),
			ID:    id,
			Type:  "function",
			Function: llm.FunctionCall{
				Name:      toolName,
				Arguments: args,
			},
		})

		// Add any text before this tool call to remaining content
		if lastEnd < match.start {
			remainingContent.WriteString(content[lastEnd:match.start])
		}
		lastEnd = match.end
	}

	// Add any remaining content after the last tool call
	if lastEnd < len(content) {
		remainingContent.WriteString(content[lastEnd:])
	}

	if len(toolCalls) == 0 {
		return nil, content, nil
	}

	remaining := strings.TrimSpace(remainingContent.String())
	return toolCalls, remaining, nil
}

// normalizeXML fixes common XML malformations from NanoGPT
func normalizeXML(content string) string {
	// Fix unclosed opening tags: <Write attr="..."\ncontent</use_tool> -> <Write attr="...">\ncontent</use_tool>
	// NanoGPT sometimes omits the closing > on the opening tag
	content = unclosedPattern.ReplaceAllStringFunc(content, func(match string) string {
		parts := unclosedPattern.FindStringSubmatch(match)
		if len(parts) >= 4 {
			return "<" + parts[1] + parts[2] + ">\n" + parts[3] + "</use_tool>"
		}
		return match
	})
	// Fix mismatched closing tags - handle variations like </use_tool>, </use_use>, etc.
	content = strings.ReplaceAll(content, "</use_use>", "</use_tool>")
	content = strings.ReplaceAll(content, "</Write_file>", "</Write>")
	content = strings.ReplaceAll(content, "</Write_FILE>", "</Write>")
	content = strings.ReplaceAll(content, "</Read_file>", "</Read>")
	content = strings.ReplaceAll(content, "</Read_FILE>", "</Read>")

	// Fix weird patterns like <Write>content</use_tool> -> <Write>content</Write>
	// Use ReplaceAllFunc to preserve the opening tag name
	content = mismatchTagPattern.ReplaceAllStringFunc(content, func(match string) string {
		// Extract parts from the match
		parts := mismatchTagPattern.FindStringSubmatch(match)
		if len(parts) >= 4 {
			tagName := parts[1]
			attrs := parts[2]
			innerContent := parts[3]
			return "<" + tagName + attrs + ">" + innerContent + "</" + tagName + ">"
		}
		return match
	})

	// Normalize self-closing tags without space before />
	content = normalizeTagPattern.ReplaceAllString(content, "$1 />")

	return content
}

// extractToolName determines the tool name from tag name and attributes
func extractToolName(tagName, attrs string) string {
	tagName = strings.TrimSpace(strings.ToLower(tagName))

	// Direct tool name tags (handle variations like Write_FILE, Write_file, etc.)
	switch {
	case strings.HasPrefix(tagName, "write"):
		return "write"
	case strings.HasPrefix(tagName, "read"):
		return "read"
	case tagName == "bash", tagName == "python", tagName == "search", tagName == "glob":
		return tagName
	case tagName == "use_tool":
		// Extract from name attribute
		if matches := attrPattern.FindAllStringSubmatch(attrs, -1); matches != nil {
			for _, match := range matches {
				if len(match) >= 3 && strings.ToLower(match[1]) == "name" {
					return strings.ToLower(match[2])
				}
			}
		}
	}

	return ""
}

// extractToolArguments extracts arguments from attributes and/or inner content
func extractToolArguments(toolName, attrs, innerContent string) string {
	args := make(map[string]interface{})

	// Extract attributes
	attrMatches := attrPattern.FindAllStringSubmatch(attrs, -1)
	for _, match := range attrMatches {
		if len(match) >= 3 {
			key := match[1]
			value := match[2]
			// Skip the "name" attribute for use_tool tags
			if strings.ToLower(key) == "name" && toolName != "" {
				continue
			}
			args[key] = value
		}
	}

	// Handle inner content
	innerContent = strings.TrimSpace(innerContent)
	if innerContent != "" {
		// Try to parse as JSON first
		var jsonContent interface{}
		if err := json.Unmarshal([]byte(innerContent), &jsonContent); err == nil {
			// If valid JSON, merge with args
			if jsonMap, ok := jsonContent.(map[string]interface{}); ok {
				for k, v := range jsonMap {
					if _, exists := args[k]; !exists {
						args[k] = v
					}
				}
			} else {
				args["content"] = jsonContent
			}
		} else {
			// Not valid JSON, add as content or arg
			if _, hasContent := args["content"]; !hasContent {
				args["content"] = innerContent
			} else {
				args["arg"] = innerContent
			}
		}
	}

	// Serialize to JSON
	result, err := json.Marshal(args)
	if err != nil || len(args) == 0 {
		return "{}"
	}

	return string(result)
}

// generateToolCallID generates a deterministic ID from the tool name and arguments.
func generateToolCallID(name, args string) string {
	hasher := sha256.New()
	hasher.Write([]byte(name))
	hasher.Write([]byte(args))
	hash := hasher.Sum(nil)
	return "nanogpt_" + hex.EncodeToString(hash)[:16]
}

// ToOpenAIToolCalls converts llm.ToolCall slice to openai.ToolCall slice.
func ToOpenAIToolCalls(toolCalls []llm.ToolCall) []openai.ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}

	result := make([]openai.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		result[i] = openai.ToolCall{
			ID:    tc.ID,
			Type:  tc.Type,
			Index: tc.Index,
			Function: openai.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return result
}

// ToOpenAIMessageContent converts a string to openai.MessageContent.
func ToOpenAIMessageContent(content string) openai.MessageContent {
	return openai.MessageContent{
		Content: &content,
	}
}

// parseFunctionParams extracts parameter key-value pairs from the body of a <function=...> element.
// Uses a scanner instead of regex to correctly handle values containing < characters.
// Each <parameter=KEY>VALUE</parameter> child is parsed. Values are attempted as JSON first;
// if that fails they are kept as plain strings. Empty values become empty strings.
func parseFunctionParams(paramsBody string) map[string]interface{} {
	args := make(map[string]interface{})

	body := paramsBody
	for {
		openIdx := findParameterOpenTag(body)
		if openIdx == -1 {
			break
		}

		gtIdx := strings.Index(body[openIdx:], ">")
		if gtIdx == -1 {
			break
		}
		key := strings.TrimSpace(body[openIdx+len("<parameter=") : openIdx+gtIdx])

		valueStart := openIdx + gtIdx + 1
		valueEnd := findParameterCloseTag(body, valueStart)

		if valueEnd == -1 {
			value := strings.TrimSpace(body[valueStart:])
			if key != "" {
				args[key] = coerceParamValue(value)
			}
			break
		}

		value := strings.TrimSpace(body[valueStart:valueEnd])
		if key != "" {
			args[key] = coerceParamValue(value)
		}

		body = body[valueEnd+len("</parameter>"):]
	}

	return args
}

// findParameterOpenTag finds the index of the next <parameter= opening tag,
// using case-insensitive matching to match the (?si) regex behavior.
func findParameterOpenTag(body string) int {
	lower := strings.ToLower(body)
	idx := strings.Index(lower, "<parameter=")
	return idx
}

// findParameterCloseTag scans from valueStart to find the matching </parameter>
// closing tag, handling nested <parameter=...> tags with depth tracking.
// Uses case-insensitive matching to match the (?si) regex behavior.
// Returns the index of the start of </parameter>, or -1 if not found.
func findParameterCloseTag(body string, valueStart int) int {
	depth := 1
	scanPos := valueStart
	lower := strings.ToLower(body)

	for scanPos < len(body) {
		nextOpen := strings.Index(lower[scanPos:], "<parameter=")
		nextClose := strings.Index(lower[scanPos:], "</parameter>")

		if nextClose == -1 {
			break
		}

		if nextOpen != -1 && nextOpen < nextClose {
			depth++
			scanPos += nextOpen + len("<parameter=")
		} else {
			depth--
			if depth == 0 {
				return scanPos + nextClose
			}
			scanPos += nextClose + len("</parameter>")
		}
	}

	return -1
}

// coerceParamValue attempts to parse a string as JSON; falls back to plain string.
// Empty strings are returned as-is.
func coerceParamValue(value string) interface{} {
	if value == "" {
		return ""
	}

	var jsonVal interface{}
	if err := json.Unmarshal([]byte(value), &jsonVal); err == nil {
		// Preserve JSON null as a sentinel string rather than Go nil.
		// Go nil marshals as JSON null which is ambiguous — the model may
		// have intended the literal string "null" rather than a JSON null value.
		// Keeping it as a string ensures round-trip consistency.
		if jsonVal == nil {
			return value
		}
		return jsonVal
	}
	return value
}

// containsFunctionTag checks for <function= in a case-insensitive manner,
// matching the (?si) behavior of functionParamPattern.
func containsFunctionTag(content string) bool {
	return strings.Contains(strings.ToLower(content), "<function=")
}

// truncateToValidUTF8 truncates content to at most maxLen bytes, backing up
// if the cut would split a multi-byte UTF-8 character.
func truncateToValidUTF8(content string, maxLen int) string {
	if len(content) <= maxLen {
		return content
	}
	for maxLen > 0 && !utf8.RuneStart(content[maxLen]) {
		maxLen--
	}
	return content[:maxLen]
}

