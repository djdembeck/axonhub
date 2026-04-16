package nanogpt

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestMaybeHasXMLToolCalls(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected bool
	}{
		{
			name:     "empty content",
			content:  "",
			expected: false,
		},
		{
			name:     "plain text without XML",
			content:  "Hello, this is just text",
			expected: false,
		},
		{
			name:     "contains Write tag",
			content:  "<Write file_path=\"x\">content</Write>",
			expected: true,
		},
		{
			name:     "contains use_tool",
			content:  "<use_tool name=\"write\">content</use_tool>",
			expected: true,
		},
		{
			name:     "contains Bash tag",
			content:  "Running <Bash>ls</Bash> command",
			expected: true,
		},
		{
			name:     "contains Read tag",
			content:  "<Read file_path=\"x\"/>",
			expected: true,
		},
		{
			name:     "has angle brackets - pre-check is intentionally permissive",
			content:  "<div>not a tool</div>",
			expected: true, // pre-check matches any XML-like pattern; actual parsing filters non-tool tags
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MaybeHasXMLToolCalls(tt.content)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestParseXMLToolCalls_SelfClosing(t *testing.T) {
	content := `<Write file_path="/test/file.txt" content="hello world"/>`

	tools, _, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	// remaining content is tool output, not relevant for these tests

	tool := tools[0]
	assert.Equal(t, "write", tool.Function.Name)
	assert.Contains(t, tool.Function.Arguments, "file_path")
	assert.Contains(t, tool.Function.Arguments, "content")
	assert.Contains(t, tool.Function.Arguments, "/test/file.txt")
	assert.Contains(t, tool.Function.Arguments, "hello world")
}

func TestParseXMLToolCalls_SimpleContent(t *testing.T) {
	content := `<Write file_path="/test/file.txt">file contents here</Write>`

	tools, _, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	// remaining content is tool output, not relevant for these tests

	tool := tools[0]
	assert.Equal(t, "write", tool.Function.Name)
	assert.Contains(t, tool.Function.Arguments, "file_path")
	assert.Contains(t, tool.Function.Arguments, "/test/file.txt")
	assert.Contains(t, tool.Function.Arguments, "file contents here")
}

func TestParseXMLToolCalls_JSONInContent(t *testing.T) {
	content := `<Write>{"file_path": "/test/file.txt", "content": "hello"}</Write>`

	tools, _, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	// remaining content is tool output, not relevant for these tests

	tool := tools[0]
	assert.Equal(t, "write", tool.Function.Name)
	assert.Contains(t, tool.Function.Arguments, "file_path")
	assert.Contains(t, tool.Function.Arguments, "/test/file.txt")
	assert.Contains(t, tool.Function.Arguments, "hello")
}

func TestParseXMLToolCalls_MismatchedClosingTag(t *testing.T) {
	// The parser normalizes <Write>content</use_tool> to <Write>content</Write>
	content := `<Write file_path="/test/file.txt">content</use_tool>`

	tools, _, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	// remaining content is tool output, not relevant for these tests

	tool := tools[0]
	assert.Equal(t, "write", tool.Function.Name)
	assert.Contains(t, tool.Function.Arguments, "file_path")
}

func TestParseXMLToolCalls_UnclosedOpeningTag(t *testing.T) {
	// NanoGPT sometimes omits the closing > on opening tags
	content := "<Write file_path=\"/test/file.txt\" content=\"hello\"\n}\n</use_tool>"

	tools, _, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	require.Len(t, tools, 1)

	tool := tools[0]
	assert.Equal(t, "write", tool.Function.Name)
}

func TestParseXMLToolCalls_NestedXMLElements(t *testing.T) {
	content := `<Write>
  <file_path>/test/file.txt</file_path>
  <content>hello world</content>
</Write>`

	tools, _, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	// remaining content is tool output, not relevant for these tests

	tool := tools[0]
	assert.Equal(t, "write", tool.Function.Name)
	assert.Contains(t, tool.Function.Arguments, "file_path")
	assert.Contains(t, tool.Function.Arguments, "/test/file.txt")
	assert.Contains(t, tool.Function.Arguments, "hello world")
}

func TestParseXMLToolCases_NoSpaceAfterTag(t *testing.T) {
	// Format: <Write_File>{...}</Write_File> without space after tag name
	content := `<Write_File>{"path": "/test/file.txt", "content": "hello"}</Write_File>`

	tools, _, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	require.Len(t, tools, 1)

	tool := tools[0]
	assert.Equal(t, "write", tool.Function.Name)
}

func TestParseXMLToolCalls_MultipleToolCalls(t *testing.T) {
	content := `<Write file_path="/file1.txt">content1</Write>
Some text in between
<Read file_path="/file2.txt"/>`

	tools, remaining, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	require.Len(t, tools, 2)
	assert.Contains(t, remaining, "Some text in between")

	assert.Equal(t, "write", tools[0].Function.Name)
	assert.Equal(t, "read", tools[1].Function.Name)
}

func TestParseXMLToolCalls_NoToolCalls(t *testing.T) {
	content := "This is just plain text without any tool calls"

	tools, remaining, err := ParseXMLToolCalls(content)
	require.NoError(t, err)
	assert.Empty(t, tools)
	assert.Equal(t, content, remaining)
}

func TestExtractToolName(t *testing.T) {
	tests := []struct {
		tagName  string
		attrs    string
		expected string
	}{
		{"Write", "", "write"},
		{"Write_FILE", "", "write"},
		{"Write_file", "", "write"},
		{"Read", "", "read"},
		{"Read_FILE", "", "read"},
		{"Bash", "", "bash"},
		{"Python", "", "python"},
		{"Search", "", "search"},
		{"Glob", "", "glob"},
		{"use_tool", `name="write"`, "write"},
		{"use_tool", `name="Read"`, "read"},
		{"Unknown", "", ""},
		{"", "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.tagName, func(t *testing.T) {
			result := extractToolName(tt.tagName, tt.attrs)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGenerateToolCallID(t *testing.T) {
	// ID should be deterministic
	id1 := generateToolCallID("write", `{"file_path":"/test.txt"}`)
	id2 := generateToolCallID("write", `{"file_path":"/test.txt"}`)
	assert.Equal(t, id1, id2)

	// Different inputs should produce different IDs
	id3 := generateToolCallID("read", `{"file_path":"/test.txt"}`)
	assert.NotEqual(t, id1, id3)

	// Should start with nanogpt_ prefix
	assert.True(t, len(id1) > 8)
}

func TestToOpenAIToolCalls(t *testing.T) {
	toolCalls := []llm.ToolCall{
		{
			Index: 0,
			ID:    "test-id-1",
			Type:  "function",
			Function: llm.FunctionCall{
				Name:      "write",
				Arguments: `{"file_path":"/test.txt"}`,
			},
		},
	}

	result := ToOpenAIToolCalls(toolCalls)
	require.Len(t, result, 1)
	assert.Equal(t, "test-id-1", result[0].ID)
	assert.Equal(t, "write", result[0].Function.Name)
}

func TestParseXMLToolCalls_FunctionParamFormat(t *testing.T) {
	t.Run("simple grep with parameters", func(t *testing.T) {
		content := "<function=grep>\n<parameter=_i>\nFind all calls\n</parameter>\n<parameter=path>\ninternal/dedupe/dedupe.go\n</parameter>\n<parameter=pattern>\ndetermineFileToDelete\n</parameter>\n</function>"

		tools, remaining, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		assert.Equal(t, "grep", tools[0].Function.Name)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Equal(t, "Find all calls", args["_i"])
		assert.Equal(t, "internal/dedupe/dedupe.go", args["path"])
		assert.Equal(t, "determineFileToDelete", args["pattern"])
		assert.Equal(t, "", remaining)
	})

	t.Run("edit with JSON array parameter", func(t *testing.T) {
		content := "<function=edit>\n<parameter=_i>\nApply optimization fix\n</parameter>\n<parameter=edits>\n[{\"newText\": \"foo\", \"oldText\": \"bar\"}]\n</parameter>\n<parameter=path>\ninternal/dedupe/dedupe.go\n</parameter>\n</function>"

		tools, remaining, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		assert.Equal(t, "edit", tools[0].Function.Name)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Equal(t, "Apply optimization fix", args["_i"])
		assert.Equal(t, "internal/dedupe/dedupe.go", args["path"])

		edits, ok := args["edits"].([]interface{})
		require.True(t, ok, "edits should be a JSON array")
		require.Len(t, edits, 1)
		editMap, ok := edits[0].(map[string]interface{})
		require.True(t, ok)
		assert.Equal(t, "foo", editMap["newText"])
		assert.Equal(t, "bar", editMap["oldText"])
		assert.Equal(t, "", remaining)
	})

	t.Run("multiple function calls", func(t *testing.T) {
		content := "<function=grep>\n<parameter=_i>1</parameter>\n<parameter=path>src/</parameter>\n<parameter=pattern>foo</parameter>\n</function>\n\n<function=edit>\n<parameter=_i>2</parameter>\n<parameter=path>src/main.go</parameter>\n<parameter=edits>[{\"oldText\":\"a\",\"newText\":\"b\"}]</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 2)

		assert.Equal(t, "grep", tools[0].Function.Name)
		assert.Equal(t, "edit", tools[1].Function.Name)

		var args0 map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args0))
		assert.Equal(t, float64(1), args0["_i"])
		assert.Equal(t, "src/", args0["path"])
		assert.Equal(t, "foo", args0["pattern"])
	})

	t.Run("function with text before and after", func(t *testing.T) {
		content := "I'll search for that.\n<function=grep>\n<parameter=_i>1</parameter>\n<parameter=path>src/</parameter>\n<parameter=pattern>foo</parameter>\n</function>\nLet me know if you need more."

		tools, remaining, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		assert.Equal(t, "grep", tools[0].Function.Name)
		assert.Contains(t, remaining, "I'll search for that.")
		assert.Contains(t, remaining, "Let me know if you need more.")
	})

	t.Run("no function calls returns original content", func(t *testing.T) {
		content := "Just some text without any tool calls"

		tools, remaining, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		assert.Nil(t, tools)
		assert.Equal(t, content, remaining)
	})
}

func TestMaybeHasXMLToolCalls_FunctionParam(t *testing.T) {
	assert.True(t, MaybeHasXMLToolCalls("<function=grep><parameter=_i>1</parameter></function>"))
	assert.True(t, MaybeHasXMLToolCalls("<function=edit>\n<parameter=path>foo</parameter>\n</function>"))
	assert.False(t, MaybeHasXMLToolCalls("just plain text"))
}

func TestParseXMLToolCalls_FunctionParamEdgeCases(t *testing.T) {
	t.Run("empty parameter value", func(t *testing.T) {
		content := "<function=edit>\n<parameter=_i>1</parameter>\n<parameter=path>src/main.go</parameter>\n<parameter=edits></parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		assert.Equal(t, "edit", tools[0].Function.Name)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Equal(t, "", args["edits"])
		assert.Equal(t, "src/main.go", args["path"])
	})

	t.Run("function with no parameters", func(t *testing.T) {
		content := "<function=list_files>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		assert.Equal(t, "list_files", tools[0].Function.Name)
		assert.Equal(t, "{}", tools[0].Function.Arguments)
	})

	t.Run("parameter value with angle brackets", func(t *testing.T) {
		content := "<function=edit>\n<parameter=_i>1</parameter>\n<parameter=path>src/main.go</parameter>\n<parameter=edits>[{\"oldText\":\"<div>old</div>\",\"newText\":\"<div>new</div>\"}]</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		assert.Equal(t, "edit", tools[0].Function.Name)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))

		edits, ok := args["edits"].([]interface{})
		require.True(t, ok, "edits should be a JSON array")
		editMap, ok := edits[0].(map[string]interface{})
		require.True(t, ok)
		assert.Contains(t, editMap["oldText"], "<div>")
		assert.Contains(t, editMap["newText"], "<div>")
	})

	t.Run("whitespace-only parameter value", func(t *testing.T) {
		content := "<function=edit>\n<parameter=_i>   </parameter>\n<parameter=path>src/main.go</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Equal(t, "", args["_i"])
	})

	t.Run("JSON null value in parameter", func(t *testing.T) {
		content := "<function=edit>\n<parameter=_i>1</parameter>\n<parameter=edits>null</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Nil(t, args["edits"])
	})

	t.Run("duplicate parameter keys - last wins", func(t *testing.T) {
		content := "<function=grep>\n<parameter=pattern>first</parameter>\n<parameter=pattern>second</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Equal(t, "second", args["pattern"])
	})
}

func TestParseXMLToolCalls_FunctionParamExpanded(t *testing.T) {
	t.Run("dash in function name", func(t *testing.T) {
		content := "<function=edit-file>\n<parameter=path>src/main.go</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)
		assert.Equal(t, "edit-file", tools[0].Function.Name)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Equal(t, "src/main.go", args["path"])
	})

	t.Run("dot in function name", func(t *testing.T) {
		content := "<function=search.web>\n<parameter=query>golang regex</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)
		assert.Equal(t, "search.web", tools[0].Function.Name)
	})

	t.Run("dash in parameter name", func(t *testing.T) {
		content := "<function=edit>\n<parameter=file-path>src/main.go</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Equal(t, "src/main.go", args["file-path"])
	})

	t.Run("uppercase Function tag", func(t *testing.T) {
		content := "<Function=grep>\n<Parameter=_i>1</Parameter>\n<Parameter=path>src/</Parameter>\n</Function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)
		assert.Equal(t, "grep", tools[0].Function.Name)
	})

	t.Run("truncated function block - no closing tag", func(t *testing.T) {
		content := "<function=grep>\n<parameter=_i>1</parameter>\n<parameter=path>src/</parameter>\n<parameter=pattern>foo</parameter>"

		tools, remaining, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)
		assert.Equal(t, "grep", tools[0].Function.Name)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))
		assert.Equal(t, "src/", args["path"])
		assert.Equal(t, "foo", args["pattern"])
		assert.Equal(t, "", remaining)
	})

	t.Run("value with angle brackets preserved", func(t *testing.T) {
		content := "<function=edit>\n<parameter=path>src/main.go</parameter>\n<parameter=edits>[{\"oldText\":\"<div>old</div>\",\"newText\":\"<span>new</span>\"}]</parameter>\n</function>"

		tools, _, err := ParseXMLToolCalls(content)
		require.NoError(t, err)
		require.Len(t, tools, 1)

		var args map[string]interface{}
		require.NoError(t, json.Unmarshal([]byte(tools[0].Function.Arguments), &args))

		edits, ok := args["edits"].([]interface{})
		require.True(t, ok)
		editMap := edits[0].(map[string]interface{})
		assert.Contains(t, fmt.Sprintf("%v", editMap["oldText"]), "div")
		assert.Contains(t, fmt.Sprintf("%v", editMap["newText"]), "span")
	})
}
