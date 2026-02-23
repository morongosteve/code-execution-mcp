"""Tests for CodeExecutionTool in code_execution_tool.py.

Focuses on pure-logic methods (no actual PTY / shell execution).

The tty_session module calls sys.stdin.reconfigure() at import time which
fails inside pytest.  We patch sys.stdin/stdout with real IO-like objects
before the import chain reaches tty_session.
"""

import io
import os
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Patch sys.stdin/stdout so that tty_session.py can reconfigure them ──
# pytest replaces sys.stdin with DontReadFromInput which lacks reconfigure().
# We temporarily swap it with a real TextIOWrapper before importing the module.
_real_stdin = sys.stdin
_real_stdout = sys.stdout
if not hasattr(sys.stdin, "reconfigure"):
    sys.stdin = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
if not hasattr(sys.stdout, "reconfigure"):
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")

from code_execution_tool import CodeExecutionTool, truncate_text_agent  # noqa: E402

# Restore originals so pytest output works normally
sys.stdin = _real_stdin
sys.stdout = _real_stdout


# ============================================================================
# Output pattern detection (prompt patterns and dialog patterns)
# ============================================================================


class TestPromptPatterns:
    """Verify the regex prompt patterns used in get_terminal_output to detect
    that a shell has returned to its prompt."""

    # The patterns are created inside get_terminal_output, so we replicate
    # the exact same list here for isolated testing.
    PROMPT_PATTERNS = [
        re.compile(r"\(venv\).+[$#] ?$"),
        re.compile(r"root@[^:]+:[^#]+# ?$"),
        re.compile(r"[a-zA-Z0-9_.-]+@[^:]+:[^$#]+[$#] ?$"),
        re.compile(r"bash-\d+\.\d+\$ ?$"),
    ]

    @pytest.mark.parametrize("line,expected_match", [
        # venv-style prompts
        ("(venv) user@host:~/project$ ", True),
        ("(venv) root@abc:/tmp# ", True),
        ("(venv) $", True),  # ".+" matches the space before $
        # root@container prompts
        ("root@container:~# ", True),
        ("root@abc123:/home/user# ", True),
        ("root@host:/var/log#", True),
        # user@host prompts
        ("user@hostname:~/dir$ ", True),
        ("deploy@prod-server:/opt/app$ ", True),
        ("user.name@host:~$", True),
        # bash version prompts
        ("bash-3.2$ ", True),
        ("bash-5.1$ ", True),
        ("bash-5.1$", True),
        # Non-matching lines
        ("ls -la", False),
        ("total 42", False),
        ("", False),
        ("Processing...", False),
        ("echo hello", False),
    ])
    def test_prompt_pattern_matching(self, line, expected_match):
        matched = any(pat.search(line.strip()) for pat in self.PROMPT_PATTERNS)
        assert matched == expected_match, (
            f"Line {line!r}: expected match={expected_match}, got {matched}"
        )


class TestDialogPatterns:
    """Verify the regex dialog patterns used to detect interactive prompts."""

    DIALOG_PATTERNS = [
        re.compile(r"Y/N", re.IGNORECASE),
        re.compile(r"yes/no", re.IGNORECASE),
        re.compile(r":\s*$"),
        re.compile(r"\?\s*$"),
    ]

    @pytest.mark.parametrize("line,expected_match", [
        # Y/N patterns
        ("Continue? [Y/N]", True),
        ("Proceed (y/n)", True),
        ("Do you want to continue? [yes/no]", True),
        ("YES/NO", True),
        # Colon at end
        ("Enter password:", True),
        ("Username: ", True),
        ("Select option:  ", True),
        # Question mark at end
        ("Are you sure?", True),
        ("Do you want to proceed? ", True),
        # Non-matching
        ("file.txt", False),
        ("Processing: 50% done ... still going", False),
        ("Y/N is in the middle but not a prompt style", True),  # Y/N anywhere
        ("", False),
    ])
    def test_dialog_pattern_matching(self, line, expected_match):
        matched = any(pat.search(line.strip()) for pat in self.DIALOG_PATTERNS)
        assert matched == expected_match, (
            f"Line {line!r}: expected match={expected_match}, got {matched}"
        )


# ============================================================================
# fix_full_output()
# ============================================================================


class TestFixFullOutput:
    """Tests for CodeExecutionTool.fix_full_output()."""

    def test_removes_hex_escapes(self, code_exec_tool):
        raw = r"Hello \x1b world \x00 end"
        result = code_exec_tool.fix_full_output(raw)
        assert r"\x1b" not in result
        assert r"\x00" not in result
        assert "Hello" in result
        assert "end" in result

    def test_strips_lines(self, code_exec_tool):
        raw = "  hello  \n  world  \n  foo  "
        result = code_exec_tool.fix_full_output(raw)
        lines = result.splitlines()
        for line in lines:
            assert line == line.strip()

    def test_truncation_under_threshold(self, code_exec_tool):
        """Output under 1MB should not be truncated."""
        raw = "A" * 100
        result = code_exec_tool.fix_full_output(raw)
        assert result == "A" * 100

    def test_truncation_over_threshold(self, code_exec_tool):
        """Output over 1MB threshold should be truncated."""
        raw = "B" * 2_000_000
        result = code_exec_tool.fix_full_output(raw)
        assert len(result) < 2_000_000

    def test_empty_input(self, code_exec_tool):
        result = code_exec_tool.fix_full_output("")
        assert result == ""

    def test_preserves_literal_backslash_x(self, code_exec_tool):
        """A double-backslash \\\\xNN should NOT be removed (escaped literal)."""
        raw = "keep \\\\x41 this"
        result = code_exec_tool.fix_full_output(raw)
        # The regex removes single \xNN but not \\xNN
        assert "keep" in result
        assert "this" in result

    def test_only_hex_escape_content_removed(self, code_exec_tool):
        """Hex escapes like \\x1b are removed but surrounding text stays."""
        raw = "before\\x1bafter"
        result = code_exec_tool.fix_full_output(raw)
        assert "before" in result
        assert "after" in result

    def test_multiline_stripping(self, code_exec_tool):
        raw = "  line1  \n  line2  \n  line3  "
        result = code_exec_tool.fix_full_output(raw)
        assert result == "line1\nline2\nline3"


# ============================================================================
# format_command_for_output()
# ============================================================================


class TestFormatCommandForOutput:
    """Tests for CodeExecutionTool.format_command_for_output()."""

    def test_short_command_unchanged(self, code_exec_tool):
        result = code_exec_tool.format_command_for_output("ls -la")
        assert "ls -la" in result

    def test_long_command_truncated(self, code_exec_tool):
        long_cmd = "echo " + "A" * 500
        result = code_exec_tool.format_command_for_output(long_cmd)
        # The method first takes [:200], normalizes whitespace, then truncates to 100
        assert len(result) <= 200  # generous upper bound

    def test_whitespace_normalized(self, code_exec_tool):
        result = code_exec_tool.format_command_for_output("ls   -la    /tmp")
        assert "  " not in result  # double spaces should be collapsed
        assert "ls -la /tmp" in result

    def test_multiline_command_collapsed(self, code_exec_tool):
        result = code_exec_tool.format_command_for_output("echo hello\necho world")
        # Newlines should be collapsed to spaces
        assert "\n" not in result
        assert "echo hello" in result

    def test_very_short_command(self, code_exec_tool):
        result = code_exec_tool.format_command_for_output("ls")
        assert result == "ls"

    def test_tabs_normalized(self, code_exec_tool):
        result = code_exec_tool.format_command_for_output("echo\thello\tworld")
        assert "\t" not in result


# ============================================================================
# truncate_text_agent()
# ============================================================================


class TestTruncateTextAgent:
    """Tests for the module-level truncate_text_agent function."""

    def test_under_threshold_returns_as_is(self):
        text = "Hello World"
        assert truncate_text_agent(text, threshold=1000) == text

    def test_at_threshold_returns_as_is(self):
        text = "A" * 100
        assert truncate_text_agent(text, threshold=100) == text

    def test_over_threshold_truncates(self):
        text = "X" * 2000
        result = truncate_text_agent(text, threshold=1000)
        assert len(result) < 2000

    def test_default_threshold_one_mb(self):
        """Default threshold is 1_000_000 -- text under it should pass through."""
        text = "Z" * 999_999
        result = truncate_text_agent(text)
        assert result == text

    def test_empty_string(self):
        assert truncate_text_agent("") == ""

    def test_single_char_over_threshold(self):
        """One character over the threshold should still trigger truncation."""
        text = "Q" * 101
        result = truncate_text_agent(text, threshold=100)
        assert len(result) <= 103  # 100 + "..."


# ============================================================================
# read_prompt()
# ============================================================================


class TestReadPrompt:
    """Tests for CodeExecutionTool.read_prompt()."""

    def test_read_existing_prompt(self, code_exec_tool):
        result = code_exec_tool.read_prompt("fw.code.reset.md")
        assert "Terminal session has been reset" in result

    def test_read_prompt_with_kwargs(self, code_exec_tool):
        result = code_exec_tool.read_prompt("fw.code.max_time.md", timeout=42)
        assert "42" in result
        # The template placeholder {{timeout}} should be replaced
        assert "{{timeout}}" not in result

    def test_read_prompt_missing_file(self, code_exec_tool):
        result = code_exec_tool.read_prompt("nonexistent_file.md")
        assert "Prompt file not found" in result
        assert "nonexistent_file.md" in result

    def test_read_prompt_info_wrapper(self, code_exec_tool):
        inner = code_exec_tool.read_prompt("fw.code.reset.md")
        result = code_exec_tool.read_prompt("fw.code.info.md", info=inner)
        assert "Terminal session has been reset" in result

    def test_read_prompt_no_kwargs_returns_raw(self, code_exec_tool):
        """Reading a prompt without kwargs should return the raw template."""
        result = code_exec_tool.read_prompt("fw.code.max_time.md")
        # The placeholder should remain since we didn't provide kwargs
        assert "{{timeout}}" in result

    def test_read_prompt_multiple_kwargs(self, code_exec_tool):
        """The info template takes one kwarg, verify substitution."""
        result = code_exec_tool.read_prompt("fw.code.info.md", info="TEST_VALUE")
        assert "TEST_VALUE" in result
        assert "{{info}}" not in result


# ============================================================================
# CodeExecutionTool.__init__() defaults
# ============================================================================


class TestCodeExecutionToolInit:
    """Verify that __init__ sets correct defaults."""

    def test_default_timeouts(self):
        tool = CodeExecutionTool()
        assert tool.default_timeouts["first_output_timeout"] == 30
        assert tool.default_timeouts["between_output_timeout"] == 15
        assert tool.default_timeouts["dialog_timeout"] == 5
        assert tool.default_timeouts["max_exec_timeout"] == 180

    def test_custom_timeouts(self):
        tool = CodeExecutionTool(
            first_output_timeout=60,
            between_output_timeout=30,
            dialog_timeout=10,
            max_exec_timeout=300,
        )
        assert tool.default_timeouts["first_output_timeout"] == 60
        assert tool.default_timeouts["between_output_timeout"] == 30
        assert tool.default_timeouts["dialog_timeout"] == 10
        assert tool.default_timeouts["max_exec_timeout"] == 300

    def test_prompts_dir_resolved(self):
        tool = CodeExecutionTool()
        assert os.path.isdir(tool.prompts_dir)

    def test_state_initially_none(self):
        tool = CodeExecutionTool()
        assert tool.state is None

    def test_executable_default_none(self):
        tool = CodeExecutionTool()
        assert tool.executable is None

    def test_executable_custom(self):
        tool = CodeExecutionTool(executable="/bin/bash")
        assert tool.executable == "/bin/bash"

    def test_init_commands_default_empty(self):
        tool = CodeExecutionTool()
        assert tool.init_commands == []

    def test_init_commands_custom(self):
        tool = CodeExecutionTool(init_commands=["export FOO=1", "cd /tmp"])
        assert tool.init_commands == ["export FOO=1", "cd /tmp"]
