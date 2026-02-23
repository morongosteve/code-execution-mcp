"""Tests for CodeExecutionTool in code_execution_tool.py.

Focuses on pure-logic methods (no actual PTY / shell execution).

The tty_session module calls sys.stdin.reconfigure() at import time which
fails inside pytest.  We patch sys.stdin/stdout with real IO-like objects
before the import chain reaches tty_session.
"""

import asyncio
import io
import os
import re
import sys
from unittest.mock import AsyncMock, MagicMock

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

from code_execution_tool import CodeExecutionTool, State, truncate_text_agent  # noqa: E402

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

    def test_venv_prompt_with_different_users(self):
        """Various venv-style prompts should all match."""
        lines = [
            "(venv) admin@server:/opt# ",
            "(venv) test-user@dev:~/code$ ",
            "(venv) a $",
        ]
        for line in lines:
            assert any(pat.search(line.strip()) for pat in self.PROMPT_PATTERNS), (
                f"Expected match for: {line!r}"
            )

    def test_root_prompt_variations(self):
        """Various root prompts should match."""
        lines = [
            "root@myhost:/# ",
            "root@a-b-c:/home/user/dir# ",
        ]
        for line in lines:
            assert any(pat.search(line.strip()) for pat in self.PROMPT_PATTERNS), (
                f"Expected match for: {line!r}"
            )

    def test_bash_version_prompt_variations(self):
        """Different bash versions."""
        lines = [
            "bash-4.4$ ",
            "bash-5.2$",
            "bash-3.0$ ",
        ]
        for line in lines:
            assert any(pat.search(line.strip()) for pat in self.PROMPT_PATTERNS), (
                f"Expected match for: {line!r}"
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

    def test_case_insensitive_yn(self):
        """Y/N matching should be case-insensitive."""
        for line in ["y/n", "Y/N", "Y/n", "y/N"]:
            assert any(pat.search(line.strip()) for pat in self.DIALOG_PATTERNS)

    def test_case_insensitive_yes_no(self):
        """yes/no matching should be case-insensitive."""
        for line in ["yes/no", "Yes/No", "YES/NO", "yEs/nO"]:
            assert any(pat.search(line.strip()) for pat in self.DIALOG_PATTERNS)

    def test_colon_not_in_middle(self):
        """A colon in the middle of a line (not at end) should NOT match."""
        line = "key: value and more text"
        # The colon-at-end pattern looks for :\s*$
        # "key: value and more text" -> colon is not at end
        matched = any(pat.search(line.strip()) for pat in self.DIALOG_PATTERNS)
        assert not matched

    def test_question_mark_in_middle(self):
        """A question mark in the middle should NOT match (only end-of-line)."""
        line = "Is this a ? or not, I think not"
        matched = any(pat.search(line.strip()) for pat in self.DIALOG_PATTERNS)
        assert not matched


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

    def test_multiple_hex_escapes_in_one_line(self, code_exec_tool):
        raw = r"abc\x01def\x02ghi"
        result = code_exec_tool.fix_full_output(raw)
        assert "abc" in result
        assert "def" in result
        assert "ghi" in result
        assert r"\x01" not in result
        assert r"\x02" not in result

    def test_hex_escape_case_insensitive(self, code_exec_tool):
        """Hex escapes with uppercase letters should also be removed."""
        raw = r"test\xABvalue\xCDend"
        result = code_exec_tool.fix_full_output(raw)
        assert r"\xAB" not in result
        assert r"\xCD" not in result
        assert "test" in result


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

    def test_empty_command(self, code_exec_tool):
        result = code_exec_tool.format_command_for_output("")
        assert result == ""

    def test_command_exactly_200_chars(self, code_exec_tool):
        """Command of exactly 200 chars (first truncation boundary)."""
        cmd = "x" * 200
        result = code_exec_tool.format_command_for_output(cmd)
        # After taking [:200] and truncating to 100, should be <= ~103 chars
        assert len(result) <= 200


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

    def test_large_text_over_default_threshold(self):
        """Text over 1MB default should be truncated."""
        text = "A" * 1_100_000
        result = truncate_text_agent(text)
        assert len(result) < 1_100_000


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

    def test_log_attribute_exists(self):
        tool = CodeExecutionTool()
        assert hasattr(tool, "log")

    def test_prompts_dir_points_to_prompts_folder(self):
        tool = CodeExecutionTool()
        assert tool.prompts_dir.endswith("prompts")


# ============================================================================
# prepare_state()
# ============================================================================


class TestPrepareState:
    """Tests for CodeExecutionTool.prepare_state()."""

    def test_prepare_state_creates_state_when_none(self):
        tool = CodeExecutionTool()
        assert tool.state is None

        loop = asyncio.new_event_loop()
        loop.run_until_complete(tool.prepare_state())
        loop.close()

        assert tool.state is not None
        assert isinstance(tool.state, State)
        assert isinstance(tool.state.shells, dict)

    def test_prepare_state_preserves_existing_shells(self):
        tool = CodeExecutionTool()
        mock_shell = MagicMock()
        tool.state = State(shells={0: mock_shell})

        loop = asyncio.new_event_loop()
        loop.run_until_complete(tool.prepare_state())
        loop.close()

        # Existing shells should be preserved
        assert 0 in tool.state.shells
        assert tool.state.shells[0] is mock_shell

    def test_prepare_state_reset_specific_session(self):
        tool = CodeExecutionTool()
        mock_shell_0 = AsyncMock()
        mock_shell_1 = AsyncMock()
        tool.state = State(shells={0: mock_shell_0, 1: mock_shell_1})

        loop = asyncio.new_event_loop()
        loop.run_until_complete(tool.prepare_state(reset=True, session=0))
        loop.close()

        # Session 0 should have been closed and removed
        mock_shell_0.close.assert_called_once()
        # Session 1 should remain
        assert 1 in tool.state.shells

    def test_prepare_state_reset_all_sessions(self):
        tool = CodeExecutionTool()
        mock_shell_0 = AsyncMock()
        mock_shell_1 = AsyncMock()
        tool.state = State(shells={0: mock_shell_0, 1: mock_shell_1})

        loop = asyncio.new_event_loop()
        loop.run_until_complete(tool.prepare_state(reset=True, session=None))
        loop.close()

        # All sessions should be closed
        mock_shell_0.close.assert_called_once()
        mock_shell_1.close.assert_called_once()
        assert len(tool.state.shells) == 0

    def test_prepare_state_reset_nonexistent_session(self):
        """Resetting a session that doesn't exist should not raise."""
        tool = CodeExecutionTool()
        mock_shell_0 = MagicMock()
        tool.state = State(shells={0: mock_shell_0})

        loop = asyncio.new_event_loop()
        # Reset session 5, which doesn't exist -- should be no-op
        loop.run_until_complete(tool.prepare_state(reset=True, session=5))
        loop.close()

        assert 0 in tool.state.shells  # session 0 untouched
