"""Security hardening utilities for code-execution-mcp.

Provides:
1. Input validation - dangerous command pattern detection, model ID sanitization
2. Rate limiting - token bucket rate limiters for commands and model loads
3. Audit logging - structured audit logger for security-relevant events
4. Resource limits - memory/disk space monitoring, output size caps
5. Output sanitization - secret/credential detection and redaction
"""

import logging
import os
import re
import time

logger = logging.getLogger("code-execution-mcp.security")

# ============================================================================
# 1. Input Validation
# ============================================================================

# Dangerous command patterns that should be warned about
# (not blocked - that's the MCP client's job)
DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/\b"),          # rm -rf /
    re.compile(r"\bmkfs\b"),                     # filesystem formatting
    re.compile(r"\bdd\s+if=.*of=/dev/\b"),      # raw disk writes
    re.compile(r":\(\)\{.*\|.*&\s*\};:"),        # fork bomb
    re.compile(r">\s*/dev/sd[a-z]"),             # redirect to disk device
    re.compile(r"\bchmod\s+-R\s+777\s+/\b"),    # chmod 777 /
]


def validate_command(command: str) -> tuple[bool, str]:
    """Validate a command for obvious dangerous patterns.

    Returns (is_safe, warning_message). Does NOT block execution - just warns.
    """
    if not command or not command.strip():
        return False, "Command is empty."

    warnings = []
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(command):
            warnings.append(f"Dangerous pattern detected: {pattern.pattern}")

    if warnings:
        msg = "; ".join(warnings)
        logger.warning("Command validation warning: %s (command: %.200s)", msg, command)
        return False, msg

    return True, ""


def sanitize_model_id(model_id: str) -> tuple[bool, str]:
    """Validate a HuggingFace model ID format.

    Model IDs should match org/model-name pattern with no path traversal.

    Returns (is_valid, error_message).
    """
    if not model_id or not model_id.strip():
        return False, "Model ID cannot be empty."

    # Reject path traversal attempts
    if ".." in model_id:
        return False, "Model ID contains path traversal sequence '..'."

    # Reject absolute paths
    if model_id.startswith("/") or model_id.startswith("\\"):
        return False, "Model ID must not be an absolute path."

    # Reject null bytes
    if "\x00" in model_id:
        return False, "Model ID contains null bytes."

    # Must match pattern: optional-org/model-name or just model-name
    # Allows alphanumeric, hyphens, underscores, dots, and single slashes
    valid_pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+(/[a-zA-Z0-9_\-\.]+)*$")
    if not valid_pattern.match(model_id):
        return False, (
            f"Invalid model ID format: '{model_id}'. "
            "Expected pattern like 'org/model-name' or 'model-name'."
        )

    return True, ""


# ============================================================================
# 2. Rate Limiting
# ============================================================================

class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._requests: list[float] = []
        self._max_requests = max_requests
        self._window = window_seconds

    def check(self) -> tuple[bool, str]:
        """Check if request is allowed. Returns (allowed, message)."""
        now = time.monotonic()

        # Evict expired entries outside the window
        self._requests = [
            t for t in self._requests if now - t < self._window
        ]

        if len(self._requests) >= self._max_requests:
            wait_time = self._window - (now - self._requests[0])
            return False, (
                f"Rate limit exceeded: {self._max_requests} requests per "
                f"{self._window}s. Try again in {wait_time:.1f}s."
            )

        self._requests.append(now)
        return True, ""

    def reset(self) -> None:
        """Reset the rate limiter, clearing all tracked requests."""
        self._requests.clear()


# Global rate limiters
execution_limiter = RateLimiter(max_requests=60, window_seconds=60)
model_load_limiter = RateLimiter(max_requests=10, window_seconds=300)


# ============================================================================
# 3. Audit Logging
# ============================================================================

class AuditLogger:
    """Structured audit logger for security-relevant events."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("code-execution-mcp.audit")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [AUDIT] %(message)s'
        ))
        if not self._logger.handlers:
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def log_command(self, session: int, command: str, user: str = "mcp-client") -> None:
        """Log a command execution event."""
        self._logger.info(
            "user=%s session=%d action=execute_command command=%.200s",
            user, session, command,
        )

    def log_model_load(self, model_id: str, backend: str) -> None:
        """Log a model load event."""
        self._logger.info(
            "action=model_load model_id=%s backend=%s", model_id, backend,
        )

    def log_model_unload(self, model_id: str) -> None:
        """Log a model unload event."""
        self._logger.info("action=model_unload model_id=%s", model_id)

    def log_rate_limit(self, limiter_name: str) -> None:
        """Log a rate limit hit."""
        self._logger.warning(
            "action=rate_limit_hit limiter=%s", limiter_name,
        )

    def log_warning(self, message: str) -> None:
        """Log a general security warning."""
        self._logger.warning("action=security_warning message=%s", message)


audit = AuditLogger()


# ============================================================================
# 4. Resource Limits
# ============================================================================

class ResourceLimits:
    """Enforce resource limits on code execution."""

    MAX_OUTPUT_SIZE: int = 5 * 1024 * 1024   # 5MB max output
    MAX_FILE_SIZE: int = 100 * 1024 * 1024    # 100MB max file write

    @staticmethod
    def get_memory_usage_mb() -> float:
        """Get current process memory usage in MB."""
        try:
            import resource as _resource
            # ru_maxrss is in KB on Linux, bytes on macOS
            usage = _resource.getrusage(_resource.RUSAGE_SELF)
            import sys
            if sys.platform == "darwin":
                return usage.ru_maxrss / (1024 * 1024)
            else:
                return usage.ru_maxrss / 1024
        except (ImportError, AttributeError):
            # Fallback: read from /proc on Linux
            try:
                with open("/proc/self/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) / 1024  # KB to MB
            except (FileNotFoundError, IOError):
                pass
        return 0.0

    @staticmethod
    def check_disk_space(path: str = "/", min_gb: float = 1.0) -> tuple[bool, str]:
        """Check if sufficient disk space is available.

        Args:
            path: Filesystem path to check.
            min_gb: Minimum free space required in GB.

        Returns:
            (has_enough_space, message)
        """
        try:
            stat = os.statvfs(path)
            free_bytes = stat.f_bavail * stat.f_frsize
            free_gb = free_bytes / (1024 ** 3)
            if free_gb < min_gb:
                return False, (
                    f"Low disk space: {free_gb:.2f} GB available at '{path}', "
                    f"minimum required: {min_gb:.1f} GB."
                )
            return True, f"Disk space OK: {free_gb:.2f} GB available at '{path}'."
        except (OSError, AttributeError) as e:
            return False, f"Could not check disk space at '{path}': {e}"


resource_limits = ResourceLimits()


# ============================================================================
# 5. Output Sanitization
# ============================================================================

# Patterns that might indicate secrets in output
SECRET_PATTERNS = [
    re.compile(
        r'(?i)(api[_-]?key|secret[_-]?key|password|token|bearer)\s*[=:]\s*["\']?([^\s"\']{8,})',
        re.IGNORECASE,
    ),
    re.compile(r'(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----'),
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),      # GitHub personal access token
    re.compile(r'sk-[a-zA-Z0-9]{32,}'),        # OpenAI API key pattern
    re.compile(r'hf_[a-zA-Z0-9]{34}'),         # HuggingFace token
]


def scan_output_for_secrets(output: str) -> list[str]:
    """Scan output for potential secret leaks.

    Returns a list of warning messages describing each detected pattern.
    """
    warnings: list[str] = []
    for pattern in SECRET_PATTERNS:
        matches = pattern.findall(output)
        if matches:
            warnings.append(
                f"Potential secret detected matching pattern: {pattern.pattern}"
            )
    return warnings


def redact_secrets(output: str) -> str:
    """Redact potential secrets from output. Returns cleaned output."""
    redacted = output
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
