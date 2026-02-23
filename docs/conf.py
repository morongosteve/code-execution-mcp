"""Sphinx configuration for API documentation."""

project = "code-execution-mcp"
copyright = "2026"
author = "code-execution-mcp contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_theme = "alabaster"
autodoc_member_order = "bysource"
