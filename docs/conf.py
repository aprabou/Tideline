"""Sphinx configuration for Tideline documentation."""

project = "Tideline"
author = "DS3 Hacks Team"
release = "0.1.0"
copyright = "2026, DS3 Hacks Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.11", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
autodoc_typehints = "description"
napoleon_google_docstring = True
