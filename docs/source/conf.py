# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import logging
import os
import sys
import tomllib
from datetime import datetime

from asgardpy.version import __base_version__, __version__  # noqa: E402

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

work_dir_path = os.path.join(os.path.dirname(__file__), "../../")

with open(os.path.join(work_dir_path, "pyproject.toml"), "rb") as f:
    project_info = tomllib.load(f)

project = project_info["project"]["name"]

# Read the list of author names
author = ""
for auth in project_info["project"]["authors"]:
    author += auth["name"] + ", "
author = author[:-2]

copyright = f"{datetime.today().year}, {author}"

version = __version__
# The full version contains alpha, beta, rc tags
release = __base_version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.towncrier.ext",
]

# Tell myst-parser to assign header anchors for h1-h3.
myst_heading_anchors = 3

suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "changes"]

source_suffix = [".rst", ".md"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "gammapy": ("https://docs.gammapy.org/1.1/", None),
    "astropy": ("https://docs.astropy.org/en/latest/", None),
    # Uncomment these if you use them in your codebase:
    #  "torch": ("https://pytorch.org/docs/stable", None),
    #  "datasets": ("https://huggingface.co/docs/datasets/master/en", None),
    #  "transformers": ("https://huggingface.co/docs/transformers/master/en", None),
}

# By default, sort documented members by type within classes and modules.
autodoc_member_order = "groupwise"

autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_field_summary = True

# Options: draft/sphinx-version/sphinx-release
towncrier_draft_autoversion_mode = 'draft'
towncrier_draft_include_empty = True
towncrier_draft_working_directory = work_dir_path
# Not yet supported:
# towncrier_draft_config_path = 'pyproject.toml'  # relative to cwd

# Include default values when documenting parameter types.
typehints_defaults = "comma"


# Define the json_url for our version switcher.
# json_url = "https://asgardpy.readthedocs.io/en/latest/_static/switcher.json"

# Define the version we use for matching in the version switcher.,
# version_match = os.getenv("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# if not version_match or version_match.isdigit():
# For local development, infer the version to match from the package.
# if "dev" in release or "rc" in release:
#     version_match = "latest"
# else:
#     version_match = release

# We want to keep the relative reference when on a pull request or locally
# json_url = "_static/switcher.json"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_title = f"{project} v{__version__}"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["css/custom.css"]

html_favicon = "_static/favicon.ico"

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/chaimain/asgardpy",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
    # "navbar_start": ["navbar-logo", "version-switcher"],
    # "switcher": {
    #     "version_match": version_match,
    #     "json_url": json_url,
    # },
}

# -- Hack to get rid of stupid warnings from sphinx_autodoc_typehints --------


class ShutupSphinxAutodocTypehintsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if "Cannot resolve forward reference" in record.msg:
            return False
        return True


logging.getLogger("sphinx.sphinx_autodoc_typehints").addFilter(ShutupSphinxAutodocTypehintsFilter())
