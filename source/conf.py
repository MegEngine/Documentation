"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Monkey patch for `mprop` package ----------------------------------------
# It will make some module to be a instance for getting or setting property
# That is good for improving user experience but Sphinx can not handle with it
# So we add a monkey patch then mprop will do nothing while building the doc.

def doNothing(*args):
    return

import mprop
mprop.init = doNothing
mprop.mproperty = property
mprop.auto_init = doNothing

# -- Package setup -----------------------------------------------------------

from datetime import datetime
import logging

# -- Path setup --------------------------------------------------------------

import megengine

# -- Project information -----------------------------------------------------

project = "MegEngine"
copyright = f"2020-{datetime.now().year}, The MegEngine Open Source Team"
author = "The MegEngine Open Source Team"
version = megengine.__version__
release = version

# -- General configuration ---------------------------------------------------
add_function_parentheses = False
add_module_names = False

# WARNING: Do not modify loading order unless you know the dependencies
extensions = [
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_panels",
    "sphinx_tabs.tabs",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

source_encoding = "utf-8"

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "build",
    "examples",
    "**/includes/**",
    "**.ipynb_checkpoints",
]

# -- Options for internationalization ----------------------------------------
language = "zh_CN"
locale_dirs = ["../locales/"]

# By default, the document `functional/loss.rst` ends up in the `functional` text domain.
# With this option set to False, it is `functional/loss`.
gettext_compact = False

# -- Options for Extensions -------------------------------------------------

# Setting for sphinx.ext.autosummary to auto-generate single html pages
# Please makesure all api pages are stored in `/refenrece/api/` directory
autosummary_generate = True

# Setting for sphinx.ext.auotdoc
autodoc_default_options = {"member-order": "bysource"}
autoclass_content = "class"
autodoc_docstring_signature = True
autodoc_preserve_defaults = True
autodoc_mock_imports = ["mprop"]

# Setting for sphinx.ext.mathjax
# The path to the JavaScript file to include in the HTML files in order to load MathJax.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# Setting for sphinxcontrib-mermaid
mermaid_version = "latest"  # from CDN unpkg.com

# Setting for sphinx.ext.intersphinx
# Useful for refenrece other projects, eg. :py:class:`zipfile.ZipFile`
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
}

# Setting for sphinx.ext.extlinks
# Can use the alias name as a new role, e.g. :issue:`123`
extlinks = {
    "src": ("https://github.com/MegEngine/MegEngine/blob/master/%s", ""),
    "docs": ("https://github.com/MegEngine/Documentation/blob/master/%s", ""),
    "issue": ("https://github.com/MegEngine/MegEngine/issues/%s", "Issue #"),
    "pull": ("https://github.com/MegEngine/MegEngine/pull/%s", "Pull Requset #"),
    "duref": (
        "http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#%s",
        "",
    ),
}

# Setting for sphinx_autodoc_typehints
typehints_fully_qualified = False

# Setting for sphinx_copybutton
copybutton_selector = "div:not(.no-copy)>div.highlight pre"
copybutton_prompt_text = (
    r">>> |\.\.\. |(?:\(.*\) )?\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True

# Setting for sphinx_panels
panels_add_bootstrap_css = False

# Setting for sphinx.ext.nbsphinx
# nbsphinx do not use requirejs (breaks bootstrap)
nbsphinx_requirejs_path = ""
logger = logging.getLogger(__name__)

try:
    import nbconvert
except ImportError:
    logger.warning("nbconvert not installed. Skipping notebooks.")
    exclude_patterns.append("**/*.ipynb")
else:
    try:
        nbconvert.utils.pandoc.get_pandoc_version()
    except nbconvert.utils.pandoc.PandocMissing:
        logger.warning("Pandoc not installed. Skipping notebooks.")
        exclude_patterns.append("**/*.ipynb")

# -- Options for HTML output -------------------------------------------------

panels_add_bootstrap_css = False  # pydata-sphinx-theme already loads this

html_logo = "logo.png"
html_favicon = "favicon.ico"
html_theme = "pydata_sphinx_theme"
html_theme_path = ["_themes"]
html_static_path = ["_static"]
html_extra_path = ["google940c72af103ac75f.html"]
html_css_files = ["css/custom.css"]
html_additional_pages = {
    'index': 'indexcontent.html'
}

html_search_language = "zh"

# Configuration for pydata-sphinx-theme, the doc URL:
# https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html
#
# WARNING: MegEngine Doc used a forked version here:
# https://github.com/MegEngine/pydata-sphinx-theme/tree/dev

html_theme_options = {
    "search_bar_text": "输入搜索文本...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MegEngine/MegEngine",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Bilibili",
            "url": "https://space.bilibili.com/649674679",
            "icon": "fas fa-play-circle",
        },
    ],
    "external_links": [
        {"name": "论坛", "url": "https://discuss.megengine.org.cn/"},
        {"name": "官网", "url": "https://megengine.org.cn/"},
    ],
    # Note: If you only want to show current version information
    # Please replace "version-switcher.html" with "current-version.html"
    # These two templates only work in MegEngine forked dev branch
    "navbar_end": ["navbar-icon-links.html", "version-switcher.html"],
    "collapse_navigation": True,
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "show_prev_next": False,
    # The following settings just work in MegEngine forked dev branch
    "use_version_switch": True,
    "version_switch_json_url": "/doc/version.json",
    "version_switch_enable_locale": True,
    "version_switch_locales": ["zh", "en"],
}

# Setting for Edit this Page button
# Should reset for self-hosted GitHub/GitLab instance.
html_context = {
    "github_url": "https://github.com",
    "github_user": "MegEngine",
    "github_repo": "Documentation",
    "github_version": "main",
    "doc_path": "source",
}
