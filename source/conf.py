# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from datetime import datetime
import re
import importlib
import inspect
import logging
import os
import sys
import sphinx
import megengine

# -- Project information -----------------------------------------------------

project = 'MegEngine'
copyright = f'2020-{datetime.now().year}, The MegEngine Open Source Team'
author = 'The MegEngine Open Source Team'
version = megengine.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'nbsphinx',
    'recommonmark',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.graphviz',
    'sphinxcontrib.mermaid',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

source_encoding = "utf-8"

master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'build',
    'examples',
    '**/includes/**',
    '**.ipynb_checkpoints'
]

# -- Options for internationalization ----------------------------------------

language = 'zh_CN'

# By default, the document `functional/loss.rst` ends up in the `functional` text domain. 
# With this option set to False, it is `functional/loss`.
gettext_compact = False

# -- Options for Extensions -------------------------------------------------

# Setting for sphinx.ext.autosummary to auto-generate single html pages 
# Please makesure all api pages are stored in `/refenrece/api/` directory
autosummary_generate = True

# Setting for sphinx.ext.auotdoc 
autodoc_default_options = {
    'member-order': 'bysource', # Need developer organize the source code
    'show-inheritance': True,  # But it can not refer the short module path
}
autoclass_content = 'class'
autodoc_typehints = 'description'
autodoc_docstring_signature = True
add_function_parentheses = False
add_module_names = False

# Setting for sphinx.ext.mathjax
# The path to the JavaScript file to include in the HTML files in order to load MathJax.
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

mathjax_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
}

# Setting for sphinxcontrib-mermaid
mermaid_version = 'latest' # from CDN unpkg.com 

# Setting for sphinx.ext.intersphinx
# Useful for refenrece other projects, eg. :py:class:`zipfile.ZipFile`
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None)
}

# Setting for sphinx.ext.extlinks
# Can use the alias name as a new role, e.g. :issue:`123`
extlinks = {
    'issue': ('https://github.com/MegEngine/MegEngine/issues/%s', 'Issue #'),
    'pull': ('https://github.com/MegEngine/MegEngine/pull/%s', 'Pull Requset #')
}

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

html_theme = 'pydata_sphinx_theme'
html_theme_path = ['_themes']
html_theme_options = {
    'search_bar_text': '输入搜索文本...',
    'search_bar_position': 'navbar',
    'github_url': 'https://github.com/MegEngine/MegEngine',
    'external_links': [
        { 'name': '论坛', 'url': 'https://discuss.megengine.org.cn/'}, 
        { 'name': '官网', 'url': 'https://megengine.org.cn/'}
    ],
    'use_edit_page_button': False,
    'navigation_with_keys': False,
    'show_prev_next': False,
    'use_version_switch': True,
    'version_switch_json_url': '/versions.json',
    'version_switch_enable_locale': True,
    'version_switch_locates': ['zh', 'en'],
}

html_sidebars = {
    '**': ['sidebar-search-bs.html', 'sidebar-nav-bs.html'],
    'index': ['sidebar-search-bs.html', 'homepage-sidebar.html']
}

html_static_path = ['_static']
html_logo = "logo.png"
html_favicon = "favicon.ico"
html_css_files = [
    'css/custom.css'
]
html_js_files = [
    'js/custom.js'
]

html_search_language = 'zh'
