# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'MegEngine'
copyright = '2021, Megvii Inc'
author = 'Megvii Inc'
language = 'zh_CN'

# -- General configuration ---------------------------------------------------

extensions = [
    'nbsphinx',
    'recommonmark',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
#    'sphinxcontrib.bibtex',
    'sphinx_copybutton'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']

exclude_patterns = ['_drafts', 'examples']


# -- Options for HTML output -------------------------------------------------

html_theme = 'classic'
html_theme_options = {
    "rightsidebar": "true"
}

html_static_path = ['_static']
html_sourcelink_suffix = ''

html_sidebars = {
    '**': ['searchbox.html', 'localtoc.html', 'relations.html']
}

html_search_language = 'zh'
html_search_options = {
    'dict': 'jieba'
}
