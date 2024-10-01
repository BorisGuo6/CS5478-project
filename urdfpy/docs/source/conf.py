# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_rtd_theme
from sphinx.domains.python import PythonDomain
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = u'urdfpy'
copyright = u'2019, Matthew Matl'
author = u'Matthew Matl'

# The short X.Y version
version = u''
# The full version, including alpha/beta/rc tags
release = u''


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver'
]
numpydoc_class_members_toctree = False
automodapi_toctreedirnm = 'generated'
automodsumm_inherited_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'urdfpydoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'urdfpy.tex', u'urdfpy Documentation',
     u'Matthew Matl', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'urdfpy', u'urdfpy Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'urdfpy', u'urdfpy Documentation',
     author, 'urdfpy', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'trimesh': ('https://trimsh.org/', None),
}

# Autosummary fix
autosummary_generate = True

# Try to suppress multiple-definition warnings by always taking the shorter
# path when two or more paths have the same base module


class MyPythonDomain(PythonDomain):

    def find_obj(self, env, modname, classname, name, type, searchmode=0):
        """Ensures an object always resolves to the desired module
        if defined there."""
        orig_matches = PythonDomain.find_obj(
            self, env, modname, classname, name, type, searchmode
        )

        if len(orig_matches) <= 1:
            return orig_matches

        # If multiple matches, try to take the shortest if all the modules are
        # the same
        first_match_name_sp = orig_matches[0][0].split('.')
        base_name = first_match_name_sp[0]
        min_len = len(first_match_name_sp)
        best_match = orig_matches[0]

        for match in orig_matches[1:]:
            match_name = match[0]
            match_name_sp = match_name.split('.')
            match_base = match_name_sp[0]

            # If we have mismatched bases, return them all to trigger warnings
            if match_base != base_name:
                return orig_matches

            # Otherwise, check and see if it's shorter
            if len(match_name_sp) < min_len:
                min_len = len(match_name_sp)
                best_match = match

        return (best_match,)


def setup(sphinx):
    """Use MyPythonDomain in place of PythonDomain"""
    sphinx.override_domain(MyPythonDomain)
