# Minimal makefile for Sphinx documentation
# You can set these variables from the command line, and also
# from the environment for the first two.
LANGUAGE      ?= zh_CN
AUTOBUILDOPTS ?= 

SPHINXOPTS    ?= -j auto -W --keep-going -D language='$(LANGUAGE)' $(AUTOBUILDOPTS)
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
HTMLAPI       ?= reference/api

# Put it first so that "make" without argument is like "make help".
help:
	@echo "============================================== Target ======================================================"
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo " \033[36m livehtml\033[0m    to make standalone HTML files and auto re-build while detects changes"
	@echo "============================================= Variables ===================================================="
	@echo "Default variables in makefile:"
	@echo "  SOURCEDIR:     ${SOURCEDIR}"
	@echo "  BUILDDIR:      ${BUILDDIR}"
	@echo "  HTMLAPI:       ${HTMLAPI}"
	@echo "  SPHINXOPTS:    ${SPHINXOPTS}"
	@echo "=============================================== Notes ======================================================"
	@echo "1. You can use\033[36m export PYTHONPATH=\"/path/to/megengine\"\033[0m to specify megengine python package path."
	@echo "2. You can use\033[36m export MGE_DOC_MODE=\"MINI\"\033[0m to skip generating API Pages, which speeds up a lot."
	@echo "For more details, please read the source code in\033[36m Makefile\033[0m."

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(SOURCEDIR)/$(HTMLAPI)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

