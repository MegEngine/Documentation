# Minimal makefile for Sphinx documentation
# You can set these variables from the command line, and also
# from the environment for the first two.
MEGENGINEPY   = `python3 -c "import os; \
                import megengine; \
                print(os.path.dirname(megengine.__file__))"`
LANGUAGE      ?= zh_CN
SPHINXOPTS    ?= -j auto -D language='$(LANGUAGE)' $(AUTOBUILDOPTS)
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
HTMLAPI       ?= reference/api
AUTOBUILDOPTS ?= 

# Put it first so that "make" without argument is like "make help".
help:
	@echo "============================================== Target ======================================================"
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "============================================= Variables ===================================================="
	@echo "Default variables in makefile:"
	@echo "MEGENGINEPY: ${MEGENGINEPY}"
	@echo "SOURCEDIR: ${SOURCEDIR}"
	@echo "BUILDDIR: ${BUILDDIR}"
	@echo "HTMLAPI: ${HTMLAPI}"
	@echo "SPHINXOPTS: ${SPHINXOPTS}"
	@echo "AUTOBUILDOPTS: ${AUTOBUILDOPTS}"
	@echo "=============================================== Notes ======================================================"
	@echo "1. You can use\033[36m export PYTHONPATH=\"/path/to/megengine\"\033[0m to specify megengine python package path."
	@echo "2. You can use\033[36m make LANGUAGE=\"[ zh_CN | en ]\" html\033[0m to specify the language displayed in documentation."
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

