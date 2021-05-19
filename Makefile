# Minimal makefile for Sphinx documentation
# You can set these variables from the command line, and also
# from the environment for the first two.
MEGENGINEPY   = `python3 -c "import os; \
                import megengine; \
                print(os.path.dirname(megengine.__file__))"`
LANGUAGE      ?= zh_CN
SPHINXOPTS    ?= -j auto -D language='$(LANGUAGE)'
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
HTTPPORT      ?= 1124
HTMLAPI       ?= reference/api

# Put it first so that "make" without argument is like "make help".
help: info
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "You can use\033[31m export PYTHONPATH=\"/path/to/megengine\"\033[0m to specify megengine python package path."
	@echo "You can use\033[31m make clean\033[0m to clean all build and generated api files."

info:
	@echo "Current MEGENGINEPY path: ${MEGENGINEPY}"
	@echo "Current SOURCEDIR relative path: ${SOURCEDIR}"
	@echo "Current BUILDDIR  relative path: ${BUILDDIR}"
	@echo "Current HTMLAPI   relative path: ${HTMLAPI}"
	@echo "Current SPHINXOPTS: ${SPHINXOPTS}"

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(SOURCEDIR)/$(HTMLAPI)

http: html
	cd $(BUILDDIR)/html
	python3 -m http.server $(HTTPPORT)

.PHONY: info help clean http Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

