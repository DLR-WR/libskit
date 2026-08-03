# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile apidocs doxygen clean-all

# Build Doxygen documentation first, then Sphinx
apidocs: doxygen html
	@echo "API documentation built successfully!"
	@echo "Open $(BUILDDIR)/html/index.html to view."

# Generate Doxygen XML for Breathe integration
doxygen:
	@echo "Generating Doxygen documentation..."
	cd $(SOURCEDIR) && doxygen Doxyfile

# Clean all build artifacts including Doxygen output
clean-all: clean
	rm -rf $(BUILDDIR)/doxygen

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
