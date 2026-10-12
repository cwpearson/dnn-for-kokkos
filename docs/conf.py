# Sphinx configuration for DNN for Kokkos.
#
# Hand-written API reference using the C++ domain (``.. cpp:function::`` etc.).
# Template signatures are written out explicitly, which renders templated C++
# far better than the previous Doxygen setup.

project = "DNN for Kokkos"
author = "DNN for Kokkos contributors"
version = "0.1.0"
release = "0.1.0"

extensions = [
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST: allow the index page to include ../README.md directly.
myst_enable_extensions = []
myst_heading_anchors = 3

# README.md uses relative links (examples/...) that have no Sphinx target;
# don't let them fail -W builds.
suppress_warnings = ["myst.xref_missing"]

html_theme = "sphinx_rtd_theme"
