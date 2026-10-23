# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# Sphinx configuration for DNN for Kokkos.
#
# Hand-written API reference using the C++ domain (``.. cpp:function::`` etc.).
# Template signatures are written out explicitly, which renders templated C++
# far better than the previous Doxygen setup.

project = "DNN for Kokkos"
author = "DNN for Kokkos contributors"


def _project_version():
    # Single source of truth is the top-level project(VERSION ...) in
    # CMakeLists.txt; docs/conf.py just mirrors it.
    import pathlib
    import re

    cmake_lists = pathlib.Path(__file__).resolve().parent.parent / "CMakeLists.txt"
    match = re.search(
        r"project\s*\(\s*KokkosDNN\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
        cmake_lists.read_text(encoding="utf-8"),
    )
    if not match:
        raise RuntimeError("could not parse KokkosDNN VERSION from CMakeLists.txt")
    return match.group(1)


version = _project_version()
release = version

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
