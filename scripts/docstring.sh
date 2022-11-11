#!/usr/bin/env bash
set -e
set -x

DOCSTRING_SRC=${1:-"src"}

interrogate $DOCSTRING_SRC -c pyproject.toml --color
pydocstyle $DOCSTRING_SRC -e --count --convention=google --add-ignore=D403
darglint -v 2 $DOCSTRING_SRC
