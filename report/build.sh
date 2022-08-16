#!/bin/env sh

latexmk             \
    -shell-escape   \
    -outdir=build/  \
    -pdf            \
    report.tex
