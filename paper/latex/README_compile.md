# How to compile the paper (paper/latex)

Tested on this lab machine on 2026-11-24 (TeX Live 2026, pdfTeX 1.40.29, latexmk 4.89, installed in ~/.local/bin),
each route from a fresh unzip of paper/latex.zip. Both local routes worked: exit code 0, main.pdf with 6 pages,
no warnings left in the final main.log.

## A. Overleaf (not tested from here: needs your Overleaf account)
1. Overleaf -> New Project -> Upload Project -> choose `paper/latex.zip`.
2. Menu -> Settings: Main document = `main.tex` (in the `latex/` folder of the zip), Compiler = pdfLaTeX.
3. Click Recompile. The PDF appears in the right pane (Download PDF icon above it).

## B. Local, one command (TESTED, works)
    cd paper/latex && latexmk -pdf main.tex
main.pdf appears in `paper/latex/main.pdf`. Clean helper files with `latexmk -c` (keeps the PDF).

## C. Local fallback without latexmk (TESTED, works)
    cd paper/latex
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex
main.pdf appears in `paper/latex/main.pdf`.

## Normal warnings (ignore)
- After the FIRST pdflatex run only: `Citation ... undefined`, `Reference ... undefined`,
  `There were undefined references`, `Label(s) may have changed. Rerun`: they disappear after bibtex + 2 more runs
  (latexmk does this by itself).
- `Underfull \hbox` / `Underfull \vbox`: spacing only.
- Red `[TBD: ...]` boxes and blue `pilot` marks in the PDF are intentional placeholders (macros \tbd, \pilot in main.tex).

## Real errors (must fix)
- Any line starting with `!` in the log (e.g. `! Undefined control sequence`, `! Missing $ inserted`,
  `! LaTeX Error: File ... not found`); pdflatex then stops or gives no PDF. Usual causes: a missing package,
  `%`, `&`, `_` or `#` not escaped in text, a figure file missing from `figures/`.
- `Citation ... undefined` that is STILL there after the full sequence: the key is missing in references.bib.
- `Overfull \hbox (XXpt too wide)` with XX > ~5pt: text or a table sticks into the margin (ACL checks this);
  shrink the table (e.g. \resizebox) or rephrase. Currently none.
- bibtex `error message` in main.blg: a broken .bib entry.

## Page limit
ACL long paper: at most 8 pages of content (formatting.md of the template); references and the mandatory
Limitations section do not count (ACL call for papers, verify for the ARR cycle). Current draft: main text ends on
page 5 (~4.5 pages), then Limitations and References; 6 pages in total.
