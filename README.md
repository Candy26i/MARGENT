# MARGENT project page

Static project page and milestone report for *MARGENT: Measuring the Marginal
Value of Delegation in Agentic Systems*.

- `index.html` — English project page
- `report.html` — 阶段实验报告 (Chinese milestone report)
- `static/images/` — figures; regenerate with the scripts in `static/charts/`
  (`/path/to/python3 static/charts/make_paper_figures.py`, `make_main_figures.py`,
  `make_medqa_figures.py`; needs matplotlib)
- `static/data/paper_tables.json` — every table of the paper, transcribed
- `static/data/main_results.json` — Appendix B predecessor diagnostic

Preview locally:

```bash
python3 -m http.server 8765
```

Published from the `gh-pages` branch of https://github.com/Candy26i/MARGENT
(Settings → Pages → branch `gh-pages`, folder `/`). No build step is needed; the page
uses only Google Fonts as an external resource.
