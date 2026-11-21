# DSP Lab 3 – FIR, IIR and adaptive LMS filtering for noise reduction

Digital Signal Processing (prof. dr hab. Vasyl Martsenyuk), UBB — Radosław Skrzypczyński, Informatyka II st., Gr. 2.

**Task:** implement an FIR filter, an IIR filter and an adaptive LMS filter with the parameters of the variant,
apply them to a noisy sinusoidal signal and observe the noise reduction. All 16 variants of the instructions are
computed on the same test signal (5 Hz sine, fₛ = 1000 Hz, white noise σ = 0.5). Filters are implemented from the
difference equations, validated against `scipy.signal.lfilter` and the Wiener solution, and compared by the
gain-independent SNR improvement ΔSNR = 10·log₁₀(|H(f₀)|²/Σh²) and by the SNR of the output relative to the clean
signal.

| File | Content |
|---|---|
| `Lab3_Filtering.ipynb` | executed notebook (theory, implementation, one figure and table per variant, comparison, interpretation) |
| `Lab3_Filtering.pdf` | PDF export of the notebook |
| `Report_Lab3.pdf` / `.docx` | report based on the course template |
| `src/Lab3_Filtering.py` | the notebook as a plain Python script (jupytext percent format) |
| `figures/`, `results/` | all figures and the result tables of the notebook |
| `scripts/` | PDF export of the notebook and report generator |

Reproduce (Python 3.14, numpy, scipy, matplotlib, pandas, jupytext, nbconvert):

```
py -3.14 -m jupytext --to ipynb --output Lab3_Filtering.ipynb src/Lab3_Filtering.py
py -3.14 -m jupyter nbconvert --to notebook --execute --inplace Lab3_Filtering.ipynb
py -3.14 -X utf8 scripts/export_notebook_pdf.py Lab3_Filtering.ipynb Lab3_Filtering.pdf
py -3.14 -X utf8 scripts/make_report.py path/to/DSP_TemplateReport.docx
```
