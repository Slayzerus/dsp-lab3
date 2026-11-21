"""Build Report_Lab3.docx / Report_Lab3.pdf from the course template.

Usage:  py -3.14 -X utf8 scripts/make_report.py [path/to/DSP_TemplateReport.docx]
Run after the notebook has been executed: the numbers come from results/*.csv, the figures from figures/.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from report_builder import Report  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parents[1] / "Lab3" / "DSP_TemplateReport.docx"
REPO = "https://github.com/Slayzerus/dsp-lab3"

SUM = pd.read_csv(ROOT / "results" / "summary.csv", index_col=0, encoding="utf-8")
INP = pd.read_csv(ROOT / "results" / "input.csv", index_col=0, encoding="utf-8").iloc[:, 0]
SNR_IN = INP["SNR_IN"]

VARIANTS = {
    1: ([0, 0, 1], [1, -1.5, 0.7], [1, 0.5, 0.1], 0.01, 4), 2: ([1, 1, 2], [0.5, 0.2], [1, -0.8], 0.05, 4),
    3: ([0, 0, 3], [0.6, 0.3, 0.1], [1, 0.5, 0.2], 0.01, 6), 4: ([0.2, 0.3, 0.2], [1, 0.5, 0.3], [1, -0.6], 0.1, 4),
    5: ([0.5, 0.3, 0.1], [0.3, 0.4, 0.3], [1, -0.4, 0.2], 0.05, 4),
    6: ([0.25, 0.25, 0.25, 0.25], [1, 1, 1], [1, -0.5], 0.1, 6), 7: ([1, -1, 0.5], [0.3, 0.4], [1, -0.5, 0.2], 0.05, 4),
    8: ([1, 0, 1], [0.5, 0.5], [1, -0.3], 0.05, 5), 9: ([0, 1, 0], [0.2, 0.8], [1, 0.5], 0.05, 3),
    10: ([0.4, 0.4, 0.4], [1, -0.6], [1, 0.5], 0.1, 5), 11: ([1, -0.2, 0.5], [0.6, 0.3], [1, 0.4], 0.05, 6),
    12: ([1, 0, -1], [0.4, 0.6], [1, -0.3], 0.02, 5), 13: ([1, 0.5, 0.5], [1, -0.5, 0.2], [1, -0.3], 0.1, 4),
    14: ([0.5, 0.5], [0.3, 0.4], [1, -0.4], 0.1, 3), 15: ([0.2, 0.3, 0.5], [1, 0.5], [1, -0.7], 0.05, 6),
    16: ([0.5, 0.5], [0.4, 0.3], [1, -0.5], 0.05, 5),
}


def fmt(c):
    return "{" + ", ".join(f"{v:g}" for v in c) + "}"


def d1(v):
    return f"{v:+.1f}" if abs(v) >= 0.05 else "0.0"


rep = Report(TEMPLATE, lab=3, date=date.today().strftime("%d.%m.%Y"),
             topic="Digital filter design and analysis (FIR and IIR filters) and adaptive LMS filtering for noise "
                   "reduction",
             variant="all variants 1–16")

# 1 ---------------------------------------------------------------------------------------------------------------
rep.heading("1. Problem statement")
rep.para("Each task requires implementing all three types of filters – **FIR**, **IIR** and the adaptive **LMS** "
         "filter – in Python with the parameters of the variant, applying them to a noisy sinusoidal signal and "
         "observing their performance for noise reduction. **All 16 variants were computed** on the same test signal, "
         "so that the results can be compared.")
rep.para("FIR: y[n] = Σ bₖ·x[n−k]. IIR: y[n] = Σ bₖ·x[n−k] − Σₖ₌₁ aₖ·y[n−k] "
         "(stable if all poles lie inside the unit circle). LMS: y[n] = wᵀ[n]·x[n], e[n] = d[n] − y[n], "
         "w[n+1] = w[n] + μ·e[n]·x[n] with x[n] = (x[n], …, x[n−M+1])ᵀ, convergent for "
         "0 < μ < 2/λ_max. Noise reduction of a fixed filter for a sinusoid at f₀ in white noise: "
         "**ΔSNR = 10·log₁₀(|H(f₀)|²/Σh[n]²)** (independent of gain and delay). In addition the "
         "output is compared with the clean signal: **SNR re s = 10·log₁₀(P_s / mean((y − s)²))**.")

# 2 ---------------------------------------------------------------------------------------------------------------
rep.heading("2. Input data")
rep.para(f"Test signal (as in the examples of the instructions): s[n] = sin(2π·5 Hz·n/fₛ), fₛ = 1000 Hz, "
         f"N = 1000 (1 s), plus white Gaussian noise with σ = 0.5 (fixed seed): x = s + v. The LMS filter uses "
         f"d[n] = s[n] as the desired signal, as in the instructions. All metrics are evaluated in the second half of "
         f"the record (after transients and LMS convergence), where the input has MSE(x, s) = {INP['MSE_IN']:.3f} and "
         f"**SNR = {SNR_IN:.2f} dB**.")
rows = [(no, fmt(f), fmt(b), fmt(a), f"{mu:g}", M) for no, (f, b, a, mu, M) in VARIANTS.items()]
rep.table(["No", "FIR b", "IIR b", "IIR a", "LMS μ", "LMS M"], rows,
          caption="Parameters of the 16 variants (Sec. 7 of the instructions)", size=9,
          col_widths=[1.0, 4.2, 3.4, 3.2, 1.6, 1.6])
rep.para("Environment: Python 3.14.2, NumPy 2.4.1, SciPy 1.18.1, Matplotlib 3.10.8, pandas 3.0.6, Jupyter.", size=11)

# 3 ---------------------------------------------------------------------------------------------------------------
rep.heading("3. Commands used")
rep.subheading("a) Source code (core of the notebook Lab3_Filtering.ipynb)")
rep.code('''
def fir_filter(x, b):              # y[n] = sum_k b[k] x[n-k], zero initial state
    M = len(b)
    xp = np.concatenate([np.zeros(M - 1), x])
    y = np.zeros(len(x))
    for n in range(len(x)):
        y[n] = np.dot(b, xp[n:n + M][::-1])
    return y

def iir_filter(x, b, a):           # y[n] = sum b[k] x[n-k] - sum_{k>=1} a[k] y[n-k]
    b, a = np.asarray(b)/a[0], np.asarray(a)/a[0]
    M, N = len(b), len(a)
    xp = np.concatenate([np.zeros(M - 1), x])
    yp = np.zeros(N - 1 + len(x))  # N-1 zeros before y[0]
    for n in range(len(x)):
        acc = np.dot(b, xp[n:n + M][::-1])
        if N > 1:
            acc -= np.dot(a[1:], yp[n:n + N - 1][::-1])
        yp[n + N - 1] = acc
    return yp[N - 1:]

def lms_filter(x, d, mu, M):       # w <- w + mu e[n] x[n], x[n] = (x[n], ..., x[n-M+1])
    xp = np.concatenate([np.zeros(M - 1), x])
    w, y, e = np.zeros(M), np.zeros(len(x)), np.zeros(len(x))
    for n in range(len(x)):
        u = xp[n:n + M][::-1]
        y[n] = w @ u
        e[n] = d[n] - y[n]
        w = w + mu * e[n] * u
    return y, e, w

def delta_snr(b, a):               # SNR gain for a sinusoid at F0 in white noise
    _, H = freqz(b, a, worN=[F0], fs=FS)
    h = lfilter(b, a, np.r_[1.0, np.zeros(4095)])      # impulse response
    return 10*np.log10(abs(H[0])**2 / np.sum(h**2))
''')
rep.para("The FIR and IIR implementations agree with `scipy.signal.lfilter`; the LMS filter reaches the Wiener "
         "solution w_opt = R⁻¹p computed from the same data. Three defects of the example code of the "
         "instructions were corrected:")
rep.bullets([
    "FIR: the loop started at n = M, so the first M output samples remained zero;",
    "IIR: for the first samples the slices of x and y are shorter than b and a[1:], so `np.dot` raises a "
    "ValueError for every filter with more than one coefficient;",
    "LMS: the input vector was x[n−1] … x[n−M] (without the current sample), contrary to Sec. 4.1–4.2 "
    "of the instructions.",
], size=11)
rep.subheading("b) Screenshots")
rep.figure(FIG / "fig01_test_signal.png", "Test signal: clean 5 Hz sinusoid and the noisy input", 14.0)
rep.figure(FIG / "fig07_variant_06.png",
           "Variant 6: FIR moving average (best FIR, ΔSNR = 6.0 dB), IIR with DC gain 6 (noise reduced, but the "
           "output is six times too large), LMS M = 6 (best LMS); bottom: magnitude responses, IIR poles/zeros, LMS "
           "learning curve", 16.0)
rep.figure(FIG / "fig13_variant_12.png",
           "Variant 12: the FIR filter {1, 0, −1} is a band-pass with zeros at DC and fₛ/2 and removes the 5 Hz "
           "signal (−27 dB); the IIR low-pass and the LMS filter reduce the noise", 16.0)
rep.link("Link to the remote repository (GitHub): ", REPO)
rep.para("The repository contains the notebook (Lab3_Filtering.ipynb) with one figure and one table per variant, its "
         "PDF export (Lab3_Filtering.pdf), the source in plain Python (src/), all figures (figures/), the result "
         "tables (results/) and this report.", size=11)

# 4 ---------------------------------------------------------------------------------------------------------------
rep.heading("4. Outcomes")
rep.para("Results for all 16 variants (ΔSNR: gain-independent improvement; SNR re s: direct comparison with the "
         f"clean signal, input {SNR_IN:.2f} dB; values in dB):")
rows = []
for no, r in SUM.iterrows():
    rows.append([no, r["FIR type"].replace(" (gain/delay)", ""), d1(r["FIR ΔSNR [dB]"]), f"{r['FIR SNR re s [dB]']:.1f}",
                 r["IIR type"], f"{r['IIR max|p|']:.2f}", d1(r["IIR ΔSNR [dB]"]), f"{r['IIR SNR re s [dB]']:.1f}",
                 f"{r['IIR SNR re s, gain corrected']:.1f}", r["LMS μ / M"], f"{r['LMS SNR re s [dB]']:.1f}"])
rep.table(["No", "FIR type", "FIR ΔSNR", "FIR SNR re s", "IIR type", "IIR max|p|", "IIR ΔSNR",
           "IIR SNR re s", "IIR SNR re s, gain corr.", "LMS μ / M", "LMS SNR re s"], rows,
          caption="Summary of all variants", size=8,
          col_widths=[0.8, 1.7, 1.2, 1.2, 1.7, 1.2, 1.2, 1.2, 1.5, 1.6, 1.2])
rep.figure(FIG / "fig18_comparison.png",
           "Comparison of all variants: (a) ΔSNR of the fixed filters, (b) SNR of the outputs relative to the clean "
           "signal (dashed: input)", 16.0)
rep.figure(FIG / "fig19_lms_weights.png",
           "LMS weight trajectories for μ = 0.01 (slow, smooth convergence) and μ = 0.1 (fast convergence, strong "
           "fluctuation); dashed: Wiener weights", 16.0)
rep.bullets([
    "**FIR:** the best filter is the 4-tap moving average of variant 6 (+6.0 dB = 10·log₁₀ 4, the bound for "
    "4 taps); the other low-pass sets give +3.0 … +4.8 dB. Pure delays (variants 1, 3, 9) give 0 dB; the "
    "high-pass/band-pass sets of variants 7 and 12 lose 9.6 and 27 dB.",
    "**IIR:** all filters are stable (max |p| ≤ 0.8). The best is variant 2 (pole 0.8, +9.8 dB), then 15 (pole "
    "0.7, +8.1 dB) and 4 (pole 0.6, +7.4 dB). These filters have a DC gain of 3.5–5, so their raw output is far "
    "from the clean signal. After gain correction they reach 9.8–10.7 dB. Variants 1, 9 and 10 are high-pass "
    "filters (−2 … −27 dB).",
    f"**LMS:** best raw result in every variant, {SUM['LMS SNR re s [dB]'].min():.1f}–"
    f"{SUM['LMS SNR re s [dB]'].max():.1f} dB (+{SUM['LMS SNR re s [dB]'].min() - SNR_IN:.1f} … "
    f"+{SUM['LMS SNR re s [dB]'].max() - SNR_IN:.1f} dB), growing with M (M = 3: 8.3–8.5 dB, M = 6: "
    "10.6–12.0 dB). All step sizes are below 2/λ_max (0.61–1.12).",
], size=11)

# 5 ---------------------------------------------------------------------------------------------------------------
rep.heading("5. Conclusions")
rep.para("For the reasons given, we conclude that:", keep_next=True)
rep.bullets([
    "Only some of the given fixed filters reduce noise. Low-pass filters do: an FIR with positive coefficients "
    "(bounded by 10·log₁₀ M, reached by the moving average) or an IIR with a real pole 0 < p < 1 "
    "(ΔSNR ≈ 10·log₁₀((1+p)/(1−p))). Pure delays leave the SNR unchanged, and high-pass or band-pass "
    "coefficient sets make it worse.",
    "The IIR filter reaches a narrow low-pass with only 2–3 coefficients (+9.8 dB, which an FIR filter would need "
    "about 10 taps for), but its stability must be checked and its phase is non-linear. FIR filters are always "
    "stable and have linear phase for symmetric coefficients.",
    "The gain at the signal frequency is as important as the shape: without |H(f₀)| ≈ 1 the output is far from "
    "the clean signal even when the noise is reduced (e.g. IIR variant 2: −8.3 dB raw, 10.7 dB after correction).",
    "The adaptive LMS filter needs no design and gives the best raw result in every variant. Longer filters "
    "(larger M) reduce more noise. A larger μ converges faster, but the weights fluctuate more. Its advantage "
    "relies on the clean signal being available as d[n], as in the instructions; in practice an adaptive noise "
    "canceller or line enhancer structure is used. A correctly scaled fixed low-pass filter (IIR variant 2) is "
    "just as good.",
], numbered=True)

rep.save(ROOT / "Report_Lab3.docx", ROOT / "Report_Lab3.pdf")
