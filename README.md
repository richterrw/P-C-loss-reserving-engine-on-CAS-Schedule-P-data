# P&C Loss Reserving Engine on CAS Schedule P Data

**Robbie Richter** · [One-page summary](https://richterrw.github.io/P-C-loss-reserving-engine-on-CAS-Schedule-P-data/) · [Full report](reports/REPORT.md)

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A from-scratch reserving engine (NumPy/SciPy, no actuarial libraries) that runs **chain ladder,
Bornhuetter-Ferguson, Cape Cod, Benktander, Munich chain ladder, Mack, and the ODP bootstrap (with
Shapland's refinements)** on real company triangles from the NAIC Schedule P. It then backtests every
method against what actually happened over the following nine years.

The point of the project is the comparison. The methods don't just give different numbers: each one
fails in its own way, and you can see why in the triangle.

> **Full generated report:** [`reports/REPORT.md`](reports/REPORT.md). It has validation tables,
> three company deep dives, a backtest of 327 company/line combinations across all six lines, and the
> extensions (Munich chain ladder, Berquist-Sherman, Shapland bootstrap, tail study).
>
> **One-page summary:** [richterrw.github.io/…](https://richterrw.github.io/P-C-loss-reserving-engine-on-CAS-Schedule-P-data/) (source in [`docs/index.html`](docs/index.html)).

---

## What this project demonstrates

* **Reserving methods from first principles.** Chain ladder, Bornhuetter-Ferguson, Cape Cod,
  Benktander, Munich chain ladder, Mack and the ODP bootstrap, written from the papers rather than
  called from a library. This covers core material from the CAS Exam 5 and Exam 7 syllabi
  (Friedland, Mack, Shapland).
* **Validation discipline.** Every method reproduces published figures exactly before it is used on
  real data, and an automated test suite enforces this on every commit.
* **Testing assumptions against outcomes.** 327 real company triangles are scored against nine years
  of actual development, so the conclusions rest on outcomes rather than on the models' own
  assumptions.
* **Explaining results, not just producing them.** Each disagreement between methods is traced to a
  cause in the data: leverage, settlement speed, the prior, or case reserve adequacy. The results
  that cut against the expected story are reported too.
* **Communicating to different audiences.** A one-page summary for a quick read, this README for
  practitioners, and a generated report with every table and chart.

---

## Headline results

| | |
|---|---|
| **Validation** | Matches published results exactly: Mack (1993) RAA s.e. **26,909**, Taylor-Ashe s.e. **2,447,095**, England-Verrall ODP scale **52,601**, Mack's (1994) assumption tests, and chainladder-python's Munich chain ladder on the Quarg-Mack data to the cent. |
| **Most accurate point estimates** | Incurred-based methods. Median reserve error is −3% to 0% for CL / BF / Cape Cod / Benktander on incurred data, against +10% to +12% for the same methods on paid data. |
| **Systematic bias** | **Paid chain ladder over-reserved in every line** (dollar-weighted bias +8% to +35%). Paid-to-incurred ratios show that claims were being settled faster in the 1990s, and a paid triangle can't see that. |
| **Company booked reserves** | Booked reserves were redundant by a median of **+21%**, and by +39% in workers' comp and +48% in med mal (dollar-weighted). On pooled median absolute error, every incurred method beat the company's own booked figure. |
| **Reconciling paid and incurred** | **Munich chain ladder removes the paid bias** (median −1% vs. +10%) and shrinks the median paid/incurred gap from 3.9% to 0.8%. On accuracy it only *ties* the best incurred methods (median absolute error 17.5% vs. 17.9% for Cape Cod), because it inherits incurred's own errors. |
| **Variability models** | Mack and the bootstrap agree closely with each other but are **miscalibrated against reality**. On paid data, 27–30% of outcomes fell below the models' 10th percentile when 10% should. The bias is in the mean, so no extra spread can fix it. |
| **A negative result worth knowing** | Without claim counts, a Berquist-Sherman settlement adjustment is *algebraically identical* to incurred chain ladder. Proved [below](#berquist-sherman-without-claim-counts) and asserted in the tests. |

![errors](reports/figures/backtest_errors.png)

---

## Data

The source is the **[CAS Loss Reserving Database](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p)**
(Meyers & Shi), built from NAIC Schedule P Parts 1–4. It has six lines (workers' comp, private
passenger auto, commercial auto, medical malpractice, products liability, other liability), accident
years 1988–1997, and **ten development years for each one**. At the 12/1997 valuation only the upper
triangle was known. The lower triangle (calendar years 1998–2006) is the realised outcome, which is
what makes an honest backtest possible.

The six CSVs in `data/raw/` are the unmodified CAS files. They were retrieved from the
[reservetestr](https://github.com/problemofpoints/reservetestr) mirror because casact.org was not
reachable from the build environment. All 42,845 upper-triangle cells match the independent copy that ships with
`chainladder-python`.

Field conventions follow Meyers (2015):

| Quantity | Definition |
|---|---|
| Paid | `CumPaidLoss` |
| Case incurred | `IncurLoss − BulkLoss` (paid + case reserves) |
| Company booked ultimate | `IncurLoss` (includes the company's own IBNR) |
| Premium | `EarnedPremNet` |
| **Actual ultimate** | case incurred at lag 10 |

**Data screen** (`clean_companies`): positive premium and positive cumulative paid and incurred in
every observed cell, at least $1M of total premium, and some movement after 1997 (a few groups have
frozen lower triangles). That leaves 327 of the 779 company/line triangles.

---

## Methods

All methods live in `reserving/` and are validated in `tests/`.

| Module | Method | Notes |
|---|---|---|
| `chainladder.py` | **Chain ladder** | Volume-weighted age-to-age factors, optional exponential tail |
| `bf.py` | **Bornhuetter-Ferguson** | `U = C + (1 − 1/CDF)·ELR·P` |
| `bf.py` | **Cape Cod** (Stanard-Bühlmann) | ELR = reported ÷ used-up premium Σ P/CDF, so it comes from the company's own data |
| `bf.py` | **Benktander** | BF iterated once; credibility `Z = 1 − (1 − 1/CDF)²` |
| `mack.py` | **Mack (1993)** | Process and parameter s.e. by AY and in total, including the covariance term; Mack's σ² extrapolation for the last age |
| `bootstrap.py` | **ODP bootstrap** (England & Verrall 2002) | Pearson residuals with a DoF adjustment, refit on each pseudo-triangle (parameter risk), gamma process risk with signed handling of negative incrementals; vectorised (10k sims in about 0.2 s) |
| `bootstrap.py` | **Shapland refinements** | Hat-matrix residual adjustment (leverage of the ODP GLM) and heteroscedastic scale parameters by development-age group |
| `munich.py` | **Munich chain ladder** (Quarg & Mack 2004) | Joint paid/incurred projection using the correlation between P/I ratio residuals and development-factor residuals |
| `adjustments.py` | **Berquist-Sherman** settlement adjustment | Counts-free proxy; see [below](#berquist-sherman-without-claim-counts) for why it reduces to incurred chain ladder |
| `tail.py` | **Tail factors** | Exponential and inverse-power curves, plus a truncation test of how well each extrapolates |
| `diagnostics.py` | Why the methods disagree | CL–BF gap decomposition, paid/incurred ratios, actual vs. expected on the latest diagonal, Mack (1994) calendar-year and correlation tests |

**A priori ELR for BF and Benktander.** The prior is the **industry loss ratio by accident year**:
a chain ladder on the line's aggregate triangle across all clean companies. It uses only data known in
1997, and it follows the underwriting cycle year by year instead of applying one flat number.

**One target for everyone.** Every point estimate is scored against case incurred at lag 10. Paid
methods get a paid-to-incurred tail equal to the oldest year's incurred/paid ratio at age 10, which is
on the 1997 diagonal and so known at the valuation date. The variability models are scored on their own
basis (paid models against paid at lag 10), so each one is judged on the quantity it actually models.

---

## Why the methods disagree

All the methods share the same development data, but they make different assumptions about what that
data means. In practice the disagreement comes from four sources (1–4). Sections 5 and 6 cover what
that means for the variability models and for reconciling the two triangles.

### 1. Leverage × surprise: chain ladder vs. BF

Every expected-loss method is a credibility blend of the chain ladder ultimate and a prior. A line of
algebra gives the identity at the centre of this project:

```
U_CL − U_BF  =  (1 − 1/CDF)  ×  (U_CL − U_prior)
                 ^ leverage       ^ surprise
```

BF gives the actual emergence credibility `Z = 1/CDF`. When a year is 97% unreported, the chain ladder
multiplies whatever has been paid by 30. BF instead trusts the prior for the 97% that hasn't happened yet.

**State Farm Other Liability, AY 1997.** $21M had been paid by 12/1997 and the paid CDF was **30.6**.
Chain ladder projects $645M, a 161% loss ratio. BF projects $307M. The actual outcome was $329M. In
this triangle, **90% of the total paid CL–BF gap sits in the three most recent accident years**, and
that pattern holds for nearly every company. When CL and BF disagree, the cause is almost always
immature years, and the size of the gap is set by the CDF.

![gap](reports/figures/othliab_1767_gap.png)

The identity also shows when **BF is the method that's wrong**. For this company the industry prior
(~68%) was far below its actual loss ratios (81–113%), so BF on paid data was 15% light. The two
errors pull in opposite directions, which is why Benktander, sitting between them, came within 2%.
BF is only as good as its prior, and chain ladder is only as good as its CDF.

### 2. Paid vs. incurred: different data, different blind spots

Paid chain ladder assumes **settlement speed is stable**. Incurred chain ladder assumes **case
reserve adequacy is stable**. The ratio of paid to incurred at a fixed age shows which assumption
broke:

* **Rising ratio**: claims are paid faster, or case reserves are set weaker. Paid CL over-projects
  because recent years look more developed than they are.
* **Falling ratio**: payment slows, or case reserves are strengthened. Incurred CL over-projects.

At State Farm Other Liability, the age-1 paid/incurred ratio jumped from 12% (AY 1995) to 25% (AY 1997).
At State Farm PP Auto, the age-2 ratio rose steadily from 78% to 84% (p < 0.01). Both are payment
speed-ups, and in both cases paid CL came in highest of all the methods. Across the whole database,
**paid chain ladder over-reserved in all six lines**, while incurred chain ladder was close to unbiased
(median −3%). This matches Meyers (2015), who traced the paid bias to faster claim settlement during
this period. The Berquist-Sherman adjustments and the Munich chain ladder (section 6) exist for exactly
this situation.

### 3. The prior: industry vs. the company's own data

BF and Cape Cod run the same formula and differ only in where the ELR comes from. **Cape Cod**
estimates one flat ELR from the company's own data, weighted toward mature years. In a book whose loss
ratios are trending, that's a problem. State Farm PP Auto's actual loss ratio fell from 87% (AY 1988)
to 68% (AY 1997). Cape Cod's flat 79% overstated every recent year, and **Cape Cod on paid data was
the worst method for that company (+16%)**. BF used the industry prior, which follows the cycle by
year, and was off by +6%. The reverse holds for a company that doesn't look like its industry, like
State Farm Other Liability above, where a company-specific prior would have done better.

### 4. Negative development and management judgement

Scpie Medical Malpractice had **incurred CDFs below 1.0** for AYs 1990–95 (0.94–0.998). Its case
reserves were over-established and later released. Incurred methods picked this up and landed within
±9% of the actual. Paid CL, blind to case reserves, was +48%. The company's own booked ultimate was
**+96%** of the actual unpaid, the most conservative figure of all. That isn't a method error. It shows
the reserve cycle: med mal and workers' comp carriers held large redundancies through the late 1990s
(booked bias +48% and +39%), and the redundancies were released over the following decade. Mack's
calendar-year test flags the incurred triangle (Z = 7, below the 95% band). Its latest diagonal came
in at 0.82–0.84× the expected value for the recent years, so the takedowns were already visible in 1997.

### 5. Mack vs. bootstrap: agreement isn't accuracy

Mack and the ODP bootstrap rest on the same mean structure (volume-weighted chain ladder), so their
centres agree. They differ in the variance assumption. Mack makes variance proportional to the
*cumulative* loss; ODP makes it proportional to the *incremental* mean. On paid data the two give
nearly the same spread (median bootstrap/Mack s.e. ratio 1.06, interquartile range 0.97–1.25). On
incurred data they diverge: incurred incrementals late in development hover around zero, which
inflates the ODP scale parameter, and the bootstrap comes out about 2.6× wider than Mack (median CV
9.6% vs. 3.6%). **Neither one is calibrated against reality:**

![calibration](reports/figures/backtest_calibration.png)

| Model, basis | below p10 | p10–p90 | above p90 | median CV (width) |
|---|---|---|---|---|
| Mack, paid | 27% | 59% | 14% | 4.1% |
| Bootstrap, paid | 28% | 60% | 13% | 4.3% |
| Bootstrap (Shapland), paid | 30% | 57% | 13% | 3.9% |
| Mack, incurred | 14% | 64% | 22% | 3.6% |
| Bootstrap, incurred | 9% | 81% | 9% | 9.6% |
| Bootstrap (Shapland), incurred | 10% | 78% | 12% | 6.6% |
| *ideal* | *10%* | *80%* | *10%* | *as narrow as possible* |

On paid data the two models fail together. Outcomes pile up below the 10th percentile, because the
chain ladder mean is biased high (section 2). Widening the distribution can't fix that. Mack on incurred
data is too narrow in both tails, which Meyers also found. The plain ODP bootstrap on incurred looks
close to calibrated, but only because it is 2.6× wider; for a handful of companies with negative incurred
development its spread is meaningless. Shapland's refinements (hat-matrix residuals and a separate
scale parameter per group of development ages) cut that width by a third (CV 9.6% → 6.6%) while keeping
coverage near nominal (10% / 78% / 12%). That makes it the best-behaved variability model here, though it
still fails a KS test at 5%. Calibration has to be judged together with sharpness: a model can pass
coverage tests just by being uninformative. The lesson: **process and parameter risk are the smaller
part. Model risk, meaning the chance that the development pattern itself changes, dominates, and no
single-model variability estimate captures it.**

### 6. Reconciling paid and incurred: Munich chain ladder

Separate chain ladders ignore what each triangle says about the other. The Munich chain ladder
(Quarg & Mack 2004) uses the observed correlation: an accident year whose paid is low *relative to its
case incurred* tends to have high paid development next period, and low incurred development. Fitted
across the 327 triangles, that correction removes the paid bias entirely:

![mcl](reports/figures/mcl_by_line.png)

| Method | median error | median abs. error | within ±20% |
|---|---|---|---|
| CL Paid | +10% | 25% | 42% |
| CL Incurred | −3% | 19% | 51% |
| MCL Paid | −1% | 20% | 50% |
| MCL Incurred | 0% | 18% | 54% |

The median gap between paid and incurred ultimates falls from 3.9% to 0.8%. But MCL reconciles towards
the *case reserves*, so where case reserves were themselves wrong it inherits the error. For Scpie
Medical Malpractice, whose case reserves were redundant, MCL is +28% against +8% for incurred chain
ladder. Reconciliation is not the same as accuracy: when paid and incurred disagree, you still have to
decide which data to believe.

---

## Berquist-Sherman without claim counts

The classic Berquist-Sherman (1977) paid adjustment restates historical paid losses to today's
disposal rate, using closed claim counts. The CAS database has no counts, so the natural substitute
uses incurred as the yardstick: restate every historical paid cell to the paid/incurred ratio that the
latest diagonal shows at that age (`reserving/adjustments.py`).

On the data, that adjustment reproduces incurred chain ladder to machine precision. The algebra shows
why. Each adjusted column is a constant multiple `r_j` of the incurred column, so the adjusted factors
are `f^I_j · r_{j+1} / r_j`. The latest paid value equals `I · r_a` for the same accident year that
set `r_a`. The projection telescopes to

```
U_adj  =  I_latest · CDF^I(a) · r_10  =  U_CL,incurred · r_10
```

and the paid-to-incurred tail `1 / r_10` cancels the last term. **A settlement adjustment that measures
speed against incurred losses can only return the incurred answer.** The independent information in
Berquist-Sherman comes from the claim counts, which is a useful thing to know before asking for
them.

## Tail factors: how far is lag 10 from ultimate?

`reserving/tail.py` fits exponential and inverse-power curves to the development portion `f − 1`. To
test the extrapolation step every tail factor depends on, each paid triangle is cut at age 5. The curves
are fitted on ages 2–5 and asked to predict development from 6 to 10, which the triangle actually shows.
The **exponential curve is close to unbiased** (median −9% of the true development), but it misses by
about half for a typical single company. The **inverse-power curve roughly doubles** the true
development. On industry aggregates, lag 10 is effectively ultimate for the auto lines (paid tail
≤ 1.005). Workers' comp still carries about 6% in case reserves and 3% in IBNR at age 10, so the
backtest's "actual" understates true ultimate there by a few percent, a shortfall every method shares.
Full tables are in [report §4.4](reports/REPORT.md#44-tail-factors).

---

## What a reserving actuary should take from this

1. **Always run paid and incurred, and then explain the gap between them.** The paid/incurred ratio by
   age is the first diagnostic to check.
2. **Use BF or Benktander for immature years, and pay as much attention to the prior as to the
   development pattern.** When the prior and the development pattern are both uncertain, Benktander
   hedges between them.
3. **Leverage is the risk.** CDF × emerged loss means one unusual payment in AY 1997 moves the
   ultimate 30×.
4. **Treat Mack or bootstrap ranges as lower bounds on uncertainty.** Calibrate them against history
   wherever you can, and judge width alongside coverage.
5. **Reconcile paid and incurred (Munich chain ladder), but decide which one to trust.** Reconciling
   them removes the inconsistency between the two, not the uncertainty.
6. **Expect booked reserves to differ from any method.** Management margin follows the cycle.

### Translation to health and life reserving

The mechanics carry over directly. Health IBNR **completion factors** are a chain ladder on a
lag triangle by incurred month, and the "paid vs. incurred" problem shows up as changes in claim
processing speed (a backlog or an auto-adjudication change). New products and recent incurred months
are handled with **BF or a PMPM-trend prior**, which is the same leverage × surprise trade-off.
Disability and LTC claim reserves face the same tension between experience and an a priori (table)
assumption that BF formalises. Calibration backtesting like section 3 is what an actuarial-memorandum
reviewer should ask for.

---

## Limitations

* The "actual" ultimate is case incurred at lag 10. Long-tail lines still had open claims in 2006. The
  tail study puts the shortfall at a few percent for workers' comp and well under 1% for the auto lines.
* The variability models are scored to lag 10 without a tail. Point estimates use an observed
  paid-to-incurred tail rather than a fitted curve, so they are scored on the same footing as the
  "actual".
* Without claim counts, a true Berquist-Sherman adjustment isn't possible ([see above](#berquist-sherman-without-claim-counts)).
* The industry prior includes the company itself, though it uses only pre-1998 data.
* The Munich chain ladder is reported as a point estimate only; its stochastic version (a bootstrap of
  MCL) isn't implemented.
* ODP is designed for paid data. Even with Shapland's refinements its incurred results are a stress
  test, not a recommendation.
* The data screen's "post-1997 movement" rule looks at outcomes. It removes only six triangles with
  essentially no development after 1997.

---

## Usage

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -q
```

```bash
python -m reserving --lob othliab --list
```

```bash
python -m reserving --lob othliab --company 1767
```

```bash
python scripts/build_report.py
```

The CLI's `--list` option shows clean companies largest first. `--company` runs every method plus
diagnostics on one company. `build_report.py` regenerates `reports/` (takes about 40 s).

As a library:

```python
from reserving.data import company
from reserving.chainladder import chain_ladder
from reserving.bf import bornhuetter_ferguson
from reserving.mack import mack
from reserving.bootstrap import odp_bootstrap
from reserving.munich import munich_chain_ladder

cd = company("wkcomp", 86)                       # Allstate workers' comp
cl = chain_ladder(cd.incurred)
bf = bornhuetter_ferguson(cl, cd.premium, elr=0.70)
m  = mack(cd.paid);  print(m.reserve.sum(), m.total_se)
b  = odp_bootstrap(cd.paid, n_sims=10_000, hat_adjust=True, hetero=True)
print(b.percentile([50, 75, 95]))
mcl = munich_chain_ladder(cd.paid, cd.incurred); print(mcl.lambda_paid, mcl.paid_ultimate.sum())
```

## Layout

```
reserving/        engine: triangle, chainladder, bf, mack, bootstrap, munich, adjustments, tail,
                  diagnostics, data, engine, backtest
scripts/          build_report.py (report + figures + CSVs)
reports/          REPORT.md, figures/, backtest_*.csv, tail_study.csv  (generated)
docs/index.html   one-page summary
data/raw/         CAS Schedule P files, one per line
data/benchmarks/  RAA, Taylor-Ashe and Quarg-Mack triangles for validation
tests/            validation against published figures + Schedule P checks + extensions
```

## References

* Mack, T. (1993). *Distribution-free calculation of the standard error of chain ladder reserve estimates.* ASTIN Bulletin 23(2).
* Mack, T. (1994). *Measuring the variability of chain ladder reserve estimates.* CAS Forum.
* England, P. & Verrall, R. (2002). *Stochastic claims reserving in general insurance.* British Actuarial Journal 8(3).
* Shapland, M. (2016). *Using the ODP Bootstrap Model: A Practitioner's Guide.* CAS Monograph 4.
* Quarg, G. & Mack, T. (2004). *Munich Chain Ladder.* Blätter der DGVFM 26(4); Variance 2(2), 2008.
* Berquist, J. & Sherman, R. (1977). *Loss Reserve Adequacy Testing: A Comprehensive, Systematic Approach.* PCAS LXIV.
* Sherman, R. (1984). *Extrapolating, Smoothing and Interpolating Development Factors.* PCAS LXXI.
* Meyers, G. (2015). *Stochastic Loss Reserving Using Bayesian MCMC Models.* CAS Monograph 1.
* Friedland, J. (2010). *Estimating Unpaid Claims Using Basic Techniques.* CAS.
* Bornhuetter, R. & Ferguson, R. (1972). *The Actuary and IBNR.* PCAS LIX.
