# Expert Analysis Guide: BLP Automobile Dataset & Discrete Choice Demand Estimation

Comprehensive reference for what IO economists and econometricians actually do with market-level discrete choice data — from raw data checks through structural estimation, policy simulation, and publication-grade validation. Grounded in the BLP (1995) automobile dataset but applicable to any differentiated-products market.

**Key references:**
- Berry, Levinsohn & Pakes (1995), *Econometrica* 63(4):841–890 — the original paper
- Nevo (2000), *JEMS* 9(4):513–548 — practitioner's guide
- Conlon & Gortmaker (2020), *RAND Journal* — PyBLP best practices
- Berry & Haile (2021), NBER WP 29305 — foundations of demand estimation
- Gandhi & Houde (2023) — differentiation instruments

---

## 1. Pre-Estimation Data Diagnostics

The most important work happens before estimation. Experts spend significant time here because mistakes at this stage corrupt everything downstream.

### 1.1 Share Accounting

**Check: all inside shares sum to strictly less than 1 in every market.**

```python
inside_sum = df.groupby('market_ids')['shares'].sum()
assert (inside_sum < 1).all(), "Inside shares must not exhaust potential market"
assert (inside_sum > 0).all()
```

The outside good share `s_0t = 1 - Σ_j s_jt` must be positive and economically plausible. For the BLP automobile dataset the outside good is ~95% (most households do not buy a new car in any given year). A very small outside good (<10%) may indicate a misdefined potential market size; a very large outside good (>99%) may indicate misscaled market sizes.

**What to plot:** `s_0t` over time. A dramatic trend in the outside good often reflects changes in market size definition, not demand.

**Market size definition:** The BLP automobile dataset uses the U.S. population of driving-age adults as the potential market. This is a researcher choice — document it and check sensitivity. Nevo (2000) uses city-quarter households. Wrong market size shifts all price elasticities.

### 1.2 Price Distribution and Endogeneity Pre-Check

Prices are almost certainly endogenous: firms set higher prices for higher unobserved quality. The raw OLS correlation between `prices` and `log(s_j/s_0)` is contaminated by this. Evidence of endogeneity:

- Running OLS gives a **positive** price coefficient in `log(share) ~ price + characteristics` — this is the classic sign flip caused by omitted quality.
- Regress `prices` on product characteristics. The residuals should be uncorrelated with instruments but correlated with unobserved quality. Examine whether high-residual products are intuitively high-end (justifies the IV approach).

**Formal Hausman endogeneity test:**
1. Regress `prices` on instruments + exogenous regressors; save residuals `v̂`.
2. Add `v̂` to the OLS logit regression. If the coefficient on `v̂` is significantly nonzero, price is endogenous. This is the control function / Hausman test.

### 1.3 Multicollinearity Among Characteristics

High correlation between `hpwt`, `space`, `mpg`, and `air` can make individual coefficients unstable. Standard VIF check:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF > 10 is problematic; > 5 warrants scrutiny
```

For the BLP data, `mpg` and `mpd` will be highly correlated (mpd = mpg × fuel price). Use one or the other in the utility specification — BLP (1995) uses `mpd` as the relevant demand variable.

### 1.4 Missing Variation / Market Thinness

Check: in markets with few products, the model may be poorly identified. For the BLP data (minimum 72 products/year), this is not an issue. For other datasets with 5–10 products per market, check whether GMM is over-identified.

**Rule of thumb:** You need the number of moments (instruments × product-level observations) to substantially exceed the number of parameters. With 33 instruments and 2,217 observations estimating ~15 structural parameters, the BLP data is well-identified.

### 1.5 Structural Breaks and Outliers

Plot each characteristic and market share over time:
- Discrete jumps suggest new product launches, model redesigns, or oil price shocks
- 1973 and 1979 oil crises appear clearly in `mpd` and `mpg` trends for the BLP data
- The entry of Japanese and European imports appears in the `region` composition after 1975

Flag product-market observations with extreme residuals after logit OLS as potential outliers — these may represent data errors or genuinely unusual products.

---

## 2. Demand-Side EDA — Standard Plots and Statistics

### 2.1 The Core Transformation: `log(s_j) - log(s_0)`

This is the starting point for every demand analysis. In the simple logit, `log(s_j) - log(s_0) = δ_j = x_j'β - αp_j + ξ_j`. Plot this transformed variable against price to get the demand relationship on a "linear" scale. The slope should be negative.

```python
df['log_share'] = np.log(df['shares'])
df['log_outside'] = np.log(1 - df.groupby('market_ids')['shares'].transform('sum'))
df['logit_y'] = df['log_share'] - df['log_outside']
```

**What to look for:**
- Slope of `logit_y ~ price`: should be −3 to −8 for automobiles after instrumentation
- Dispersion of `logit_y` within a market: reflects unobserved quality heterogeneity `ξ_j`
- Time trend in `logit_y` holding price constant: quality upgrading over time

### 2.2 Share Distribution

A right-skewed distribution is normal — most models have small shares, a few models dominate. Plot the share distribution with log-scale x-axis. Check:
- **Top model share** per market: should not exceed ~1% for a broad market
- **Share of top-3 models** per market: concentration measure
- **Coefficient of variation** of shares within a market: high CV = heterogeneous products

### 2.3 Price-Characteristic Hedonic Regression

Regress `price ~ characteristics + year_FE`:

```python
# R² ~ 0.6-0.8 is typical for automobiles
# Positive coefficients on hpwt, air, space are expected
# Negative coefficient on mpd expected (economy cars cheaper)
```

**Why this matters:** The residuals from this regression are the "unexplained price premium" — related to brand equity and cost shocks. These residuals motivate the IV strategy. High R² means characteristics explain most price variation; low R² means unobserved quality is dominant and IV matters more.

### 2.4 Within-Market vs. Cross-Market Variation

BLP-style estimation uses **cross-market** variation (different years have different product sets and prices) for identification. Check:

- **Product entry/exit**: which models are present in some but not all markets? This variation in product availability is a key source of identification.
- **Price variation for the same model across markets**: a car model present in multiple years with different prices provides direct within-model price variation.
- **Characteristics change**: the `trend` variable in BLP captures quality upgrading. Plot `hpwt` and `mpg` over time — should rise monotonically after 1975.

### 2.5 Market Concentration

Compute the Herfindahl-Hirschman Index (HHI) per market:

```python
hhi = df.groupby('market_ids').apply(
    lambda g: ((g['shares'] / g['shares'].sum()) ** 2).sum() * 10000
)
# HHI < 1500: competitive | 1500-2500: moderate | >2500: concentrated
```

For the BLP data, the HHI on the inside market runs ~800–1,000 throughout 1971–1990 — consistently unconcentrated. An antitrust analyst would use this as a benchmark for assessing post-merger concentration changes (the "delta HHI" threshold is 200 points under DOJ/FTC guidelines).

### 2.6 Cross-Sectional Substitution Suggestiveness

Before running formal elasticities, examine: when a product has a price spike in year t, do similar products gain share? Plot share changes against price changes for product pairs defined as similar (same segment, same origin region). This is informal evidence of reasonable substitution patterns and motivates nest choice.

---

## 3. Simple Logit Baseline — OLS and IV

### 3.1 The OLS Logit

**Model:** `log(s_jt) - log(s_0t) = δ_jt = α·p_jt + x_jt'β + ξ_jt`

Estimated by OLS on the transformed dependent variable. The OLS price coefficient **will be biased upward** (toward zero or positive) because price and `ξ_jt` are positively correlated — firms charge more for high unobserved quality.

In the BLP data, OLS typically yields `α ≈ +0.02` (positive — wrong sign), while IV yields `α ≈ -0.09` (correct sign). This sign flip is one of the most dramatic illustrations of endogeneity bias in empirical IO.

**From the BLP data**: 1,494 of 2,217 product-market observations have inelastic demand under OLS. After instrumentation, only 22 observations remain inelastic — a massive correction.

### 3.2 The IIA Problem

Under plain logit, the cross-price elasticity between products j and k is:

`ε_jk = α · p_k · s_k` (for j ≠ k)

This implies that **all products substitute equally** regardless of similarity — a Chevy Corvette and a Ford Pinto are treated as equally substitutable. The "red bus / blue bus" paradox: introducing a blue bus with identical attributes to the red bus halves the red bus share (correct) but also halves the car share (wrong).

**The IIA is a hard testable restriction** — if it holds, adding or removing a product should not affect relative shares of other products. For automobile data it clearly fails: Japanese entry in the 1970s displaced similar-segment domestic models disproportionately.

### 3.3 Hausman-McFadden IIA Test

**Test procedure:**
1. Estimate the full model on all J products.
2. Drop product k from the choice set. Re-estimate on the restricted set.
3. Under IIA, the parameter vector should be the same. Test: `H = (β̂_R - β̂_F)'[V̂_R - V̂_F]⁻¹(β̂_R - β̂_F) ~ χ²(K)`

A significant test statistic rejects IIA and justifies nested logit or BLP. In practice for automotive data this test almost always rejects at the 1% level. The test has known power issues with weak instruments; Horowitz (1983) and Small-Hsiao (1985) proposed alternatives.

**What "wrong" logit estimates look like:**
- Positive price coefficient (sign flip)
- Price elasticities all around −1 to −2 (too homogeneous)
- Cross-price elasticities identical across all product pairs
- Predicted post-merger price increases too small (logic: too-easy substitution to outside good)

---

## 4. Nested Logit Estimation

### 4.1 Nest Construction

Nests should group products that consumers consider close substitutes — those with similar characteristics, same market segment, or same origin. **Expert practice:**

- **For automobiles**: segment (compact/mid-size/luxury) or origin region (US/EU/JP) or fuel type
- **Nest assignment should be pre-specified** using economic theory, not chosen to maximize model fit (data-mining nests inflates σ estimates)
- **Robustness check**: vary the nesting structure and show results are stable

In the BLP dataset, the three natural nests are US domestic, European imports, and Japanese imports. This structure aligns with the competitive dynamics of the 1971–1990 period.

### 4.2 The Nested Logit Equation

`log(s_jt) - log(s_0t) = α·p_jt + x_jt'β + σ·log(s_{j|g,t}) + ξ_jt`

where `s_{j|g,t} = s_jt / Σ_{k∈g} s_kt` is the within-nest share.

**Key parameter:** σ ∈ [0,1] (nesting parameter / dissimilarity parameter)
- σ = 0: collapses to standard logit (IIA holds)
- σ → 1: consumers never switch nests (extreme nesting correlation)
- 0.5–0.8 is typical for automobiles in published papers

**Validity check:** If estimated σ < 0 or σ > 1, the model is misspecified. Negative σ is the most common failure mode and usually indicates: wrong nest structure, endogenous within-nest share not instrumented, or overly broad market definition.

### 4.3 IV for Within-Nest Share

`log(s_{j|g,t})` is endogenous — it depends on unobserved quality. Standard instruments:
- **Count of products in the nest** per market: more products → lower within-nest share mechanically; plausibly exogenous to individual product quality
- **Characteristics of other products in the nest**: sum of competitors' `hpwt`, `mpg`, etc.

Check the first-stage: regress `log(s_{j|g,t})` on instruments + controls. F-statistic should exceed 10. Partial R² of instruments should be substantial.

### 4.4 Comparing Nested Logit to Plain Logit

In the PyBLP cereal example:
- Plain logit: mean own-price elasticity ≈ −30 (too elastic, homogeneous substitution)
- Nested logit (σ̂ ≈ 0.89): mean own-price elasticity ≈ −72 (very elastic within-group)
- The adjusted own-price elasticity formula: `ε_jj ≈ α·p_j / (1 - σ)` — larger σ → larger elasticities within nests

For automobiles, Goldberg (1995) reports average elasticity of 3.28 under nested logit; BLP (1995) report 3–6 depending on specification.

---

## 5. Generalized Nested Logit (GNL)

### 5.1 What Makes It "Generalized"

Standard nested logit is a special case of GNL where products belong to **exactly one** nest with fixed membership. GNL allows:
1. **Overlapping nest membership**: a product can partially belong to multiple nests (Ford Fusion as both a mid-size car and a domestic car)
2. **Heterogeneous nesting parameters**: different σ_g for each nest rather than a single σ
3. **Product-specific allocation parameters**: α_jg ∈ [0,1] is the fraction of product j allocated to nest g, with Σ_g α_jg^(1/λ_g) = 1 as the adding-up constraint

### 5.2 GNL Utility Specification

Consumer i's utility from product j:
`U_ij = δ_j + ζ_{ig} + (1-σ_g)ε_ij`

where ζ_{ig} is the nest-specific shock. The choice probability becomes:

`s_j = Σ_g α_jg · P(nest g) · P(j|nest g, chosen nest g)`

The additional flexibility comes at a cost: with J products and G nests, there are up to J×G allocation parameters to estimate, requiring more variation in the data.

### 5.3 Practical Differences from Standard Nested Logit

- GNL is consistent with random utility maximization for any σ_g ∈ [0,1]; standard NL is not always RUM-consistent
- Substitution patterns are richer: a product in two nests substitutes with products from both groups
- Estimation is harder: outer-loop optimization over σ_g and α_jg simultaneously
- Ford used GNL precisely because trims can belong to multiple competitive clusters (e.g., the Fiesta ST competes with both hot hatches and small economy cars)

---

## 6. BLP Random-Coefficients Logit — The Full Model

### 6.1 Model Specification

Consumer i's utility from product j in market t:

`U_ijt = α_i · p_jt + x_jt'β_i + ξ_jt + ε_ijt`

where the random coefficients are:
`(α_i, β_i) = (ᾱ, β̄) + Σ·ν_i + Π·D_i`

- `ν_i ~ N(0,I)`: unobserved consumer heterogeneity (random draws)
- `D_i`: observed demographics (income, age, etc.)
- `Σ`: diagonal matrix of random coefficient standard deviations
- `Π`: matrix of demographic interaction parameters
- `ε_ijt ~ Type-I Extreme Value`: i.i.d. idiosyncratic shocks

This specification allows **flexible substitution patterns**: consumers with high income are less sensitive to price (negative income interaction with price coefficient), creating realistic substitution between similar vehicles.

### 6.2 The Contraction Mapping (Inner Loop)

The central computational challenge: given parameters `θ = (Σ, Π)`, find the mean utilities `δ_jt` that make predicted market shares match observed shares exactly.

**Fixed-point iteration (Berry 1994):**
`δ_jt^{n+1} = δ_jt^n + log(s_jt^{obs}) - log(σ_jt(δ^n; θ))`

where `σ_jt(δ; θ) = ∫ exp(δ_jt + μ_ijt) / (1 + Σ_k exp(δ_kt + μ_ikt)) dF(ν_i, D_i)` is the simulated market share.

**Convergence criterion**: typically `max|δ^{n+1} - δ^n| < 10^{-12}`. This must be tight — loose convergence in the inner loop biases the outer-loop GMM estimates.

**Key insight**: After the contraction mapping, `ξ_jt = δ_jt - x_jt'β_L` (where β_L is recovered by IV-GMM on the linearized problem). These `ξ_jt` are the structural errors — unobserved quality.

### 6.3 GMM Objective Function (Outer Loop)

**Moment conditions:** `E[ξ_jt(θ) · Z_jt] = 0`

where `Z_jt` is the instrument matrix. The GMM objective:

`Q(θ) = ξ(θ)' Z W Z' ξ(θ)`

Minimized over `θ = (Σ, Π)`. The optimal weighting matrix `W` is the inverse of the moment covariance: `W = (Z'ΞZ/N)^{-1}` where `Ξ = diag(ξ_jt²)`. Two-step GMM: use identity weighting in step 1, update `W` using step-1 residuals, re-estimate.

### 6.4 Identification

The random coefficients are identified by:
- **Cross-market variation**: different markets have different product sets and demographics — products with the same mean utility but different characteristics lead to different shares across markets
- **Micro moments**: if household-level data is available showing which income group bought which car, this directly identifies `Π` (the demographic interactions)
- **Optimal instruments** (Chamberlain 1987, Gandhi-Houde 2019): the expected Jacobian `E[∂ξ/∂θ]` evaluated at the truth — identified using variation in competitor characteristics

**Key identification problem**: with only market-level data and no micro data, the random coefficients `Σ` are identified purely from the relationship between market-level shares and product characteristics across markets. Thin variation → poorly identified `Σ`.

### 6.5 Micro Moments

When household-level purchase data is available (e.g., Consumer Expenditure Survey in Goldberg 1995, or NielsenIQ panel data), it provides additional moment conditions:

`m_data - m_model(θ) = 0`

where `m` could be:
- Mean income of buyers of model j
- Fraction of high-income households buying luxury vehicles
- Covariance of income and price paid

Micro moments dramatically improve identification of `Π` (how demographics interact with product characteristics) because they provide direct evidence on who buys what, not just aggregate shares.

---

## 7. Instrument Diagnostics

### 7.1 Types of Instruments

**BLP instruments (original):**
- `BLP_own_jt^k = Σ_{k≠j, same firm} x_k^k` — sum of characteristic k across same-firm products
- `BLP_rival_jt^k = Σ_{k, different firm} x_k^k` — sum of characteristic k across rival products

**Differentiation instruments (Gandhi-Houde 2019, preferred in modern work):**
- `GH_jt^k = Σ_{k≠j} 1(|x_j^k - x_k^k| < ε)` — count of nearby competitors in characteristic space
- Captures "how isolated is product j" — more isolated → higher markup power → positively correlated with price
- Outperforms BLP instruments when markets are large and products are densely packed

**Supply-side instruments:**
- Input cost shifters (steel prices, aluminum prices for automobiles)
- Geographic distance to production facilities
- These instruments shift marginal costs but are excluded from demand (valid exclusions if cost shocks don't directly affect consumer utility)

### 7.2 First-Stage Regression

For each endogenous variable (price, within-nest share), run:
`endogenous_var = Z'γ + X_exog'δ + error`

**Diagnostics:**
- **First-stage F-statistic > 10**: rule of thumb from Staiger & Stock (1997) for single endogenous variable. For multiple endogenous variables use Cragg-Donald or Kleibergen-Paap statistics.
- **Partial R²**: proportion of price variation explained by instruments alone (after partialling out exogenous regressors). Should be substantial.
- **Sanderson-Windmeijer F-statistic**: preferred for multiple endogenous regressors — tests each endogenous variable's exclusion separately.

In the BLP data: the demand instruments for prices typically achieve F ~ 20–40 in the first stage, indicating adequate strength.

### 7.3 Overidentification Tests

When the model is overidentified (more instruments than endogenous variables), test whether excess instruments are valid:

**Hansen J-test:** `J = N · Q_min ~ χ²(L-K)` where L = number of instruments, K = number of endogenous variables. Rejection suggests some instruments are invalid (correlated with `ξ_jt`).

**Sargan test (homoskedastic version)**: equivalent in large samples.

**Important caveat:** The J-test can fail to detect instrument invalidity when all instruments are invalid in the same direction (e.g., all correlated with unobserved quality). Rely on economic reasoning, not just the test.

### 7.4 Hausman Test for Price Endogeneity

Compare OLS and IV estimates. Under the null (price exogenous), both are consistent and should give similar estimates. Under the alternative (price endogenous), IV is consistent but OLS is not.

`H = (β̂_IV - β̂_OLS)'[Var(β̂_IV) - Var(β̂_OLS)]^{-1}(β̂_IV - β̂_OLS) ~ χ²(K)`

For automobile data this test strongly rejects price exogeneity, confirming the need for IV.

---

## 8. Price Elasticity Analysis

### 8.1 Formulas

**Plain logit own-price elasticity:**
`ε_jj = α · p_j · (s_j - 1) ≈ -α · p_j` (since s_j ≈ 0)

**Nested logit own-price elasticity:**
`ε_jj = (α / (1-σ)) · p_j · (1 - σ · s_{j|g} - (1-σ) · s_j)`

**BLP random-coefficients own-price elasticity:**
`ε_jj = -(p_j / s_j) · ∫ α_i · σ_ij · (1 - σ_ij) dF(ν_i, D_i)`

**Cross-price elasticity (BLP):**
`ε_jk = (p_k / s_j) · ∫ α_i · σ_ij · σ_ik dF(ν_i, D_i)`

Cross-price elasticities are **not symmetric**: ε_jk ≠ ε_kj because they depend on shares and prices of different products.

### 8.2 Typical Automotive Values from Published Papers

| Study | Dataset | Mean own-price elasticity |
|-------|---------|--------------------------|
| BLP (1995) | U.S. autos 1971-1990 | −3 to −6 |
| Goldberg (1995) | U.S. autos 1983-1987 | −3.28 (avg) |
| Verboven (1996) | European autos 1970-1999 | −4 to −8 |
| Knittel & Metaxoglou (2014) | U.S. autos | varies −1 to −10 by spec |

**Sanity checks:**
- All products should have `|ε_jj| > 1` (elastic demand) for a profit-maximizing firm — a firm would not price on the inelastic portion of demand
- As a rule: 1,494 of 2,217 BLP observations have inelastic demand under OLS; only 22 under IV. Use this ratio as a diagnostic.
- Luxury products should have smaller (in absolute value) elasticities than economy cars — the wealthy are less price-sensitive
- Cross-price elasticities should be larger between similar products (same segment, same origin) than dissimilar products

### 8.3 Elasticity Matrix

Compute the full J×J matrix of price derivatives per market:

```python
# In PyBLP:
elasticities = results.compute_elasticities()  # returns J×J matrix per market
# Rows: % change in shares; Columns: % change in prices
# Diagonal: own-price elasticities (should be negative)
# Off-diagonal: cross-price elasticities (should be positive — complements very rare in autos)
```

**What to look for in the matrix:**
- **Diagonal dominance**: each product's own response should exceed any single cross-response
- **Symmetry pattern** (not value): if product j substitutes strongly to k, k should substitute somewhat to j
- **Block structure**: products within the same segment/nest should have larger cross-elasticities than products in different segments

### 8.4 Diversion Ratios

The diversion ratio from product j to product k:
`D_jk = -(∂s_k/∂p_j) / (∂s_j/∂p_j) = ε_jk · s_j / (ε_jj · s_k)`

Measures what fraction of lost j-sales go to product k when j's price rises. Diversion to the outside good:
`D_j0 = 1 - Σ_{k≠j} D_jk`

**Antitrust application**: A merger between firms producing products j and k is potentially harmful if D_jk is large (consumers strongly substitute between the merging products). FTC/DOJ typically scrutinize mergers where D > 0.1.

In the BLP data, US domestic firms (firm IDs 1–5) should show higher mutual diversion ratios than diversion to Japanese imports, consistent with the market structure of the 1970s–80s.

---

## 9. Market Power and Markups

### 9.1 Bertrand-Nash First-Order Conditions

Under multi-product Bertrand competition, each firm maximizes profits over its portfolio. The first-order conditions:

`p - mc = -[∂s/∂p · O]^{-1} · s`

where `O` is the ownership matrix: `O_jk = 1` if products j and k are owned by the same firm, 0 otherwise. This can be written as:

`markup_j = p_j - mc_j = -Σ_{k: same firm} (∂s_k/∂p_j)^{-1} · s_k`

Or in matrix form: `markup = -[Ω ⊙ (∂s/∂p)]^{-1} · s`

where `Ω = O'` and `⊙` is element-wise product.

### 9.2 Recovering Marginal Costs

Given demand estimates (elasticities) and prices, recover implied marginal costs:
`mc_j = p_j - markup_j`

**Sanity check:** Marginal costs must be positive. Negative implied marginal costs indicate model misspecification — usually a wrong conduct assumption (e.g., assuming Bertrand when the firm is more competitive) or wrong instruments that produced an upward-biased price coefficient.

**In PyBLP:**
```python
costs = results.compute_costs()  # implied marginal costs from Bertrand FOCs
markups = results.compute_markups()  # = price - cost
```

### 9.3 Lerner Index

`L_j = (p_j - mc_j) / p_j = -1 / ε_jj` (for single-product firm under Bertrand)

For multi-product firms: `L_j = markup_j / p_j`. This is the price-cost margin as a fraction of price.

**Typical automotive values:**
- BLP (1995) report markups of approximately 10–30% for U.S. autos (Lerner index 0.10–0.30)
- Grieco et al. (2023, NBER WP 29013) find average U.S. auto Lerner index declined substantially from 1980–2016 due to import competition
- Very high Lerner indices (>40%) indicate market power concerns

### 9.4 Industry-Level Markup Evolution

Plot the average (sales-weighted) Lerner index over time:

```python
weighted_markup = (df['markups'] * df['shares']).groupby('market_ids').sum() / \
                   df.groupby('market_ids')['shares'].sum()
```

**Expected pattern for BLP data:** Import competition 1975–1985 should compress domestic markups. This narrative is directly testable and ties to the academic literature (Berry, Levinsohn & Pakes 1999 on Japanese VER).

---

## 10. Supply Side Estimation

### 10.1 Joint Demand-Supply Estimation

Instead of recovering marginal costs residually from demand, jointly estimate a cost function:

`log(mc_jt) = w_jt'γ + ω_jt`

where `w_jt` are cost shifters (materials, labor, R&D proxies) and `ω_jt` is an unobserved cost shock.

The supply-side moment condition: `E[ω_jt · Z_jt^S] = 0`

Supply-side instruments `Z^S`: characteristics of rival firms (shift demand but not own costs), or upstream input prices.

**Why bother?** Efficiency gains — joint estimation reduces variance of all estimates. Also: supply-side moments help pin down the conduct assumption. If the data reject Bertrand, one may test for collusion.

### 10.2 BLP Supply Specification

BLP (1995) use log-linear marginal costs:
`log(mc_jt) = log(hpwt) + log(air) + log(mpd) + log(space) + trend + ω_jt`

In PyBLP:
```python
product_formulations = (
    pyblp.Formulation('0 + prices + hpwt + air + mpd + space', absorb='C(market_ids)'),
    pyblp.Formulation('0 + hpwt + air + mpd + space'),          # nonlinear X2
    pyblp.Formulation('0 + log(hpwt) + air + log(mpd) + log(space) + trend'),  # supply X3
)
```

### 10.3 Testing the Conduct Assumption

The markup formula `p - mc = -[Ω ⊙ (∂s/∂p)]^{-1} · s` depends on the ownership matrix `Ω`. Researchers test:
- **Bertrand competition**: `Ω_jk = 1` iff same firm
- **Collusion**: `Ω_jk = 1` for all j, k in the industry
- **Cournot**: different FOC structure

A model with incorrect conduct produces systematically negative marginal costs for some products. Formal conduct tests use auxiliary regressions (Bresnahan 1982) or the "testing firm conduct" framework of Berry & Haile (2014).

---

## 11. Counterfactual Analysis

### 11.1 Merger Simulation

**Goal:** What happens to prices and welfare if firm A and firm B merge?

**Algorithm:**
1. Estimate demand and costs.
2. Change ownership matrix: set `Ω_jk = 1` for all products of the merged entity.
3. Solve new Bertrand equilibrium prices by iterating the ζ-markup contraction:
   `p^{n+1} = mc + ζ(p^n)` where `ζ_j = -Σ_k [∂s_k/∂p_j · Ω_{jk}]^{-1} · s_k(p^n)`
4. Compare pre- and post-merger prices, quantities, profits, consumer surplus.

**Welfare calculation:**

Consumer surplus per agent: `CS_i = (1/α_i) · log(1 + Σ_j exp(δ_j + μ_ij))`

Aggregate: `ΔCS = M · ∫ [CS_i(post) - CS_i(pre)] dF(ν_i, D_i)`

where M is market size (number of potential buyers).

**PyBLP implementation:**
```python
# Merger simulation
merger_results = results.compute_prices(
    firm_ids=merged_firm_ids,   # new ownership structure
    costs=costs
)
delta_cs = results.compute_consumer_surpluses() - pre_merger_cs
```

**Two approaches:**
1. **Approximate merger**: assume shares and derivatives unchanged; use current Bertrand markup formula with new ownership. Fast, first-order approximation. Used in Hausman, Leonard & Zona (1994).
2. **Full simulation**: iterate to new equilibrium. Required for realistic price changes. Can differ substantially from approximate when merging parties have large shares.

### 11.2 Policy Counterfactuals — Automotive Specific

**Fuel economy standards (CAFE):** Goldberg (1995, 1998) examined the welfare effects of CAFE regulations. Methodologically: add a constraint to the pricing problem forcing average fleet fuel economy above a threshold. This changes which products firms offer and how they price.

**Import tariffs / Voluntary Export Restraints:** BLP (1999) study Japanese VER directly. Counterfactual: remove the VER, solve new equilibrium with Japanese imports unconstrained. Key finding: VER cost U.S. consumers ~$10,000/car (1983 USD) in lost consumer surplus while benefiting domestic automakers.

**New product introduction:** What is the welfare gain from the introduction of the minivan in 1984? Calculate consumer surplus with and without the new segment. Berry, Levinsohn & Pakes (1996) pioneered this counterfactual.

**Electric vehicle subsidies:** Modern application — a $7,500 tax credit shifts the effective price of EVs down; re-solve equilibrium to find new market shares and CS.

### 11.3 Welfare Decomposition

Total welfare change = ΔCS + ΔPS − any externalities

`ΔCS = -M · α̅ · Σ_j s_j · Δp_j` (first-order approximation)

`ΔPS = Σ_j (Δp_j - Δmc_j) · s_j · M + Σ_j (p_j - mc_j) · Δs_j · M`

Producer surplus requires knowing both price and cost changes — hence the importance of the supply side.

---

## 12. Model Validation and Specification Tests

### 12.1 In-Sample Fit

**Predicted vs. actual market shares:**

```python
predicted_shares = results.compute_shares()
residuals = observed_shares - predicted_shares
# By construction: sum(residuals_per_market) = 0 (contraction mapping ensures this)
# Check RMSE, MAE, R² of predicted vs actual
```

The contraction mapping ensures exact in-sample fit for mean utilities (δ), but predicted shares depend on both δ and random coefficients. Non-trivial in-sample fit errors in shares indicate the random coefficient draws are not integrating accurately.

**Aggregate market share fit over time:** Plot total inside share (Σ_j s_jt) over time — should match observed total sales / market size closely.

### 12.2 Out-of-Sample Prediction

Use a time-based split:
- Estimate model on years 1971–1985
- Predict market shares for 1986–1990
- Compare to actual shares

This tests whether the model's substitution parameters are stable over time and can extrapolate through the Japanese import surge and the 1979 oil crisis.

**Key benchmark**: mean absolute percentage error (MAPE) for individual model shares. Publishing-grade papers typically achieve MAPE < 20% at the product-market level for automobiles.

### 12.3 Specification Tests for Nesting Structure

**Test: is σ significantly different from 0?** Use the t-statistic on the σ estimate. If not, the nested logit adds nothing over plain logit.

**Test: is σ significantly different from 1?** If σ → 1, consumers never switch nests, implying separate markets — this may call for separate demand systems per segment.

**Compare alternative nesting structures** using the GMM objective value (lower = better fit). For non-nested models use the Rivers-Vuong test or information criteria (BIC with appropriate adjustment for nested vs. non-nested comparison).

### 12.4 Testing Random Coefficient Significance

In the BLP model, each random coefficient standard deviation (diagonal of Σ) should be significantly positive. A zero standard deviation means no heterogeneity in that preference dimension — the coefficient is homogeneous and the random coefficient is unnecessary.

**Check:** t-statistics on all elements of Σ. In practice for automobiles, heterogeneity in price sensitivity (σ_price > 0) is almost always significant; heterogeneity in fuel efficiency preferences may be less robust.

### 12.5 GMM Fit Diagnostics

**GMM objective at optimum:** Should be close to zero relative to initial value. Compare first-step GMM objective (identity weighting) to second-step (optimal weighting) — a large improvement suggests significant information in the moment conditions.

**Gradient norm at optimum:** Should be very small (<10^{-6}). A large gradient norm indicates optimization failure.

**Condition numbers of weighting and covariance matrices:** Very large condition numbers (>10^8) indicate near-collinear instruments or numerical precision issues.

PyBLP reports all of these automatically at convergence.

---

## 13. Academic Applications of the BLP Dataset

### 13.1 The Original BLP (1995) Paper

**Berry, Levinsohn & Pakes (1995)**, *Econometrica* 63(4):841–890

- Introduced random-coefficients logit estimation from market-level shares
- Dataset: U.S. automobiles 1971–1990 (the bundled PyBLP dataset)
- Key results: mean own-price elasticities of −3 to −6; average markups 10–30%; heterogeneity in price sensitivity driven by income
- First application of contraction mapping + GMM for BLP estimation

### 13.2 BLP (1999) — Japanese Voluntary Export Restraints

**Berry, Levinsohn & Pakes (1999)**, *American Economic Review* 89(3):400–430

- Counterfactual: removed the 1981–1994 VER constraining Japanese auto exports
- Welfare cost: U.S. consumers lost ~$10B/year; domestic automakers captured most gains
- Methodology advance: introduced supply-side estimation jointly with demand

### 13.3 Goldberg (1995) — Micro Data and CAFE Standards

**Goldberg, Pinelopi K. (1995)**, *Econometrica* 63(4):891–951

- Used Consumer Expenditure Survey micro data to estimate demand with household demographics
- Nested logit by vehicle segment (subcompact/compact/mid-size/luxury/truck/van)
- Policy application: simulated the welfare effects of CAFE fuel economy standards
- Average estimated own-price elasticity: 3.28
- Key insight: CAFE benefits differ dramatically across income groups

### 13.4 Verboven (1996) — European Market Power

**Verboven, Frank (1996)**, *RAND Journal of Economics* 27(2):240–268

- Applied nested logit to Belgium/France/Germany/Italy/UK panel 1970–1999
- Documented substantial price discrimination across European countries
- Found evidence of pricing parallel to national market segmentation (pre-EU single market)

### 13.5 Knittel & Metaxoglou (2014) — Estimation Robustness

**Knittel & Metaxoglou (2014)**, *Review of Economics and Statistics*

- Re-estimated BLP using 50 different starting values and optimization algorithms
- Found parameter estimates vary dramatically across local optima — a critical cautionary finding
- Recommended: use multiple starting values, report confidence sets rather than point estimates
- Modern BLP practitioners use gradient-based optimizers (BFGS, trust-region) with many random starts

### 13.6 Conlon & Gortmaker (2020) — PyBLP Best Practices

**Conlon & Gortmaker (2020)**, *RAND Journal of Economics* 51(4):1261–1307

- Introduced PyBLP Python package with best practices for BLP estimation
- Key recommendations: use optimal instruments, cluster standard errors by product, use log-linear cost specifications, integrate demographics using empirical distribution from CPS
- Showed that differentiation instruments (Gandhi-Houde type) dominate traditional BLP instruments

### 13.7 Grieco, Murry, Yurukoglu (2023) — Market Power Evolution

**Grieco, Murry, Yurukoglu (2023)**, NBER WP 29013

- Estimated BLP model for U.S. auto market 1980–2016
- Documented long-run decline in average markups due to import competition
- Decomposed price changes into: cost changes, markup changes, and quality upgrading
- A template for how to use the BLP methodology for long-run industry analysis

---

## 14. PyBLP Workflow — Beyond Basic Estimation

### 14.1 Full Estimation Pipeline

```python
import pyblp
import pandas as pd

products = pd.read_csv(pyblp.data.BLP_PRODUCTS_LOCATION)
agents   = pd.read_csv(pyblp.data.BLP_AGENTS_LOCATION)

# Step 1: Specify formulations
product_formulations = (
    pyblp.Formulation('0 + prices + hpwt + air + mpd + space', absorb='C(market_ids)'),
    pyblp.Formulation('0 + hpwt + air + mpd + space'),
    pyblp.Formulation('0 + log(hpwt) + air + log(mpd) + log(space) + trend'),
)
agent_formulation = pyblp.Formulation('0 + I(1/income)')

# Step 2: Set up the problem
problem = pyblp.Problem(
    product_formulations, products,
    agent_formulation=agent_formulation, agent_data=agents
)

# Step 3: Set initial parameters and solve
sigma_0 = np.diag([3.612, 0, 4.628, 1.818, 1.050])  # initial nonlinear params
pi_0    = np.array([[0], [-43.501], [0], [0], [0]])    # demographics interactions
results = problem.solve(sigma=sigma_0, pi=pi_0, optimization=pyblp.Optimization('l-bfgs-b'))
```

### 14.2 Post-Estimation: All Available Outputs

```python
# Demand-side
elasticities  = results.compute_elasticities()          # J×J matrix per market
diversion     = results.compute_diversion_ratios()      # includes diversion to outside
agg_elast     = results.compute_aggregate_elasticities()# total market response

# Supply-side
costs         = results.compute_costs()                  # implied marginal costs
markups       = results.compute_markups()                # price - cost
profits       = results.compute_profits()                # per-product profits
hhi           = results.compute_hhi()                    # market concentration

# Welfare
cs            = results.compute_consumer_surpluses()    # integral over utility distribution

# Merger simulation
merged_results = results.compute_prices(
    firm_ids=new_ownership, costs=costs
)
```

### 14.3 Optimal Instruments

Replace hand-crafted BLP instruments with Chamberlain (1987) optimal instruments based on the estimated model:

```python
updated_problem = results.compute_optimal_instruments().to_problem()
results_opt = updated_problem.solve(sigma=results.sigma, pi=results.pi)
```

Optimal instruments improve efficiency (smaller standard errors) by using the expected Jacobian of moments as the instrument matrix. For the BLP automobile data, switching to optimal instruments typically reduces standard errors by 20–40%.

### 14.4 Bootstrapping for Inference

```python
bootstrap_results = results.bootstrap(
    draws=1000, seed=0
)
# Extract confidence intervals
ci_lower, ci_upper = np.percentile(
    [r.compute_consumer_surpluses() for r in bootstrap_results], [5, 95], axis=0
)
```

Parametric bootstrap (not re-estimation from data) is computationally tractable — draws from the parameter distribution and re-solves for equilibrium prices and welfare. Produces valid confidence intervals for non-linear post-estimation objects (elasticities, markups, merger price effects).

### 14.5 Incorporating Micro Moments

```python
micro_moments = pyblp.MicroMoment(
    name="mean income of buyers",
    value=observed_mean_income,         # from household survey
    parts=pyblp.MicroPart(
        name="buyer income",
        dataset=micro_dataset,
        compute_values=lambda A, X, Y, Z: A.demographics[:, 0]  # income
    )
)
results_micro = problem.solve(sigma=sigma_0, pi=pi_0,
                               micro_moments=[micro_moments])
```

Micro moments tie the structural model to observable moments in household data, dramatically sharpening identification of preference heterogeneity.

### 14.6 Problem Simulation for Model Testing

Before applying to real data, test the pipeline on simulated data where the true parameters are known:

```python
simulation = pyblp.Simulation(
    product_formulations, product_data=simulated_data,
    sigma=true_sigma, pi=true_pi, xi=true_xi
)
simulated_problem = simulation.replace_endogenous()  # solve for equilibrium prices/shares
recovered_results = simulated_problem.solve(sigma=initial_sigma, pi=initial_pi)
# Verify recovered_results.sigma ≈ true_sigma
```

This validates that the optimization finds the correct parameters and that the instruments are strong enough to identify them. A standard check before running on real data.

---

## Key Differences: Expert vs. Novice Analysis

| Aspect | Novice | Expert |
|--------|--------|--------|
| Starting point | Jump straight to BLP estimation | Logit OLS first, document endogeneity bias |
| Instruments | Use whatever is bundled | Test strength (F-stat), validity (J-test), consider differentiation IVs |
| Nesting | Arbitrary grouping | Theory-grounded, robustness across alternative nests |
| Convergence | Accept first local optimum | Multiple starts, report gradient norm, check condition numbers |
| Elasticities | Report mean elasticity | Full matrix, sanity checks, comparison to literature values |
| Marginal costs | Ignore | Recover from FOCs, check positivity, report distribution |
| Counterfactuals | None | Merger simulation, policy counterfactual, welfare decomposition |
| Inference | Point estimates only | Bootstrap confidence intervals for non-linear objects |
| Validation | In-sample R² | Out-of-sample, specification tests, IIA tests |
| Documentation | Table of coefficients | Full replication package, robustness appendix |

---

## Checklists

### Pre-Estimation Checklist
- [ ] Inside shares sum to < 1 in every market
- [ ] Outside good share is economically plausible (>5%)
- [ ] Market size definition documented and sensitivity-tested
- [ ] Prices regressed on characteristics — residuals examined
- [ ] Hausman test for price endogeneity
- [ ] VIF check for multicollinearity
- [ ] Structural breaks identified and flagged (oil crises, new entrants)
- [ ] Instrument first-stage F-statistic > 10

### Estimation Checklist
- [ ] OLS logit baseline reported (for comparison)
- [ ] IIA test run and documented
- [ ] Nested logit with theory-motivated nests
- [ ] σ ∈ (0,1) verified
- [ ] BLP problem set up with correct formulations
- [ ] Multiple starting values tried (≥ 10)
- [ ] Convergence criteria tightened to < 10^{-12}
- [ ] Two-step GMM with optimal weighting
- [ ] Standard errors clustered by product
- [ ] Hansen J-test for overidentification

### Post-Estimation Checklist
- [ ] Elasticity matrix computed; diagonal negative, off-diagonal positive
- [ ] Own-price elasticities in range |3–8| for automobiles
- [ ] Marginal costs positive for all products
- [ ] Lerner indices in plausible range (10–40%)
- [ ] At least one counterfactual (merger or policy)
- [ ] Consumer surplus computed
- [ ] Bootstrap confidence intervals for key objects
- [ ] Out-of-sample prediction evaluated

---

## Sources

- PyBLP documentation: https://pyblp.readthedocs.io/en/stable/
- PyBLP background (full methodology): https://pyblp.readthedocs.io/en/stable/background.html
- PyBLP post-estimation tutorial: https://pyblp.readthedocs.io/en/stable/_notebooks/tutorial/post_estimation.html
- PyBLP nested logit tutorial: https://pyblp.readthedocs.io/en/stable/_notebooks/tutorial/logit_nested.html
- Conlon & Gortmaker best practices paper: https://chrisconlon.github.io/site/pyblp.pdf
- Nevo (2000) practitioner guide: https://pages.stern.nyu.edu/~wgreene/Econometrics/Nevo-BLP.pdf
- Berry & Haile (2021) foundations: https://www.nber.org/system/files/working_papers/w29305/w29305.pdf
- Kawaguchi EmpiricalIO notes: https://kohei-kawaguchi.github.io/EmpiricalIO/demand.html
- Grieco et al. market power evolution: https://web.stanford.edu/~ayurukog/CarMarkupsMarch2023.pdf
- Goldberg (1995) semantic scholar: https://www.semanticscholar.org/paper/Product-Differentiation-and-Oligopoly-in-Markets:-Goldberg/c7a77272f730c66ccdcaa3cc615b3655a0bc88d0
- Merger simulation overview: https://www.ee-mc.com/tools/merger-simulation-models.html
- Conlon diversion ratio paper: https://chrisconlon.github.io/site/diversion.pdf
- Mark Ponder BLP tutorial: https://mark-ponder.com/tutorials/static-discrete-choice-models/random-coefficients-blp/
