# Dataset Options for Replicating the Ford Demand Forecasting & Optimization Project

## Project Context

The original Ford project (BSPA / UK market, Ford Fiesta) used:
- **Generalized Nested Logit (GNL)** for trim-level demand forecasting across 6 trims × 4 customer segments
- **Nested Logit** booking model for 8 financial instruments across 3 nests
- **Hierarchical structure**: Nameplate → Trim/Entity → Finance Option → Term Length
- **Key variables**: List price, discount, APR, deposit allowance, other cash, privilege (employee discount), trade-in equity, dealer incentives, material costs
- **Optimization**: SQP with 864 decision variables, profit maximization subject to APR ordering constraints
- **Frequency**: Monthly time series

No public dataset replicates all of this simultaneously. The ranking below is ordered by how closely each dataset lets you reconstruct the methodology, domain, and structure — and by how directly published econometric literature has applied analogous methods to it.

---

## Rankings

### Rank 1 — BLP Automobile Dataset (via PyBLP)

**Best match for: methodology fidelity, automotive domain, discrete choice tooling**

| Attribute | Detail |
|-----------|--------|
| **Source** | `pip install pyblp` → `pyblp.data.BLP_PRODUCTS_LOCATION` |
| **URL** | https://pyblp.readthedocs.io / https://github.com/jeffgortmaker/pyblp |
| **Access** | Completely free, no registration. Installs with the Python package. |
| **Domain** | Automotive (U.S. passenger car market) |
| **Coverage** | 1971–1990, 20 annual markets, 2,217 product-market observations across 997 distinct models |
| **Hierarchy** | Firm → Nameplate (base-model level only — **no trim hierarchy**) |
| **Variables** | `market_ids`, `car_ids`, `firm_ids`, `shares`, `prices` (list retail, 1983 USD), `hpwt` (HP/weight), `air` (A/C dummy), `mpd`, `mpg`, `space`, `trend`, `region`, pre-computed BLP instruments |
| **Financing variables** | None (no APR, deposit allowance, or term lengths) |
| **Temporal frequency** | Annual |
| **Academic citations** | Berry, Levinsohn & Pakes (1995), *Econometrica* 63(4):841–890. The canonical paper for BLP random-coefficients discrete choice demand estimation. Thousands of citations. |
| **Methods applied** | Multinomial logit, Nested Logit, Generalized Nested Logit, BLP random-coefficients logit, SQP pricing optimization — the exact methods used in the Ford project |
| **Competition/Leaderboard** | None, but the PyBLP package is the standard replication benchmark against which new implementations are evaluated |
| **Adversarial verification** | 3-0 confirmed on all key claims |

**Why Rank 1:** This is the *canonical* dataset for automotive discrete choice demand estimation in economics. Every method used in the Ford project (NL, GNL, BLP, SQP pricing) has been applied to this data and documented in published papers. PyBLP's Python package makes it immediately usable with reproducible tutorials. The main limitations are: (a) no trim-level hierarchy (nameplate only), (b) no financing variables, (c) annual not monthly frequency, (d) U.S. not UK market, (e) ends in 1990. Despite these, it is the best starting point for demonstrating that you can run the methodology end-to-end.

**How to adapt:** Use `firm_ids` as the nest grouping. Construct a pseudo-trim hierarchy by treating each model-year observation as a variant and grouping by manufacturer segment (e.g., compact/mid-size/luxury). SQP profit maximization is directly illustrated in the PyBLP supply-side tutorial.

---

### Rank 2 — DVM-CAR Dataset (Deep Visual Marketing Car Dataset)

**Best match for: automotive trim-level hierarchy, UK market, structural similarity to Ford data**

| Attribute | Detail |
|-----------|--------|
| **Source** | https://deepvisualmarketing.github.io/ |
| **URL** | https://deepvisualmarketing.github.io/ · Paper: https://arxiv.org/abs/2109.00881 |
| **Access** | CC BY-NC license, persistent DOI. Free download — **but sales volume CSV availability was NOT confirmed** (see caveats below). |
| **Domain** | Automotive (UK new car registrations) |
| **Coverage** | ~10 years, 899 car models, UK/GB market |
| **Hierarchy** | Brand → Model → Trim/Entity (confirmed 3-level hierarchy; 330,000 trim-level records) |
| **Variables** | Six relational tables: Basic (nameplate metadata), Sales (volume — availability unconfirmed), Price (list price), Trim (selling price, engine type, fuel type, engine size), Ad (advertising images), Image (vehicle photos) |
| **Financing variables** | None (no APR, deposit, term lengths) |
| **Temporal frequency** | Monthly (if Sales table is accessible) |
| **Academic citations** | Presented at IEEE Big Data 2022. Primarily used for computer vision / marketing image analysis, but the structured tables support econometric analysis. |
| **Methods applied** | No confirmed published application of discrete choice models to this specific dataset. You would be among the first. |
| **Competition/Leaderboard** | None |
| **Adversarial verification** | Trim-level price/attribute claim: 3-0 confirmed. Sales data CSV download claim: **0-3 refuted** (unresolved) |

**Why Rank 2:** This is the only confirmed public dataset with a genuine nameplate → trim product hierarchy for the UK automotive market — the exact market in the Ford project. The Trim table has selling prices at trim level, engine type, and engine size. If the Sales table is downloadable as CSV, this becomes the single best substitute dataset because you get: UK market, trim-level prices, product hierarchy, and monthly registration volumes.

**Critical caveat:** Adversarial verification could not confirm that sales volume data is downloadable as CSV files. You must manually visit https://deepvisualmarketing.github.io/ and attempt to download the Sales table before deciding to invest in this dataset. The dataset paper (arXiv:2109.00881) mentions a "Sales" table but its download accessibility was not verified.

---

### Rank 3 — Verboven European Car Market Dataset

**Best match for: European automotive market, multi-country panel, econometric research lineage**

| Attribute | Detail |
|-----------|--------|
| **Source** | https://sites.google.com/site/frankverbo/data-and-software/data-set-on-the-european-car-market |
| **Access** | Reportedly free (Stata + Excel formats via Google Drive link on author's page) — **access not fully verified; check URL directly** |
| **Domain** | Automotive (European car market) |
| **Coverage** | 1970–1999, 5 countries: Belgium, France, Germany, Italy, UK |
| **Hierarchy** | Manufacturer → Model (nameplate level; no trim hierarchy) |
| **Variables** | Model-level prices, sales volumes, market shares, physical characteristics (HP, weight, size), country-year panels |
| **Financing variables** | None confirmed |
| **Temporal frequency** | Annual |
| **Academic citations** | Verboven, F. (1996), "International Price Discrimination in the European Car Market," *RAND Journal of Economics* 27(2):240–268. Goldberg & Verboven (2001), "The Evolution of Price Dispersion in the European Car Market," *Review of Economic Studies* 68(4):811–848. |
| **Methods applied** | Nested Logit, BLP demand estimation applied extensively in published papers using this data |
| **Competition/Leaderboard** | None |
| **Adversarial verification** | 1-0 (insufficient votes — 2 verifiers failed mid-response). Treat as unverified. |

**Why Rank 3:** Verboven (1996) is among the earliest and most-cited applications of nested logit to automotive demand in a *European* setting with UK included — directly analogous to the Ford UK project. The multi-country panel replicates the France/Germany/Spain/Italy/UK market structure that Ford operated in. The main gaps are: no trim hierarchy (model-level only), annual frequency, no financing variables. But the econometric tradition here is directly transferable to your nested logit and pricing optimization work.

**Action required:** Visit the Google Sites URL and verify the download links are still live before committing.

---

### Rank 4 — Nevo (2000) Cereal Dataset (via PyBLP / Mixtape-Sessions)

**Best match for: nested logit methodology benchmark, free tooling, fully reproducible tutorials**

| Attribute | Detail |
|-----------|--------|
| **Source** | `pyblp.data.NEVO_PRODUCTS_LOCATION` (after `pip install pyblp`) OR https://github.com/Mixtape-Sessions/Demand-Estimation |
| **Access** | Completely free, no registration |
| **Domain** | CPG — breakfast cereals |
| **Coverage** | 94 markets (47 U.S. cities × 2 quarters, 1988), 2,256 product-market observations |
| **Hierarchy** | Firm → Brand (2-level: 5 firms, 24 brands) |
| **Variables** | `market_ids`, `city_ids`, `quarter`, `product_ids`, `firm_ids`, `brand_ids`, `shares`, `prices`, `sugar`, `mushy`, 20 pre-computed demand instruments |
| **Financing variables** | Not applicable (CPG) |
| **Temporal frequency** | Quarterly |
| **Academic citations** | Nevo, A. (2000), "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand," *Journal of Economics & Management Strategy* 9(4):513–548. One of the most-cited applied econometrics papers ever. |
| **Methods applied** | Nested Logit, BLP random-coefficients logit, profit maximization — the exact modeling framework used in the Ford project |
| **Competition/Leaderboard** | None |
| **Adversarial verification** | 3-0 confirmed on all key claims (structure, variables, download) |

**Why Rank 4:** Although this is cereal not cars, it is the standard teaching and benchmarking dataset for the entire BLP/nested logit methodology ecosystem. Every discrete choice method in the Ford project has a direct tutorial implementation on this data. The firm→brand hierarchy is structurally analogous to manufacturer→nameplate, and groups within firms provide nesting analogous to Ford's trim groupings. The Mixtape-Sessions GitHub has hands-on exercises walking through exactly the estimation pipeline you need.

**How to adapt:** Treat firms as manufacturers, brands as nameplates, and within-firm brand groupings as the nesting structure for GNL. The PyBLP tutorials for nested logit and random-coefficients logit use this data explicitly, so you can follow them step-by-step.

---

### Rank 5 — Dominick's Finer Foods Dataset

**Best match for: free CPG scanner data, product hierarchy, weekly time series, academic replication tradition**

| Attribute | Detail |
|-----------|--------|
| **Source** | https://www.chicagobooth.edu/research/kilts/datasets/dominicks |
| **Access** | Free download, no fee, only acknowledgment required |
| **Domain** | CPG — grocery/supermarket (Dominick's chain, Chicago area) |
| **Coverage** | ~9 years of weekly store-level scanner data (exact scope: verify with Kilts Center data manual; commonly cited figures were adversarially refuted) |
| **Hierarchy** | Department → Category → Brand → SKU/UPC (3–4 levels) |
| **Variables** | `PRICE`, `MOVE` (units sold), `PROFIT` (margin), `DEAL` (promotion code) — in weekly movement files per category |
| **Financing variables** | Not applicable |
| **Temporal frequency** | Weekly |
| **Academic citations** | Numerous published papers in marketing and IO economics. Used in Chevalier, Kashyap & Rossi (2003) *AER*; multiple discrete choice demand estimation papers |
| **Methods applied** | Nested logit and probit demand models applied in published literature. Price instruments constructed from wholesale cost data. |
| **Competition/Leaderboard** | None |
| **Adversarial verification** | Download access and movement file variable list: 3-0 confirmed. Scope/size statistics: 1-2 refuted (treat commonly cited numbers as unverified) |

**Why Rank 5:** Freely downloadable without institutional barriers, with a category→brand→SKU structure that maps cleanly onto nameplate→variant nesting. Weekly frequency is even richer than the Ford project's monthly data. Price variation within categories supports the construction of BLP-style instruments. The main limitation is that it's CPG (not automotive) and has no analog to financing variables.

---

### Rank 6 — M5 Forecasting Competition (Walmart Retail)

**Best match for: hierarchical demand forecasting, active competition with leaderboard, rich product/store structure**

| Attribute | Detail |
|-----------|--------|
| **Source** | https://www.kaggle.com/c/m5-forecasting-accuracy |
| **Access** | Free (Kaggle account required) |
| **Domain** | Retail — Walmart (food, hobby, household products) |
| **Coverage** | 1,941 daily time series per store-department × 10 stores across 3 states (California, Texas, Wisconsin). ~5.4 years of daily sales. Aggregates to 42,840 hierarchical series. |
| **Hierarchy** | Item → Product Sub-category → Product Category → Department → Store → State → National (7 levels) |
| **Variables** | Daily unit sales per item, store, date. Separate `sell_prices.csv` provides weekly item prices per store. Calendar features (events, SNAP days). |
| **Financing variables** | None |
| **Temporal frequency** | Daily (aggregatable to weekly/monthly) |
| **Academic citations** | Makridakis, Spiliotis & Assimakopoulos (2022), *International Journal of Forecasting*. Benchmark paper for hierarchical time series forecasting. |
| **Methods applied** | Predominantly ML-based (LightGBM, DeepAR, etc.) for the competition; structural demand models less explored on this data |
| **Competition/Leaderboard** | Yes — active Kaggle competition leaderboard. Two tracks: accuracy (WRMSSE metric) and uncertainty (quantile score). |
| **Adversarial verification** | Confirmed via direct Kaggle dataset inspection (not adversarially verified in this research run, but claims are well-established) |

**Why Rank 6:** The M5 dataset has the deepest product hierarchy of any free dataset here (7 levels) and is the benchmark for hierarchical time series forecasting. The price data file enables price-sensitivity modeling. The leaderboard makes it useful for benchmarking your approach against others. However, the dominant methods are ML-based (not structural discrete choice), so applying GNL/nested logit here would be novel and methodologically more complex — there's no natural "nesting structure" because items span unrelated product categories. This is the best choice if your primary goal is hierarchical demand forecasting performance; less so if you want to replicate the structural discrete choice + optimization pipeline.

---

### Rank 7 — IRI Marketing Data Set

**Best match for: rich CPG hierarchy + household panel data enabling individual-level discrete choice estimation**

| Attribute | Detail |
|-----------|--------|
| **Source** | Via INFORMS Society for Marketing Science (fee-based or scholarship) |
| **URL** | https://pubsonline.informs.org/doi/10.1287/mksc.1080.0450 |
| **Access** | **NOT free** — requires payment of IRI fees (INFORMS scholarships available for academics). Delivered on USB drive. Post-2023 IRI/Circana merger may affect access pathway — contact INFORMS/Circana directly. |
| **Domain** | CPG — 30 product categories |
| **Coverage** | 5 years of store-level weekly data × 47 U.S. markets. Household panel data in 2 markets. |
| **Hierarchy** | Category → Brand → UPC/SKU |
| **Variables** | Store-level: units, revenue, price, promotions, ACV. Household panel: purchase occasion, quantity, price paid, promotional context. |
| **Financing variables** | Not applicable |
| **Temporal frequency** | Weekly |
| **Academic citations** | Bronnenberg, Kruger & Mela (2008), *Marketing Science* 27(4):745–748. Extensively used in applied IO and marketing science papers. |
| **Methods applied** | Nested logit, random-coefficients logit, household-level discrete choice models — all published and peer-reviewed. |
| **Competition/Leaderboard** | None |
| **Adversarial verification** | Coverage/panel data claims: 3-0 confirmed. Fee-based access: 2-1 confirmed. |

**Why Rank 7:** The household panel component is particularly valuable — it lets you estimate individual-level discrete choice models analogous to the Ford project's Retail/Employee × Conquest/Renewal segmentation. The 30-category × 47-market panel is the richest available CPG dataset. Ranked below free alternatives because of the cost barrier.

---

### Rank 8 — NielsenIQ Retail Scanner Data (Chicago Booth Kilts Center)

**Best match for: massive scale, deep CPG product hierarchy, proven in IO/marketing econometrics**

| Attribute | Detail |
|-----------|--------|
| **Source** | https://www.chicagobooth.edu/research/kilts/research-data/nielseniq |
| **Access** | **NOT freely downloadable** — requires an institutional subscription contract between researcher's university and Chicago Booth. Accessed via Kilts Apply Portal, delivered via Globus. |
| **Domain** | CPG — retail grocery and general merchandise |
| **Coverage** | 2.6–4.5 million UPCs, national U.S. coverage, multiple years |
| **Hierarchy** | UPC → Product Category → Product Group → Department (4 levels; 1,100 categories × 125 groups × 10 departments) |
| **Variables** | Weekly store-level: units, price, price multiplier, baseline units, baseline price, feature indicator, display indicator |
| **Financing variables** | None |
| **Temporal frequency** | Weekly |
| **Academic citations** | Widely used in top-tier IO economics and marketing science journals. |
| **Methods applied** | Nested logit, BLP random-coefficients logit, demand estimation at category and brand level — published in numerous peer-reviewed papers |
| **Competition/Leaderboard** | None |
| **Adversarial verification** | Variable list and access terms: 3-0 confirmed. Hierarchy statistics: 2-1 confirmed. |

**Why Rank 8:** This is the scale-up version of Dominick's — the hierarchy goes 4 levels deep with millions of UPCs, which gives you a richer nesting structure for GNL. If you have institutional access through your university, this is the most powerful free-to-use-with-access CPG dataset. Ranked below IRI only because IRI additionally provides household panel data for individual-level estimation.

---

### Rank 9 — Stallion Beverage Demand Dataset (Kaggle)

**Best match for: monthly SKU-level demand with agency (distributor) hierarchy; direct ML competition context**

| Attribute | Detail |
|-----------|--------|
| **Source** | https://www.kaggle.com/datasets/utathya/future-volume-prediction |
| **Access** | Free (Kaggle account required) |
| **Domain** | Beverage industry (alcoholic beverage distribution) |
| **Coverage** | Monthly agency-SKU sales, multiple years. Includes ~900 agencies and ~65 SKUs. |
| **Hierarchy** | Agency (distributor/retailer) → SKU (2-level: analogous to store → product variant) |
| **Variables** | Monthly volume sold, agency ID, SKU ID. Some volume/price-related features. |
| **Financing variables** | None |
| **Temporal frequency** | Monthly — directly matches the Ford project's forecasting horizon |
| **Academic citations** | None — Kaggle dataset. Referenced in PyTorch Forecasting tutorials by Jan Beitner (temporal fusion transformer). |
| **Methods applied** | ML-based (TFT, LSTM, LightGBM) in tutorials; no discrete choice application published. |
| **Competition/Leaderboard** | Associated with PyTorch Forecasting library benchmarks; no formal Kaggle leaderboard for this dataset. |
| **Adversarial verification** | Claims confirmed from Kaggle description; not adversarially verified in this research run. |

**Why Rank 9:** The monthly frequency is the strongest match to the Ford forecasting cycle among all Kaggle datasets here. The 2-level Agency → SKU hierarchy maps cleanly to a simplified version of the Ford structure (e.g., dealer region → trim). The main weaknesses are: no pricing data, no discrete choice literature on this data, and only 2 hierarchy levels. Best used if you want to first demonstrate time-series forecasting with hierarchy before layering in the discrete choice components.

---

## Summary Scorecard

| Rank | Dataset | Domain | Trim Hierarchy | Pricing | Financing | Discrete Choice Applied | Free Access | Monthly/Weekly |
|------|---------|--------|----------------|---------|-----------|-------------------------|-------------|----------------|
| 1 | BLP Automobile (PyBLP) | Automotive | Nameplate only | Yes | No | Yes (canonical) | Yes | Annual |
| 2 | DVM-CAR | Automotive (UK) | Yes (confirmed) | Yes | No | Not confirmed | Yes* | Monthly* |
| 3 | Verboven European Cars | Automotive (EU/UK) | Nameplate only | Yes | No | Yes (NL/BLP) | Unverified | Annual |
| 4 | Nevo Cereal (PyBLP) | CPG | 2-level | Yes | No | Yes (canonical) | Yes | Quarterly |
| 5 | Dominick's Finer Foods | CPG | 3–4 level | Yes | No | Yes | Yes | Weekly |
| 6 | M5 Forecasting (Walmart) | Retail | 7-level | Yes | No | No | Yes | Daily |
| 7 | IRI Marketing Data | CPG | 3-level | Yes | No | Yes | Paid | Weekly |
| 8 | NielsenIQ Retail Scanner | CPG | 4-level | Yes | No | Yes | Institutional | Weekly |
| 9 | Stallion Beverage | Beverages | 2-level | Partial | No | No | Yes | Monthly |

*DVM-CAR sales CSV accessibility was not confirmed by adversarial verification — check manually.

---

## Recommended Path Forward

**For methodology replication (discrete choice + optimization):** Start with the **BLP dataset (Rank 1)** using PyBLP. All the estimation and optimization machinery is documented, and you can demonstrate GNL, nested logit, BLP random-coefficients, and SQP pricing in one coherent pipeline. Supplement with **Nevo cereal (Rank 4)** for the nested logit-specific tutorials.

**For structural similarity to the Ford project:** Investigate **DVM-CAR (Rank 2)** first — manually verify the Sales table download. If sales data is available, this is the closest possible public substitute: UK market, trim-level hierarchy, monthly data. Pair with the BLP methodology to get both structure and methods aligned.

**For a competition/leaderboard element:** Use **M5 (Rank 6)** as the benchmarking layer. You can build the hierarchical forecasting component on M5 (to get a public score) and the discrete choice component on BLP/Nevo (to demonstrate the structural modeling). The two projects can be framed as complementary.

---

## Key Gap: Financing Variables

No verified public dataset contains automotive financing variables (APR, deposit allowance, term lengths, trade-in equity). This dimension of the Ford project has no public analog. The options are:

1. **Omit the financing layer** and replicate only the GNL demand model and SQP profit optimization based on list price and incentives.
2. **Simulate financing variables** using reasonable parameter ranges and demonstrate the optimization structure, citing the Ford project's methodology as the source of the parameter bounds.
3. **Use RDC/confidential data** — if you have access to a research data center with CFPB auto loan data or NHTSA/IHS automotive data, these sometimes include financing terms at the model level.

---

## Sources

- PyBLP documentation: https://pyblp.readthedocs.io/en/stable/
- Berry, Levinsohn & Pakes (1995), *Econometrica* 63(4):841–890
- Nevo (2000), *JEMS* 9(4):513–548
- Bronnenberg, Kruger & Mela (2008), *Marketing Science* 27(4):745–748
- DVM-CAR paper: https://arxiv.org/abs/2109.00881
- Verboven (1996), *RAND Journal of Economics* 27(2):240–268
- Mixtape-Sessions Demand Estimation GitHub: https://github.com/Mixtape-Sessions/Demand-Estimation
- Chicago Booth Kilts Center: https://www.chicagobooth.edu/research/kilts
- M5 Forecasting: https://www.kaggle.com/c/m5-forecasting-accuracy
- Stallion Dataset: https://www.kaggle.com/datasets/utathya/future-volume-prediction
