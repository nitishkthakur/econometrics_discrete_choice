# FORD — Competition Analysis (Competitive Landscape Tool)
## Ford Motor Company | Global Data Insight & Analytics (GDIA)

---

## Project Overview

Built a **Competitive Landscape identification system** for Ford vehicles — an ML-based tool that, given a Ford vehicle (at entity/trim level), ranks competitor vehicles from a database by similarity. The system accounts for product features, pricing, and customer preferences to identify the most relevant competitors at a granular entity level.

The project was delivered in 3 phases (Q4 2022 → Q2 2023 GtM) and resulted in a deployed internal tool called **VaMOS**, democratised for business use in Phase 3.

---

## Objective

1. Identify competition based on objective and subjective features at **entity (trim) level**
2. Perform **entity matching** for newly introduced vehicles (applicable to CM 2.0 for entity-level imputation on unseen data)

**Output**: A ranked list of competitor vehicles (nearest entities) for any queried Ford vehicle

---

## Data Sources

Three data sources were merged:

| Source | Contents |
|--------|----------|
| **DATAFORCE** | Overall segment sales share, fuel type share — comprehensive list of competitor registrations from 2015 |
| **Mystery Shopping (KANTAR)** | List price, vehicle specification, fuel type, Competitor VM — sourced from KANTAR; limited contract information from competitors from 2016 |
| **Business Input** | Relevant list of competitors, current market insights |

Pre-processing: files merged month-wise using Python, validated against Ford list price data. Data range: Jan 2016 – Feb 2022. Final merge at Nameplate Level and Fuel-Type level via Alteryx.

A fourth source (**NSE / FCE Data**) was used for tracking competitor landscape (limited info on competition monthly payment, term, APR, cash incentives, vehicle configuration).

---

## Methodology

### Algorithm: Nearest Neighbour Approach

Competitors are identified using a **nearest neighbour** algorithm. For each Ford entity queried, the tool computes a distance to every vehicle in the competitor database and ranks them from closest (most similar) to furthest (least similar).

**Features used:**

| Feature | Type |
|---------|------|
| HP (Horsepower) | Continuous |
| List Price | Continuous |
| Fuel Type | Continuous or Categorical (configurable) |
| Segment | Continuous or Categorical (configurable) |
| WD (Wheel Drive) | Categorical — One Hot Encoded |
| Door count | Categorical — One Hot Encoded |
| Transmission | Categorical — One Hot Encoded |

Feature scaling performed on both continuous and categorical features. Users have flexibility to manually select features to include based on domain knowledge.

### Feature Weighting: AHP (Analytic Hierarchy Process)

Feature importance is configurable in two modes:

- **Relative importance (AHP)**: Pairwise comparison matrix — e.g., all variables set 3× more important than List Price
- **Absolute importance**: All variables equally weighted (default)

Users can set custom weights per feature for each Ford entity query — e.g., for a performance vehicle like Mach E GT, a user can set greater importance for HP and lesser for price.

### Competitor identification granularity

- Identification done at **Trim/Entity level** (not just nameplate)
- Output includes a ranked competitor list with distance scores for each Ford entity queried

---

## Output Examples

### Mach E GT — Top Competitors (ranked by distance)
Tesla Model Y (Performance), Tesla Model Y (Long Range), Jaguar I-Pace EV400, Mercedes EQC 400 AMG, Tesla Model 3 Performance AWD, BMW iX, Audi e-tron Sport 50 quattro

### Mach E Extended Range RWD — Top Competitors
Tesla Model 3 Standard Range RWD, Polestar 2 Long Range, Audi Q3 S-Line PHEV, Kia EV6, Skoda ENYAQ EV

### Kuga — Top Competitors
Hyundai Tucson, Volkswagen Tiguan, Nissan Qashqai, Peugeot 3008, Opel Grandland, Kia Sportage, Toyota CH-R

The output table shows: HP, List Price, WD, Fuel, Doors, Transmission, Segment, and Distance score for each competitor.

---

## Phased Delivery

| Phase | Timeline | Deliverables |
|-------|----------|-------------|
| Phase 1 | Q4 2022 GtM | Scope data sources, cleanse and merge. Match competitor vehicles with Ford vehicles at cluster level for private retail conquest customers |
| Phase 2 | Q1 2023 GtM | Model to predict volume as a result of competitor actions. Tool to enable user to set instalment walk: match competitor offerings and calculate impact on per-unit basis for Subvention and Contribution Margin. Improve accuracy of forecasted monthly payment–contribution margin–volume |
| Phase 3 | Q2 2023 GtM | Deploy and democratise via **VaMOS** for business use |

---

## Completed vs In Progress (at time of presentation)

**Completed:**
- Generate Competitive Landscape Report identifying competitors for Ford vehicles at entity level, accounting for product similarity, pricing, and customer preference
- Add Equalized Monthly Payment to the Competition Landscape report

**Work In Progress:**
- Dynamic Simulation of Competitive Landscape — user provides pricing and financing information for a Ford entity; tool generates the updated competitor landscape
- Add aggregated competition information at nameplate level for optimal positioning from a pricing point of view

**Future Steps:**
- Incorporate competitor environment changes (pricing, supply, VM) into forecast
- Vehicle aesthetics/styling to be considered for competitor identification (IHS Data Exploration — vehicle size information)
- Transparent handling of TPI

---

## Markets Covered

UK, France, Spain, Italy, Germany (retail channel)

---

## Tech Stack

Python (data merging, pre-processing), Alteryx (data pipeline), DATAFORCE, KANTAR/Mystery Shopping data, IHS data

---

## Resume-Relevance Notes

- **Algorithm**: Nearest Neighbour with AHP-based feature weighting — not just "ML competitor identification"
- **Scale**: Entity/trim level across 5 European markets; database covers competitor registrations from 2015
- **Web app**: Built internal tool (VaMOS) deployed for business use in Q2 2023
- **Novelty**: Feature type flexibility (continuous/categorical treatment of fuel type and segment), configurable weights per query
- Slides marked CONFIDENTIAL — do not reference specific competitor pricing numbers or internal data volumes publicly
