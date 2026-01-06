# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Affordable Housing Viability Analyzer - A Streamlit app for analyzing multifamily housing project viability and income affordability in British Columbia.

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
app.py                          # Main Streamlit entry point (two tabs)
models/
  project_viability.py          # Financial model (DSCR, cash-on-cash, viability grid)
  income_distribution.py        # BC income distribution (log-normal model)
  listing.py                    # MultifamilyListing Pydantic model
  param_mapper.py               # Maps listings to viability params + analyze_portfolio()
visualizations/
  income_band.py                # Bell curve + affordability band (animated)
  viability_space.py            # 2D heatmap + portfolio scatter
data/
  synthetic_listings.csv        # Sample BC multifamily listings (50 properties, 20+ units)
  market_rents.csv              # Market rent per sqft by BC city (for missing cap_rate)
```

## App Structure

**Tab 1: Portfolio Overview**
- Summary metrics (total, viable count, avg DSCR, price range)
- Listings table with viability status
- Scatter plot showing all listings on viability space (green=viable, red=not viable)

**Tab 2: Project Deep-Dive**
- Select a specific listing from dropdown
- Detailed metrics (DSCR, cash-on-cash, margin)
- Position highlighted on viability space
- Full financial breakdown
- Affordability band analysis

## Financial Model

**Viability Criteria:**
- DSCR >= 1.0 (NOI covers debt service)
- Cash-on-cash return >= user-defined target

**Loan Structure:** 20-year term, 50-year amortization

**Key parameters in `PARAM_INFO` dict:**
- `acquisition_cost_psf` - $/sqft purchase price
- `interest_rate` - senior debt rate
- `rent_psf` - monthly rent per sqft
- `equity_return_required` - target cash-on-cash return

## Data

- `accumulated_rental_inflation.csv` - 25-year rent escalation scenarios (used for animation)
- `synthetic_listings.csv` - Sample BC multifamily properties for analysis
- `market_rents.csv` - Market rent per sqft by city (used when cap_rate is missing)
- BC income distribution: median ~$95K CAD (2025 est.), log-normal with sigma=0.65

## Market Rent Fallback

When a listing doesn't have a cap_rate (needed to derive rent_psf), the app uses market rent by city from `data/market_rents.csv`. This allows all listings to be analyzed even with incomplete data.

The rent source is shown in the deep-dive tab: "(cap rate)" or "(market)".

## Listings CSV Format

The app accepts CSV files with multifamily property listings. Required columns:

| Column | Type | Description |
|--------|------|-------------|
| id | string | Unique identifier |
| address | string | Property address |
| city | string | City name |
| asking_price | float | Asking price in CAD |

Optional columns (enhance analysis):

| Column | Type | Description |
|--------|------|-------------|
| building_sqft | float | Total building square footage |
| num_units | int | Number of units (should be 20+) |
| year_built | int | Year constructed |
| cap_rate | float | Cap rate as decimal (e.g., 0.045 for 4.5%) |
| noi | float | Net operating income |
| source | string | Data source identifier |
| url | string | Link to listing |
| listing_status | string | active/sold/pending |
