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
app.py                          # Main Streamlit entry point (3 tabs)
models/
  project_viability.py          # Financial model (DSCR, cash-on-cash, viability grid)
  income_distribution.py        # BC income distribution (log-normal model)
visualizations/
  income_band.py                # Bell curve + affordability band (animated)
  viability_space.py            # 2D parameter space heatmap
```

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
- BC income distribution: median ~$90K CAD, log-normal with sigma=0.65
