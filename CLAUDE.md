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

## Development Approach

### Test assumptions before implementation
Before writing substantial code for external integrations (APIs, scrapers, third-party services), first verify that the integration is feasible with a minimal test. Do not write multiple files or complete implementations before confirming basic connectivity and access.

### Flag feasibility risks in plans
When creating implementation plans, explicitly identify and call out risks that could block the entire approach (e.g., anti-bot measures, authentication requirements, rate limits, API availability). Ask for validation of these risks before proceeding.

### Prefer incremental milestones
Break work into small, testable steps. Confirm each step works before moving to the next:
1. Verify we can access the data source
2. Parse a single record
3. Add persistence/caching
4. Build UI integration

### No mock data without explicit approval
Do not substitute sample/mock/fake data to demonstrate features when real data acquisition fails. Instead, surface the failure clearly and propose realistic alternatives (manual input, CSV import, different data source, commercial API).

### Surface blockers immediately
If you encounter access denied errors, authentication failures, or other blockers, stop and report immediately rather than attempting multiple workarounds silently. Present the options and let the user decide how to proceed.

## Multi-Instance Workflow

### When working in a git worktree or feature branch
You may be one of several Claude Code instances working on this codebase simultaneously. Each instance operates in its own worktree/branch.

**Before starting work:**
1. Run `git status` to confirm which branch you're on
2. Note your branch name—this defines your scope

**Scope discipline:**
- Only commit to your assigned branch
- Do not modify files outside your assigned scope unless explicitly told to
- If you need changes in files another instance owns, ask and wait for coordination

**Before making broad changes:**
Ask first before:
- Renaming or moving files
- Modifying shared configuration (package.json, requirements.txt, pyproject.toml, etc.)
- Changing interfaces or function signatures used by other modules
- Running formatters or linters on files outside your scope

### Branch/scope assignments
<!-- Update this section when running multiple instances -->
When a session starts, scope will be specified. Examples:
- "You're working on the `/scrapers` module only"
- "You own feature-x branch, focus on authentication"

If scope hasn't been specified, ask before starting.

### Coordination protocol
If work requires changes to files outside your scope:
1. Stop and describe what change is needed
2. Wait for approval, manual change, or coordination with the other instance
3. Do not assume the change can be made

### Merging and conflicts
Do not attempt to merge branches or resolve conflicts from other instances. Report when your branch is ready to merge—integration will be handled separately.
