"""
Parameter Mapper

Maps listing data to project viability model parameters.
Derives minimum rent needed for DSCR = 1.0 as a model output.
"""

from typing import Optional, List
from pathlib import Path
import pandas as pd

from models.listing import MultifamilyListing
from models.project_viability import (
    ProjectParameters,
    ProjectViabilityModel,
    HOUSING_CHARGE_SCENARIOS,
    calculate_project_irr
)


def calculate_min_rent_for_dscr(
    acquisition_cost_psf: float,
    interest_rate: float = 0.045,
    max_ltv: float = 0.95,
    opex_ratio: float = 0.35,
    amortization_years: int = 50,
    equity_return_required: float = 0.0
) -> float:
    """
    Calculate minimum rent_psf needed to achieve DSCR >= 1.0 AND target equity return.

    The minimum rent is the higher of:
    1. Rent for DSCR = 1.0 (covers debt service)
    2. Rent for target equity return (provides cash flow to equity)

    For DSCR = 1.0:
        rent_psf = (cost × LTV × payment_factor) / (12 × (1 - opex))

    For target equity return:
        Cash Flow = target_return × Equity
        NOI = Debt Service + target_return × Equity
        rent_psf = cost × (LTV × payment_factor + target_return × (1 - LTV)) / (12 × (1 - opex))

    Returns the higher of the two (or they're equal when target_return = 0).
    """
    # Calculate annual payment factor (same as in ProjectViabilityModel)
    r = interest_rate
    n = amortization_years
    if r == 0:
        annual_payment_factor = 1 / n
    else:
        monthly_rate = r / 12
        num_payments = n * 12
        monthly_factor = (monthly_rate * (1 + monthly_rate)**num_payments) / \
                        ((1 + monthly_rate)**num_payments - 1)
        annual_payment_factor = monthly_factor * 12

    # Rent for DSCR = 1.0 (debt coverage)
    rent_for_dscr = (acquisition_cost_psf * max_ltv * annual_payment_factor) / (12 * (1 - opex_ratio))

    # Rent for target equity return
    # NOI = Debt Service + target_return × Equity
    # rent × 12 × (1-opex) = cost × LTV × payment + target_return × cost × (1-LTV)
    # rent = cost × (LTV × payment + target_return × (1-LTV)) / (12 × (1-opex))
    rent_for_equity = (acquisition_cost_psf * (max_ltv * annual_payment_factor + equity_return_required * (1 - max_ltv))) / (12 * (1 - opex_ratio))

    # Return the higher of the two requirements
    return max(rent_for_dscr, rent_for_equity)


def listing_to_params(
    listing: MultifamilyListing,
    assumptions: dict = None
) -> dict:
    """
    Convert a listing to viability model parameters.

    Note: rent_psf is NOT derived here - it's calculated by the model
    as the minimum rent needed for DSCR = 1.0.

    Args:
        listing: MultifamilyListing object
        assumptions: Dict of assumption values for missing data:
            - sqft_per_unit: Average sqft per unit (default 800)

    Returns:
        Dict with parameters including acquisition_cost_psf
    """
    if assumptions is None:
        assumptions = {}

    sqft_per_unit = assumptions.get('sqft_per_unit', 800)

    params = {
        'id': listing.id,
        'name': listing.display_name,
        'address': listing.address,
        'city': listing.city,
        'source': listing.source,
        'url': listing.url,
    }

    # Calculate acquisition_cost_psf
    if listing.building_sqft and listing.building_sqft > 0:
        # Direct calculation from building sqft
        params['acquisition_cost_psf'] = listing.asking_price / listing.building_sqft
        params['building_sqft'] = listing.building_sqft
        params['sqft_estimated'] = False
    elif listing.num_units and listing.num_units > 0:
        # Estimate sqft from units
        estimated_sqft = listing.num_units * sqft_per_unit
        params['acquisition_cost_psf'] = listing.asking_price / estimated_sqft
        params['building_sqft'] = estimated_sqft
        params['sqft_estimated'] = True
    else:
        # Cannot calculate without sqft or units
        params['acquisition_cost_psf'] = None
        params['building_sqft'] = None
        params['sqft_estimated'] = None

    # Store raw values
    params['asking_price'] = listing.asking_price
    params['num_units'] = listing.num_units
    params['year_built'] = listing.year_built

    return params


def listings_to_dataframe(
    listings: List[MultifamilyListing],
    assumptions: dict = None,
    use_market_rent_fallback: bool = True
) -> pd.DataFrame:
    """
    Convert a list of listings to a DataFrame with viability parameters.

    Args:
        listings: List of MultifamilyListing objects
        assumptions: Dict of assumptions for missing data
        use_market_rent_fallback: If True, use market rent when cap_rate unavailable

    Returns:
        DataFrame with columns for plotting on viability space
    """
    rows = []
    for listing in listings:
        params = listing_to_params(listing, assumptions)
        rows.append(params)

    df = pd.DataFrame(rows)

    # Filter to only listings with calculable acquisition_cost_psf
    df = df[df['acquisition_cost_psf'].notna()]

    return df


def analyze_portfolio(
    listings: List[MultifamilyListing],
    fixed_params: dict,
    assumptions: dict = None,
    scenario: str = 'moderate'
) -> pd.DataFrame:
    """
    Analyze all listings and compute viability metrics using IRR over 25 years.

    The model:
    1. Derives minimum rent needed for DSCR = 1.0 on Day 1
    2. Projects cash flows over 25 years using housing charge increase scenario
    3. Calculates IRR from the cash flow stream
    4. Checks affordability at the minimum required rent

    A project is viable if:
    - DSCR >= 1.0 on Day 1 (covered by construction)
    - IRR over 25 years >= target equity return
    - Rent is affordable at target income percentile

    Args:
        listings: List of MultifamilyListing objects
        fixed_params: Dict of fixed parameters (interest rates, LTV, affordability, etc.)
        assumptions: Dict of assumptions (sqft_per_unit)
        scenario: Housing charge increase scenario ('conservative', 'moderate', 'aggressive')

    Returns:
        DataFrame with columns:
        - All listing fields (address, city, price, units, etc.)
        - acquisition_cost_psf, min_rent_psf (derived for DSCR=1)
        - irr_25yr, is_affordable, rent_burden, viability_status
    """
    if assumptions is None:
        assumptions = {}

    # Get rate schedule for the selected scenario
    rate_schedule = HOUSING_CHARGE_SCENARIOS.get(scenario, HOUSING_CHARGE_SCENARIOS['moderate'])

    rows = []
    for listing in listings:
        params = listing_to_params(listing, assumptions)

        # Skip if we can't calculate acquisition cost
        if params.get('acquisition_cost_psf') is None:
            params['min_rent_psf'] = None
            params['irr_25yr'] = None
            params['is_viable'] = None
            params['is_affordable'] = None
            params['rent_burden'] = None
            params['viability_status'] = 'Missing Data'
            rows.append(params)
            continue

        # Calculate minimum rent for DSCR = 1.0 on Day 1
        # Note: We don't include equity_return_required here since IRR is computed over time
        min_rent_psf = calculate_min_rent_for_dscr(
            acquisition_cost_psf=params['acquisition_cost_psf'],
            interest_rate=fixed_params.get('interest_rate_senior', 0.045),
            max_ltv=fixed_params.get('max_ltv', 0.95),
            opex_ratio=fixed_params.get('opex_ratio', 0.35),
            amortization_years=50,
            equity_return_required=0.0  # DSCR=1 only; IRR handles return target
        )
        params['min_rent_psf'] = min_rent_psf

        # Check affordability and calculate IRR
        try:
            # Build params for viability model
            viability_params = fixed_params.copy()
            viability_params['acquisition_cost_psf'] = params['acquisition_cost_psf']
            viability_params['rent_psf'] = min_rent_psf

            project = ProjectParameters(**viability_params)
            model = ProjectViabilityModel(project)

            # Check affordability on Day 1 (is min rent affordable at target income?)
            is_affordable, aff_metrics = model.check_affordability()

            params['is_affordable'] = is_affordable
            params['rent_burden'] = aff_metrics.get('rent_burden')
            params['target_income'] = aff_metrics.get('target_income')
            params['monthly_rent'] = aff_metrics.get('monthly_rent')

            # Project cash flows over 25 years with selected scenario
            yearly_metrics = model.calculate_metrics_over_time(
                total_sqft=params.get('building_sqft', 10000),
                years=25,
                rate_schedule=rate_schedule
            )

            # Extract cash flows for years 1-25 (skip year 0)
            cash_flows = [m['cash_flow'] for m in yearly_metrics if m['year'] > 0]
            equity = yearly_metrics[0]['equity']

            # Calculate IRR
            irr_25yr = calculate_project_irr(equity, cash_flows)
            params['irr_25yr'] = irr_25yr

            # Target IRR from fixed params
            target_irr = fixed_params.get('equity_return_required', 0.0)
            meets_irr_target = irr_25yr >= target_irr

            # Viability = DSCR>=1 (by construction) AND IRR target AND affordable
            params['is_viable'] = is_affordable and meets_irr_target

            if params['is_viable']:
                params['viability_status'] = 'Viable'
            elif not is_affordable:
                burden_pct = params['rent_burden'] * 100 if params['rent_burden'] else 0
                threshold_pct = fixed_params.get('affordability_ratio', 0.30) * 100
                params['viability_status'] = f'Not Affordable ({burden_pct:.0f}% > {threshold_pct:.0f}%)'
            else:
                # Affordable but IRR too low
                params['viability_status'] = f'IRR Too Low ({irr_25yr:.1%} < {target_irr:.1%})'

        except Exception as e:
            params['irr_25yr'] = None
            params['is_viable'] = None
            params['is_affordable'] = None
            params['rent_burden'] = None
            params['viability_status'] = f'Error: {str(e)}'

        rows.append(params)

    return pd.DataFrame(rows)


def get_plottable_listings(
    listings: List[MultifamilyListing],
    x_param: str,
    y_param: str,
    assumptions: dict = None
) -> pd.DataFrame:
    """
    Get listings that can be plotted on the viability space.

    Filters to listings that have values for both x and y parameters.

    Args:
        listings: List of MultifamilyListing objects
        x_param: Parameter name for x-axis
        y_param: Parameter name for y-axis
        assumptions: Dict of assumptions for missing data

    Returns:
        DataFrame with columns: name, x_param, y_param, url
    """
    df = listings_to_dataframe(listings, assumptions)

    if df.empty:
        return df

    # Map viability param names to listing param names
    param_mapping = {
        'acquisition_cost_psf': 'acquisition_cost_psf',
        'rent_psf': 'rent_psf',
    }

    x_col = param_mapping.get(x_param, x_param)
    y_col = param_mapping.get(y_param, y_param)

    # Check if we have the required columns
    if x_col not in df.columns:
        return pd.DataFrame()

    # Filter to rows with both values
    if y_col in df.columns:
        mask = df[x_col].notna() & df[y_col].notna()
        result = df[mask].copy()
        result['x'] = result[x_col]
        result['y'] = result[y_col]
    else:
        # If y_param not derivable from listings, can only plot on x-axis
        mask = df[x_col].notna()
        result = df[mask].copy()
        result['x'] = result[x_col]
        result['y'] = None  # Will need to be set by user or estimated

    return result
