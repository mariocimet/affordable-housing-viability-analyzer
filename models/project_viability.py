"""
Project Viability Model for Affordable Housing Analysis

Calculates financial viability based on:
- Positive cash flow (DSCR >= 1.0)
- Equity return threshold (user-defined, cumulative over 25 years)
- Affordability for target income band (maintained throughout)

Viability is determined by optimizing rent increases over 25 years:
- Rents can increase up to the affordability ceiling each year
- A project is viable if it can achieve target cumulative return
  while never exceeding the affordability constraint
"""

import numpy as np
import numpy_financial as npf
from scipy import stats
from scipy.optimize import minimize, LinearConstraint
from dataclasses import dataclass
from typing import Tuple, List, Optional


# Housing charge increase scenarios (25 years)
# Pattern: rates tend to moderate over time
HOUSING_CHARGE_SCENARIOS = {
    'conservative': [0.02] * 25,  # Flat 2%
    'moderate': [0.025] * 5 + [0.025] * 5 + [0.0225] * 5 + [0.02] * 10,
    'aggressive': [0.03] * 5 + [0.0275] * 5 + [0.0225] * 5 + [0.02] * 10,
}


def calculate_project_irr(
    equity: float,
    cash_flows: List[float],
) -> float:
    """
    Calculate the Internal Rate of Return (IRR) for a project.

    Args:
        equity: Initial equity investment (positive value)
        cash_flows: List of annual cash flows (surplus) for years 1-25

    Returns:
        IRR as a decimal (e.g., 0.05 for 5%)
    """
    # Build full cash flow array: Year 0 is -equity, then annual surpluses
    all_flows = [-equity] + list(cash_flows)

    try:
        irr = npf.irr(all_flows)
        # Handle edge cases
        if np.isnan(irr) or np.isinf(irr):
            return 0.0
        return irr
    except Exception:
        return 0.0


@dataclass
class ProjectParameters:
    """Parameters for a housing project."""
    # Project Parameters
    acquisition_cost_psf: float = 400  # $/sqft purchase price (Seller)
    rent_psf: float = 2.5  # Monthly rent per sqft (Project)
    opex_ratio: float = 0.35  # Operating expenses as % of revenue (Project)
    housing_charge_increase: float = 0.03  # Annual rent increase (Project)

    # Lender Parameters
    interest_rate_senior: float = 0.05  # Interest rate on senior debt (Primary Lender)
    interest_rate_secondary: float = 0.08  # Interest rate on secondary debt (Secondary Lender)
    max_ltv: float = 0.75  # Maximum loan-to-value (Primary Lender)

    # Economic Parameters
    median_income: float = 95000  # BC median household income 2025 est. (Economic)
    risk_free_rate: float = 0.03  # Risk-free rate (Economic)
    inflation_assumption: float = 0.02  # Inflation assumption (Economic)

    # Conventional Parameters
    affordability_ratio: float = 0.30  # Max rent as % of income (affordability threshold)
    sqft_per_unit: float = 800  # Average unit size (Conventional)
    target_percentile: float = 50  # Target income percentile (e.g., 50 = median)

    # Target Returns
    equity_return_required: float = 0.0  # Target IRR over 25 years (user-defined)

    # Loan structure
    amortization_years: int = 50
    term_years: int = 20


class ProjectViabilityModel:
    """Calculate project viability metrics."""

    def __init__(self, params: ProjectParameters):
        self.params = params

    def annual_mortgage_payment_factor(self, interest_rate: float = None) -> float:
        """
        Calculate annual payment per dollar of loan principal.
        Uses standard mortgage amortization formula.
        """
        r = interest_rate if interest_rate is not None else self.params.interest_rate_senior
        n = self.params.amortization_years

        if r == 0:
            return 1 / n

        # Monthly rate and payments
        monthly_rate = r / 12
        num_payments = n * 12

        # Monthly payment factor
        monthly_factor = (monthly_rate * (1 + monthly_rate)**num_payments) / \
                        ((1 + monthly_rate)**num_payments - 1)

        # Annual payment factor
        return monthly_factor * 12

    def calculate_metrics(self, total_sqft: float = 10000) -> dict:
        """
        Calculate all financial metrics for a project.

        Args:
            total_sqft: Total building square footage (default 10,000 for normalization)

        Returns:
            Dictionary of financial metrics
        """
        p = self.params

        # Acquisition
        acquisition_cost = p.acquisition_cost_psf * total_sqft

        # Financing
        loan_amount = acquisition_cost * p.max_ltv
        equity = acquisition_cost * (1 - p.max_ltv)

        # Revenue & NOI
        annual_revenue = p.rent_psf * total_sqft * 12
        noi = annual_revenue * (1 - p.opex_ratio)

        # Debt service
        annual_payment_factor = self.annual_mortgage_payment_factor()
        annual_debt_service = loan_amount * annual_payment_factor

        # Cash flow
        cash_flow = noi - annual_debt_service

        # Ratios
        dscr = noi / annual_debt_service if annual_debt_service > 0 else float('inf')
        cash_on_cash = cash_flow / equity if equity > 0 else float('inf')

        return {
            'acquisition_cost': acquisition_cost,
            'loan_amount': loan_amount,
            'equity': equity,
            'annual_revenue': annual_revenue,
            'noi': noi,
            'annual_debt_service': annual_debt_service,
            'cash_flow': cash_flow,
            'dscr': dscr,
            'cash_on_cash': cash_on_cash,
        }

    def calculate_metrics_over_time(
        self,
        total_sqft: float = 10000,
        years: int = 25,
        income_growth_rate: float = None,
        rate_schedule: List[float] = None
    ) -> list:
        """
        Calculate financial metrics over time as rents increase.

        Revenue grows by housing charge increases (per schedule or uniform).
        Debt service stays constant (fixed-rate mortgage).
        Income grows by income_growth_rate for affordability tracking.

        Args:
            total_sqft: Total building square footage
            years: Number of years to project (default 25)
            income_growth_rate: Annual income growth rate (default: inflation_assumption)
            rate_schedule: List of housing charge increase rates for each year.
                          If None, uses uniform rate from params.housing_charge_increase

        Returns:
            List of dicts with yearly metrics including 'equity' and IRR-ready cash flows
        """
        p = self.params
        if income_growth_rate is None:
            income_growth_rate = p.inflation_assumption

        # Build rate schedule if not provided
        if rate_schedule is None:
            rate_schedule = [p.housing_charge_increase] * years

        # Ensure schedule has enough rates
        if len(rate_schedule) < years:
            rate_schedule = rate_schedule + [rate_schedule[-1]] * (years - len(rate_schedule))

        # Initial calculations (Year 0)
        acquisition_cost = p.acquisition_cost_psf * total_sqft
        loan_amount = acquisition_cost * p.max_ltv
        equity = acquisition_cost * (1 - p.max_ltv)

        # Debt service is constant over the loan term
        annual_payment_factor = self.annual_mortgage_payment_factor()
        annual_debt_service = loan_amount * annual_payment_factor

        # Track cumulative cash flow for IRR-like metrics
        cumulative_cash_flow = 0
        current_rent_psf = p.rent_psf
        current_median_income = p.median_income

        results = []

        for year in range(years + 1):
            # Revenue grows with housing charge increases (use schedule)
            if year > 0:
                year_rate = rate_schedule[year - 1] if year <= len(rate_schedule) else rate_schedule[-1]
                current_rent_psf *= (1 + year_rate)
                current_median_income *= (1 + income_growth_rate)

            annual_revenue = current_rent_psf * total_sqft * 12
            noi = annual_revenue * (1 - p.opex_ratio)
            cash_flow = noi - annual_debt_service

            # Ratios
            dscr = noi / annual_debt_service if annual_debt_service > 0 else float('inf')
            cash_on_cash = cash_flow / equity if equity > 0 else float('inf')

            # Cumulative
            cumulative_cash_flow += cash_flow
            cumulative_return = cumulative_cash_flow / equity if equity > 0 else 0

            # Affordability at this year's rent and income
            monthly_rent = current_rent_psf * p.sqft_per_unit
            annual_rent = monthly_rent * 12

            # Income distribution for this year
            mu = np.log(current_median_income)
            sigma = 0.65
            dist = stats.lognorm(s=sigma, scale=np.exp(mu))
            target_income = dist.ppf(p.target_percentile / 100)
            rent_burden = annual_rent / target_income if target_income > 0 else float('inf')
            is_affordable = rent_burden <= p.affordability_ratio

            # Viability at this point
            positive_cash_flow = dscr >= 1.0
            meets_equity_return = cash_on_cash >= p.equity_return_required
            is_viable = positive_cash_flow and meets_equity_return and is_affordable

            results.append({
                'year': year,
                'rent_psf': current_rent_psf,
                'annual_revenue': annual_revenue,
                'noi': noi,
                'debt_service': annual_debt_service,
                'cash_flow': cash_flow,
                'dscr': dscr,
                'cash_on_cash': cash_on_cash,
                'cumulative_cash_flow': cumulative_cash_flow,
                'cumulative_return': cumulative_return,
                'monthly_rent': monthly_rent,
                'median_income': current_median_income,
                'rent_burden': rent_burden,
                'is_affordable': is_affordable,
                'is_viable': is_viable,
                'equity': equity,  # For IRR calculation
            })

        return results

    def check_affordability(self) -> Tuple[bool, dict]:
        """
        Check if rent is affordable for the target income percentile.

        Returns:
            Tuple of (is_affordable, affordability_metrics)
        """
        p = self.params

        # Calculate monthly rent
        monthly_rent = p.rent_psf * p.sqft_per_unit
        annual_rent = monthly_rent * 12

        # Create income distribution
        mu = np.log(p.median_income)
        sigma = 0.65  # Standard BC income distribution sigma
        dist = stats.lognorm(s=sigma, scale=np.exp(mu))

        # Get income at target percentile
        target_income = dist.ppf(p.target_percentile / 100)

        # Calculate rent burden
        rent_burden = annual_rent / target_income if target_income > 0 else float('inf')

        # Affordability check: rent must be <= affordability_ratio of target income
        is_affordable = rent_burden <= p.affordability_ratio

        return is_affordable, {
            'monthly_rent': monthly_rent,
            'target_income': target_income,
            'rent_burden': rent_burden,
            'is_affordable': is_affordable
        }

    def is_viable(self) -> Tuple[bool, dict]:
        """
        Determine if project meets viability criteria:
        1. DSCR >= 1.0 (positive cash flow)
        2. Cash-on-cash >= equity_return_required
        3. Rent affordable for target income band

        Returns:
            Tuple of (is_viable, metrics_dict)
        """
        metrics = self.calculate_metrics()

        # Financial criteria
        positive_cash_flow = metrics['dscr'] >= 1.0
        meets_equity_return = metrics['cash_on_cash'] >= self.params.equity_return_required

        # Affordability criteria
        is_affordable, affordability_metrics = self.check_affordability()

        metrics['positive_cash_flow'] = positive_cash_flow
        metrics['meets_equity_return'] = meets_equity_return
        metrics['is_affordable'] = is_affordable
        metrics['affordability'] = affordability_metrics

        # All three criteria must be met
        metrics['is_viable'] = positive_cash_flow and meets_equity_return and is_affordable

        return metrics['is_viable'], metrics


def analyze_rent_roll_affordability(
    rent_roll: 'RentRoll',
    median_income: float = 95000,
    affordability_ratio: float = 0.30,
    sigma: float = 0.65
) -> list:
    """
    Analyze affordability for each unit type in a rent roll.

    Args:
        rent_roll: RentRoll object with unit types
        median_income: Median household income
        affordability_ratio: Max rent as % of income (e.g., 0.30 = 30%)
        sigma: Income distribution sigma (0.65 for BC)

    Returns:
        List of dicts with affordability metrics per unit type:
        - unit_type, sqft, monthly_rent, count
        - annual_rent, min_income_required
        - percentile_required, pct_can_afford
    """
    mu = np.log(median_income)
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    results = []
    for ut in rent_roll.unit_types:
        if ut.count == 0:
            continue

        annual_rent = ut.monthly_rent * 12
        min_income = annual_rent / affordability_ratio
        percentile = dist.cdf(min_income) * 100

        results.append({
            'unit_type': ut.name,
            'sqft': ut.sqft,
            'monthly_rent': ut.monthly_rent,
            'count': ut.count,
            'rent_psf': ut.rent_psf,
            'annual_rent': annual_rent,
            'min_income_required': min_income,
            'percentile_required': percentile,
            'pct_can_afford': 100 - percentile
        })

    return results


def calculate_viability_grid(
    x_param: str,
    y_param: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    fixed_params: dict,
    resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate viability across a 2D parameter space.

    Args:
        x_param: Name of parameter for X axis
        y_param: Name of parameter for Y axis
        x_range: (min, max) for X axis
        y_range: (min, max) for Y axis
        fixed_params: Dictionary of fixed parameter values
        resolution: Grid resolution

    Returns:
        Tuple of (X grid, Y grid, viability grid, margin grid)
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    viability = np.zeros_like(X)
    margin = np.zeros_like(X)  # How much above/below threshold

    # Map display param names to ProjectParameters field names
    param_mapping = {'min_rent_psf': 'rent_psf'}

    for i in range(resolution):
        for j in range(resolution):
            params_dict = fixed_params.copy()
            # Map param names if needed
            x_field = param_mapping.get(x_param, x_param)
            y_field = param_mapping.get(y_param, y_param)
            params_dict[x_field] = X[i, j]
            params_dict[y_field] = Y[i, j]

            params = ProjectParameters(**params_dict)
            model = ProjectViabilityModel(params)
            is_viable, metrics = model.is_viable()

            # Viability score: 0-3 based on criteria met
            score = (int(metrics['positive_cash_flow']) +
                    int(metrics['meets_equity_return']) +
                    int(metrics['is_affordable']))
            viability[i, j] = score

            # Margin: minimum of how far above each threshold
            dscr_margin = metrics['dscr'] - 1.0
            equity_margin = metrics['cash_on_cash'] - params.equity_return_required
            # For affordability, use rent burden margin (how much below the limit)
            aff = metrics['affordability']
            affordability_margin = params.affordability_ratio - aff['rent_burden']
            margin[i, j] = min(dscr_margin, equity_margin, affordability_margin)

    return X, Y, viability, margin


# Parameter display names, ranges, and categories
PARAM_INFO = {
    # Project Parameters (Seller/Project driven)
    'acquisition_cost_psf': {
        'name': 'Cost per Sq Ft',
        'category': 'Project',
        'driver': 'Cost of Seller\'s Capital',
        'default_range': (100, 1000),
        'default': 400,
        'format': '${:.0f}'
    },
    'rent_psf': {
        'name': 'Rent per Sq Ft (Monthly)',
        'category': 'Project',
        'driver': 'Rental Market, Affordability, Debt Requirements',
        'default_range': (0.5, 10.0),
        'default': 2.5,
        'format': '${:.2f}'
    },
    'min_rent_psf': {
        'name': 'Min Rent for DSCR=1 ($/sqft)',
        'category': 'Derived',
        'driver': 'Acquisition cost, financing terms',
        'default_range': (0.5, 10.0),
        'default': 3.5,
        'format': '${:.2f}'
    },
    'opex_ratio': {
        'name': 'Operating Expense Ratio',
        'category': 'Project',
        'driver': 'Inflation, Cost of Administration, Labour Market',
        'default_range': (0.0, 1.0),
        'default': 0.25,
        'format': '{:.1%}'
    },
    'housing_charge_increase': {
        'name': 'Housing Charge Increase',
        'category': 'Project',
        'driver': 'Historical Trend',
        'default_range': (-0.5, 0.5),
        'default': 0.03,
        'format': '{:.1%}',
        'precise_input': True
    },

    # Lender Parameters
    'interest_rate_senior': {
        'name': 'Interest Rate (Senior Debt)',
        'category': 'Primary Lender',
        'driver': 'ALM, Cost of Capital, GoC Yield',
        'default_range': (0.0, 0.20),
        'default': 0.048,
        'format': '{:.1%}',
        'precise_input': True
    },
    'interest_rate_secondary': {
        'name': 'Interest Rate (Secondary Debt)',
        'category': 'Secondary Lender',
        'driver': 'ALM, Cost of Capital, GoC Yield',
        'default_range': (0.0, 0.25),
        'default': 0.08,
        'format': '{:.1%}',
        'precise_input': True
    },
    'max_ltv': {
        'name': 'Maximum Loan to Value',
        'category': 'Primary Lender',
        'driver': 'ALM, Cost of Capital, GoC Yield',
        'default_range': (0.0, 1.0),
        'default': 0.80,
        'format': '{:.1%}'
    },

    # Economic Parameters
    'median_income': {
        'name': 'Median Income',
        'category': 'Economic',
        'driver': 'Labour Market, Wage Fluctuation',
        'default_range': (30000, 200000),
        'default': 95000,
        'format': '${:,.0f}'
    },
    'risk_free_rate': {
        'name': 'Risk Free Rate',
        'category': 'Economic',
        'driver': 'Inflation Expectations',
        'default_range': (0.0, 0.15),
        'default': 0.03,
        'format': '{:.1%}',
        'precise_input': True
    },
    'inflation_assumption': {
        'name': 'Inflation Assumption',
        'category': 'Economic',
        'driver': 'Historical Trend',
        'default_range': (-0.10, 0.15),
        'default': 0.02,
        'format': '{:.1%}',
        'precise_input': True
    },

    # Conventional Parameters
    'affordability_ratio': {
        'name': 'Affordability Ratio',
        'category': 'Affordability',
        'driver': 'Max rent as % of household income',
        'default_range': (0.0, 1.0),
        'default': 0.30,
        'format': '{:.1%}'
    },
    'sqft_per_unit': {
        'name': 'Sq Ft per Household',
        'category': 'Conventional',
        'driver': 'Living Space Expectations',
        'default_range': (200, 2000),
        'default': 800,
        'format': '{:.0f}'
    },
    'target_percentile': {
        'name': 'Target Income Percentile',
        'category': 'Affordability',
        'driver': 'Income level to target (e.g., 50 = median)',
        'default_range': (0, 100),
        'default': 50,
        'format': '{:.0f}%ile'
    },

    # Target Returns
    'equity_return_required': {
        'name': 'Target IRR (25-yr)',
        'category': 'Target',
        'driver': 'User-defined minimum annualized return over 25 years',
        'default_range': (0.0, 0.15),
        'default': 0.05,
        'format': '{:.1%}'
    }
}
