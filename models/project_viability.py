"""
Project Viability Model for Affordable Housing Analysis

Calculates financial viability based on:
- Positive cash flow (DSCR >= 1.0)
- Equity return threshold (user-defined)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


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
    median_income: float = 90000  # Median household income (Economic)
    risk_free_rate: float = 0.03  # Risk-free rate (Economic)
    inflation_assumption: float = 0.02  # Inflation assumption (Economic)

    # Conventional Parameters
    affordability_ratio: float = 0.30  # Max rent as % of income (Conventional)
    sqft_per_unit: float = 800  # Average unit size (Conventional)

    # Target Returns
    equity_return_required: float = 0.0  # Target cash-on-cash return (user-defined)

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

    def is_viable(self) -> Tuple[bool, dict]:
        """
        Determine if project meets viability criteria.

        Returns:
            Tuple of (is_viable, metrics_dict)
        """
        metrics = self.calculate_metrics()

        positive_cash_flow = metrics['dscr'] >= 1.0
        meets_equity_return = metrics['cash_on_cash'] >= self.params.equity_return_required

        metrics['positive_cash_flow'] = positive_cash_flow
        metrics['meets_equity_return'] = meets_equity_return
        metrics['is_viable'] = positive_cash_flow and meets_equity_return

        return metrics['is_viable'], metrics


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

    for i in range(resolution):
        for j in range(resolution):
            params_dict = fixed_params.copy()
            params_dict[x_param] = X[i, j]
            params_dict[y_param] = Y[i, j]

            params = ProjectParameters(**params_dict)
            model = ProjectViabilityModel(params)
            is_viable, metrics = model.is_viable()

            # Viability score: 0 = not viable, 1 = one criterion, 2 = both
            score = int(metrics['positive_cash_flow']) + int(metrics['meets_equity_return'])
            viability[i, j] = score

            # Margin: minimum of how far above each threshold
            dscr_margin = metrics['dscr'] - 1.0
            equity_margin = metrics['cash_on_cash'] - params.equity_return_required
            margin[i, j] = min(dscr_margin, equity_margin)

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
    'opex_ratio': {
        'name': 'Operating Expense Ratio',
        'category': 'Project',
        'driver': 'Inflation, Cost of Administration, Labour Market',
        'default_range': (0.0, 1.0),
        'default': 0.35,
        'format': '{:.1%}'
    },
    'housing_charge_increase': {
        'name': 'Housing Charge Increase',
        'category': 'Project',
        'driver': 'Historical Trend',
        'default_range': (-0.5, 0.5),
        'default': 0.03,
        'format': '{:.1%}'
    },

    # Lender Parameters
    'interest_rate_senior': {
        'name': 'Interest Rate (Senior Debt)',
        'category': 'Primary Lender',
        'driver': 'ALM, Cost of Capital, GoC Yield',
        'default_range': (0.0, 1.0),
        'default': 0.05,
        'format': '{:.1%}'
    },
    'interest_rate_secondary': {
        'name': 'Interest Rate (Secondary Debt)',
        'category': 'Secondary Lender',
        'driver': 'ALM, Cost of Capital, GoC Yield',
        'default_range': (0.0, 1.0),
        'default': 0.08,
        'format': '{:.1%}'
    },
    'max_ltv': {
        'name': 'Maximum Loan to Value',
        'category': 'Primary Lender',
        'driver': 'ALM, Cost of Capital, GoC Yield',
        'default_range': (0.0, 1.0),
        'default': 0.75,
        'format': '{:.1%}'
    },

    # Economic Parameters
    'median_income': {
        'name': 'Median Income',
        'category': 'Economic',
        'driver': 'Labour Market, Wage Fluctuation',
        'default_range': (30000, 200000),
        'default': 90000,
        'format': '${:,.0f}'
    },
    'risk_free_rate': {
        'name': 'Risk Free Rate',
        'category': 'Economic',
        'driver': 'Inflation Expectations',
        'default_range': (0.0, 1.0),
        'default': 0.03,
        'format': '{:.1%}'
    },
    'inflation_assumption': {
        'name': 'Inflation Assumption',
        'category': 'Economic',
        'driver': 'Historical Trend',
        'default_range': (-0.5, 0.5),
        'default': 0.02,
        'format': '{:.1%}'
    },

    # Conventional Parameters
    'affordability_ratio': {
        'name': 'Affordability Ratio',
        'category': 'Conventional',
        'driver': 'Household Financial Planning Expectations',
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

    # Target Returns
    'equity_return_required': {
        'name': 'Target Equity Return',
        'category': 'Target',
        'driver': 'User-defined',
        'default_range': (0.0, 1.0),
        'default': 0.0,
        'format': '{:.1%}'
    }
}
