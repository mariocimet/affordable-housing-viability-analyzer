"""
BC Income Distribution Model

Models household income distribution in British Columbia using log-normal distribution.
Based on Statistics Canada data for BC median household income.
"""

import numpy as np
from scipy import stats
from typing import Tuple, List


class BCIncomeDistribution:
    """
    Model BC household income distribution.

    Uses log-normal distribution which is standard for income modeling.
    Parameters calibrated to BC Statistics Canada data.
    """

    # BC median household income (2023 estimate, CAD)
    DEFAULT_MEDIAN = 90000

    # Log-normal sigma parameter (controls spread/inequality)
    # Higher sigma = more inequality, longer right tail
    DEFAULT_SIGMA = 0.65

    def __init__(self, median: float = None, sigma: float = None):
        """
        Initialize income distribution.

        Args:
            median: Median household income (default: $90,000 CAD)
            sigma: Log-normal sigma parameter (default: 0.65)
        """
        self.median = median or self.DEFAULT_MEDIAN
        self.sigma = sigma or self.DEFAULT_SIGMA

        # For log-normal, median = exp(mu), so mu = ln(median)
        self.mu = np.log(self.median)

        # Create the distribution
        self.dist = stats.lognorm(s=self.sigma, scale=np.exp(self.mu))

    def pdf(self, income: np.ndarray) -> np.ndarray:
        """Probability density function."""
        return self.dist.pdf(income)

    def cdf(self, income: float) -> float:
        """Cumulative distribution function (percentile for given income)."""
        return self.dist.cdf(income)

    def ppf(self, percentile: float) -> float:
        """Percent point function (income for given percentile)."""
        return self.dist.ppf(percentile)

    def income_at_percentile(self, percentile: float) -> float:
        """Get income at a given percentile (0-100)."""
        return self.ppf(percentile / 100)

    def percentile_at_income(self, income: float) -> float:
        """Get percentile (0-100) for a given income."""
        return self.cdf(income) * 100

    def get_percentile_incomes(self) -> dict:
        """Get income values at common percentiles."""
        percentiles = [10, 25, 50, 75, 90, 95]
        return {p: self.income_at_percentile(p) for p in percentiles}

    def affordability_band(
        self,
        monthly_rent: float,
        affordability_ratio: float = 0.30,
        upper_percentile: float = 75
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the affordability band for a given rent.

        Args:
            monthly_rent: Monthly rent in dollars
            affordability_ratio: Max rent as fraction of income (default 30%)
            upper_percentile: Upper income percentile limit (default 75th)

        Returns:
            Tuple of (lower_income, upper_income, lower_percentile, upper_percentile)
            where lower_income is the minimum income needed to afford the rent
        """
        annual_rent = monthly_rent * 12

        # Minimum income needed (rent = affordability_ratio * income)
        min_income_needed = annual_rent / affordability_ratio
        lower_percentile = self.percentile_at_income(min_income_needed)

        # Upper bound is the program limit
        upper_income = self.income_at_percentile(upper_percentile)

        return min_income_needed, upper_income, lower_percentile, upper_percentile

    def get_curve_data(
        self,
        num_points: int = 500,
        max_income: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data points for plotting the income distribution curve.

        Args:
            num_points: Number of points to generate
            max_income: Maximum income to plot (default: 99th percentile)

        Returns:
            Tuple of (income_values, density_values)
        """
        if max_income is None:
            max_income = self.income_at_percentile(99)

        incomes = np.linspace(1000, max_income, num_points)
        densities = self.pdf(incomes)

        return incomes, densities


def calculate_rent_over_time(
    initial_rent: float,
    escalation_rates: List[float]
) -> List[float]:
    """
    Calculate rent trajectory over time given escalation rates.

    Args:
        initial_rent: Starting monthly rent
        escalation_rates: List of annual escalation rates (as decimals)

    Returns:
        List of rents for each year (starting with initial)
    """
    rents = [initial_rent]
    current_rent = initial_rent

    for rate in escalation_rates:
        current_rent *= (1 + rate)
        rents.append(current_rent)

    return rents


def calculate_band_over_time(
    distribution: BCIncomeDistribution,
    initial_rent: float,
    escalation_rates: List[float],
    affordability_ratio: float = 0.30,
    upper_percentile: float = 75
) -> List[dict]:
    """
    Calculate affordability band evolution over time.

    Args:
        distribution: Income distribution model
        initial_rent: Starting monthly rent
        escalation_rates: Annual rent escalation rates
        affordability_ratio: Max rent as fraction of income
        upper_percentile: Upper income percentile limit

    Returns:
        List of dicts with band data for each year
    """
    rents = calculate_rent_over_time(initial_rent, escalation_rates)
    bands = []

    for year, rent in enumerate(rents):
        min_income, max_income, lower_pct, upper_pct = distribution.affordability_band(
            rent, affordability_ratio, upper_percentile
        )

        # Band width in percentiles
        band_width = upper_pct - lower_pct if lower_pct < upper_pct else 0

        bands.append({
            'year': year,
            'rent': rent,
            'min_income': min_income,
            'max_income': max_income,
            'lower_percentile': lower_pct,
            'upper_percentile': upper_pct,
            'band_width': band_width,
            'households_served_pct': max(0, band_width)
        })

    return bands
