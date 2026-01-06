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

    # BC median household income (2025 estimate, CAD)
    DEFAULT_MEDIAN = 95000

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
        affordability_ratio_lower: float = 0.25,
        affordability_ratio_upper: float = 0.30
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the affordability band for a given rent.

        The band represents households where rent is between lower and upper
        affordability ratios of their income:
        - Lower ratio (e.g., 25%): sets upper income bound (rent is at least this % of income)
        - Upper ratio (e.g., 30%): sets lower income bound (rent is at most this % of income)

        Args:
            monthly_rent: Monthly rent in dollars
            affordability_ratio_lower: Min rent as fraction of income (e.g., 0.25)
            affordability_ratio_upper: Max rent as fraction of income (e.g., 0.30)

        Returns:
            Tuple of (min_income, max_income, lower_percentile, upper_percentile)
        """
        annual_rent = monthly_rent * 12

        # Minimum income needed (rent = upper_ratio * income means they can just afford it)
        min_income = annual_rent / affordability_ratio_upper
        lower_percentile = self.percentile_at_income(min_income)

        # Maximum income (rent = lower_ratio * income means rent is "cheap enough" for program)
        # If lower_ratio is 0, there's no upper income cap
        if affordability_ratio_lower > 0:
            max_income = annual_rent / affordability_ratio_lower
            upper_percentile = self.percentile_at_income(max_income)
        else:
            max_income = self.income_at_percentile(100)
            upper_percentile = 100

        return min_income, max_income, lower_percentile, upper_percentile

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
    affordability_ratio_lower: float = 0.25,
    affordability_ratio_upper: float = 0.30,
    income_growth_rate: float = 0.02
) -> List[dict]:
    """
    Calculate affordability band evolution over time.

    Both rents and incomes evolve: rents by escalation_rates, incomes by income_growth_rate.

    Args:
        distribution: Income distribution model (baseline)
        initial_rent: Starting monthly rent
        escalation_rates: Annual rent escalation rates
        affordability_ratio_lower: Min rent as fraction of income (program floor)
        affordability_ratio_upper: Max rent as fraction of income (affordability cap)
        income_growth_rate: Annual income growth rate (inflation)

    Returns:
        List of dicts with band data for each year
    """
    rents = calculate_rent_over_time(initial_rent, escalation_rates)
    bands = []

    current_median = distribution.median

    for year, rent in enumerate(rents):
        # Create distribution for this year with inflated incomes
        year_distribution = BCIncomeDistribution(
            median=current_median,
            sigma=distribution.sigma
        )

        min_income, max_income, lower_pct, upper_pct = year_distribution.affordability_band(
            rent, affordability_ratio_lower, affordability_ratio_upper
        )

        # Band width in percentiles
        band_width = upper_pct - lower_pct if lower_pct < upper_pct else 0

        bands.append({
            'year': year,
            'rent': rent,
            'median_income': current_median,
            'min_income': min_income,
            'max_income': max_income,
            'lower_percentile': lower_pct,
            'upper_percentile': upper_pct,
            'band_width': band_width,
            'households_served_pct': max(0, band_width)
        })

        # Grow income for next year
        current_median *= (1 + income_growth_rate)

    return bands
