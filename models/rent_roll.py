"""
Rent Roll Model

Defines unit types and rent roll for multifamily properties.
"""

from pydantic import BaseModel, Field, computed_field
from typing import List


class UnitType(BaseModel):
    """A unit type in a rent roll."""
    name: str = Field(description="Unit type name (e.g., 'Studio', '1BR', '2BR')")
    bedrooms: int = Field(default=1, description="Number of bedrooms (0=Studio, 1=1BR, 2=2BR, etc.)")
    sqft: float = Field(description="Square footage per unit")
    monthly_rent: float = Field(description="Monthly rent for this unit type")
    count: int = Field(description="Number of units of this type")

    @computed_field
    @property
    def annual_revenue(self) -> float:
        """Annual revenue from this unit type."""
        return self.monthly_rent * self.count * 12

    @computed_field
    @property
    def rent_psf(self) -> float:
        """Rent per sqft per month."""
        return self.monthly_rent / self.sqft if self.sqft > 0 else 0


class RentRoll(BaseModel):
    """Complete rent roll for a property."""
    unit_types: List[UnitType] = Field(default_factory=list)

    @computed_field
    @property
    def total_units(self) -> int:
        """Total number of units."""
        return sum(ut.count for ut in self.unit_types)

    @computed_field
    @property
    def total_sqft(self) -> float:
        """Total rentable square footage."""
        return sum(ut.sqft * ut.count for ut in self.unit_types)

    @computed_field
    @property
    def total_annual_revenue(self) -> float:
        """Total annual revenue from all units."""
        return sum(ut.annual_revenue for ut in self.unit_types)

    @computed_field
    @property
    def weighted_avg_rent_psf(self) -> float:
        """Weighted average rent per sqft (weighted by sqft)."""
        total_sqft = self.total_sqft
        if total_sqft == 0:
            return 0
        weighted_sum = sum(ut.rent_psf * ut.sqft * ut.count for ut in self.unit_types)
        return weighted_sum / total_sqft

    @computed_field
    @property
    def avg_unit_sqft(self) -> float:
        """Average sqft per unit."""
        total_units = self.total_units
        if total_units == 0:
            return 800  # default
        return self.total_sqft / total_units


# Default unit types for BC multifamily
DEFAULT_UNIT_TYPES = [
    {"name": "Studio", "bedrooms": 0, "sqft": 450, "monthly_rent": 1400, "count": 0},
    {"name": "1 Bedroom", "bedrooms": 1, "sqft": 650, "monthly_rent": 1800, "count": 0},
    {"name": "2 Bedroom", "bedrooms": 2, "sqft": 900, "monthly_rent": 2400, "count": 0},
    {"name": "3 Bedroom", "bedrooms": 3, "sqft": 1200, "monthly_rent": 3000, "count": 0},
]


def create_default_rent_roll(num_units: int) -> RentRoll:
    """
    Create a default rent roll distributing units across types.

    Distribution: 20% studio, 40% 1BR, 30% 2BR, 10% 3BR
    """
    distribution = [0.20, 0.40, 0.30, 0.10]
    unit_types = []
    allocated = 0

    for i, (default, pct) in enumerate(zip(DEFAULT_UNIT_TYPES, distribution)):
        if i < 3:
            count = int(num_units * pct)
        else:
            # Last type gets remainder
            count = num_units - allocated

        allocated += count
        unit_types.append(UnitType(
            name=default["name"],
            bedrooms=default["bedrooms"],
            sqft=default["sqft"],
            monthly_rent=default["monthly_rent"],
            count=max(0, count)
        ))

    return RentRoll(unit_types=unit_types)
