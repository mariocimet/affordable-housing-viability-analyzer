"""
Multifamily Listing Data Model

Pydantic model for scraped multifamily property listings.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, computed_field


class MultifamilyListing(BaseModel):
    """A multifamily property listing from a commercial real estate site."""

    # Required fields
    id: str = Field(description="Unique identifier for the listing")
    source: str = Field(description="Source site (colliers, cbre, etc.)")
    address: str = Field(description="Property address")
    city: str = Field(description="City name")
    asking_price: float = Field(description="Asking price in CAD")

    # Highly desirable fields
    building_sqft: Optional[float] = Field(default=None, description="Total building square footage")
    num_units: Optional[int] = Field(default=None, description="Number of units")

    # Optional fields
    year_built: Optional[int] = Field(default=None, description="Year constructed")
    cap_rate: Optional[float] = Field(default=None, description="Cap rate as decimal (e.g., 0.05 for 5%)")
    noi: Optional[float] = Field(default=None, description="Net operating income")
    lot_size: Optional[float] = Field(default=None, description="Lot size in sqft or acres")

    # Metadata
    url: Optional[str] = Field(default=None, description="URL to the listing page")
    scraped_at: datetime = Field(default_factory=datetime.now, description="When the listing was scraped")
    listing_status: str = Field(default="active", description="Status: active, sold, pending")

    @computed_field
    @property
    def price_per_unit(self) -> Optional[float]:
        """Calculate price per unit if num_units is available."""
        if self.num_units and self.num_units > 0:
            return self.asking_price / self.num_units
        return None

    @computed_field
    @property
    def price_per_sqft(self) -> Optional[float]:
        """Calculate price per sqft if building_sqft is available."""
        if self.building_sqft and self.building_sqft > 0:
            return self.asking_price / self.building_sqft
        return None

    @computed_field
    @property
    def display_name(self) -> str:
        """Human-readable name for the listing."""
        price_str = f"${self.asking_price / 1_000_000:.1f}M" if self.asking_price >= 1_000_000 else f"${self.asking_price:,.0f}"
        units_str = f" ({self.num_units} units)" if self.num_units else ""
        return f"{self.address}, {self.city} - {price_str}{units_str}"

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'id': self.id,
            'source': self.source,
            'address': self.address,
            'city': self.city,
            'asking_price': self.asking_price,
            'building_sqft': self.building_sqft,
            'num_units': self.num_units,
            'year_built': self.year_built,
            'cap_rate': self.cap_rate,
            'noi': self.noi,
            'price_per_unit': self.price_per_unit,
            'price_per_sqft': self.price_per_sqft,
            'url': self.url,
            'scraped_at': self.scraped_at.isoformat(),
            'listing_status': self.listing_status,
        }

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
