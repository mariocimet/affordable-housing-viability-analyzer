"""
Affordable Housing Viability Analyzer

Streamlit app for detailed analysis of multifamily housing project viability
and income affordability in British Columbia.
"""

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import os
import time
import json
import copy
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.project_viability import PARAM_INFO, ProjectParameters, ProjectViabilityModel, HOUSING_CHARGE_SCENARIOS, calculate_project_irr
from models.param_mapper import analyze_portfolio, listing_to_params
from models.listing import MultifamilyListing
from models.rent_roll import RentRoll, UnitType, DEFAULT_UNIT_TYPES, create_default_rent_roll
from visualizations.viability_space import create_portfolio_scatter


# ============================================
# LOG-NORMAL DISTRIBUTION HELPERS
# ============================================

def fit_lognormal_from_percentiles(median: float, p75: float) -> tuple[float, float]:
    """
    Fit a log-normal distribution from median (50th) and 75th percentile.

    For LogNormal(μ, σ):
    - Median = e^μ
    - P75 = e^(μ + σ * Φ^(-1)(0.75))

    Returns:
        (mu, sigma) parameters for the log-normal distribution
    """
    mu = np.log(median)
    # Φ^(-1)(0.75) ≈ 0.6745
    z_75 = stats.norm.ppf(0.75)
    sigma = (np.log(p75) - mu) / z_75
    return mu, sigma


def get_percentile_value(mu: float, sigma: float, percentile: float) -> float:
    """Get value at a given percentile of log-normal distribution."""
    z = stats.norm.ppf(percentile / 100)
    return np.exp(mu + sigma * z)


def get_income_percentile(mu: float, sigma: float, income: float) -> float:
    """Get the percentile for a given income value."""
    if income <= 0:
        return 0
    z = (np.log(income) - mu) / sigma
    return stats.norm.cdf(z) * 100


def create_income_distribution_figure(
    mu: float,
    sigma: float,
    title: str,
    median: float,
    p75: float,
    rent_line: float = None,
    affordability_ratio: float = 0.30
) -> go.Figure:
    """
    Create a plotly figure showing the income distribution with reference points.

    Args:
        mu, sigma: Log-normal parameters
        title: Chart title
        median: Median income (50th percentile)
        p75: 75th percentile income
        rent_line: Optional - annual rent to show affordability threshold
        affordability_ratio: What fraction of income should go to rent
    """
    # Generate x values for the distribution
    x_min = np.exp(mu - 3 * sigma)
    x_max = np.exp(mu + 3 * sigma)
    x = np.linspace(x_min, x_max, 500)

    # Calculate PDF
    pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    fig = go.Figure()

    # Main distribution curve
    fig.add_trace(go.Scatter(
        x=x, y=pdf,
        mode='lines',
        name='Income Distribution',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)'
    ))

    # Add median line
    median_y = stats.lognorm.pdf(median, s=sigma, scale=np.exp(mu))
    fig.add_trace(go.Scatter(
        x=[median, median],
        y=[0, median_y],
        mode='lines',
        name=f'Median (50th): ${median:,.0f}',
        line=dict(color='#28A745', width=2, dash='dash')
    ))

    # Add 75th percentile line
    p75_y = stats.lognorm.pdf(p75, s=sigma, scale=np.exp(mu))
    fig.add_trace(go.Scatter(
        x=[p75, p75],
        y=[0, p75_y],
        mode='lines',
        name=f'75th %ile: ${p75:,.0f}',
        line=dict(color='#FFC107', width=2, dash='dash')
    ))

    # Add rent affordability line if provided
    if rent_line is not None:
        min_income_required = rent_line / affordability_ratio
        min_income_y = stats.lognorm.pdf(min_income_required, s=sigma, scale=np.exp(mu))
        pct_can_afford = 100 - get_income_percentile(mu, sigma, min_income_required)

        fig.add_trace(go.Scatter(
            x=[min_income_required, min_income_required],
            y=[0, min_income_y],
            mode='lines',
            name=f'Min Income for Rent: ${min_income_required:,.0f}',
            line=dict(color='#DC3545', width=2)
        ))

        # Shade the "can afford" region
        x_afford = x[x >= min_income_required]
        pdf_afford = stats.lognorm.pdf(x_afford, s=sigma, scale=np.exp(mu))
        fig.add_trace(go.Scatter(
            x=np.concatenate([[min_income_required], x_afford, [x_afford[-1]]]),
            y=np.concatenate([[0], pdf_afford, [0]]),
            fill='toself',
            fillcolor='rgba(40, 167, 69, 0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'Can Afford ({pct_can_afford:.1f}%)',
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Annual Household Income ($)',
        yaxis_title='Probability Density',
        xaxis=dict(
            tickformat='$,.0f',
            range=[0, x_max * 0.8]
        ),
        yaxis=dict(showticklabels=False),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig

st.set_page_config(
    page_title="Affordable Housing Analyzer",
    page_icon="🏠",
    layout="wide"
)

st.title("Affordable Housing Viability Analyzer")

st.caption(
    "Evaluate multifamily acquisition opportunities against financial viability criteria (DSCR, cash-on-cash return) "
    "and affordability constraints for target income bands in British Columbia."
)


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_param_input(param_key, prefix=""):
    """Create appropriate input for a parameter."""
    info = PARAM_INFO[param_key]
    key = f"{prefix}{param_key}"

    if info.get('precise_input'):
        value = st.number_input(
            info['name'],
            min_value=float(info['default_range'][0]) * 100,
            max_value=float(info['default_range'][1]) * 100,
            value=float(info['default']) * 100,
            step=0.1,
            format="%.1f",
            key=key,
            help=f"Driver: {info.get('driver', '')}"
        ) / 100
        st.caption(f"Current: {value:.1%}")
    elif '%ile' in info['format']:
        value = st.slider(
            info['name'],
            min_value=int(info['default_range'][0]),
            max_value=int(info['default_range'][1]),
            value=int(info['default']),
            step=1,
            format="%d",
            key=key,
            help=f"Driver: {info.get('driver', '')}"
        )
    elif '%' in info['format']:
        min_val = int(info['default_range'][0] * 100)
        max_val = int(info['default_range'][1] * 100)
        default_val = int(info['default'] * 100)
        value = st.slider(
            info['name'],
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=1,
            format="%d%%",
            key=key,
            help=f"Driver: {info.get('driver', '')}"
        ) / 100
    elif '$' in info['format'] and ',' in info['format']:
        value = st.number_input(
            info['name'],
            min_value=float(info['default_range'][0]),
            max_value=float(info['default_range'][1]),
            value=float(info['default']),
            step=1000.0,
            format="%.0f",
            key=key,
            help=f"Driver: {info.get('driver', '')}"
        )
    elif '$' in info['format']:
        value = st.number_input(
            info['name'],
            min_value=float(info['default_range'][0]),
            max_value=float(info['default_range'][1]),
            value=float(info['default']),
            step=1.0 if info['default'] > 10 else 0.1,
            key=key,
            help=f"Driver: {info.get('driver', '')}"
        )
    else:
        value = st.number_input(
            info['name'],
            min_value=float(info['default_range'][0]),
            max_value=float(info['default_range'][1]),
            value=float(info['default']),
            step=10.0,
            key=key,
            help=f"Driver: {info.get('driver', '')}"
        )

    return value


# ============================================
# DEFAULT TRESAH LISTING
# ============================================

def create_tresah_listing():
    """Create the default Tresah project listing from the viability workbook."""
    return MultifamilyListing(
        id="TRESAH_001",
        source='workbook',
        address="Tresah Co-op Housing",
        city="Victoria",
        asking_price=103000000.0,  # $103M total project cost
        building_sqft=117280.0,     # Residential sqft from rent roll
        num_units=179,
        year_built=2025,            # New construction
        cap_rate=None,
        url='',
    )

# Default rent roll for Tresah (from workbook)
TRESAH_RENT_ROLL = [
    {'name': '1 Bedroom', 'bedrooms': 1, 'sqft': 483.0, 'monthly_rent': 2310.0, 'count': 66},
    {'name': '1 Bedroom Premium', 'bedrooms': 1, 'sqft': 672.0, 'monthly_rent': 2310.0, 'count': 77},
    {'name': '2 Bedroom', 'bedrooms': 2, 'sqft': 877.0, 'monthly_rent': 2915.0, 'count': 22},
    {'name': '2 Bedroom Townhouse', 'bedrooms': 2, 'sqft': 1026.0, 'monthly_rent': 2915.0, 'count': 14},
]


# ============================================
# PER-LISTING SCENARIO PARAMETERS
# ============================================

DEFAULT_SCENARIO_PARAMS = {
    'opex_ratio': 0.25,
    'vacancy_bad_debt': 0.025,  # Combined vacancy and bad debt
    'interest_rate_senior': 0.048,
    'max_ltv': 0.80,
    'inflation_assumption': 0.02,
    'equity_return_required': 0.05,
    'haircut_pct': 0.0,
    'rate_mode': 'moderate',
    'custom_rates': {
        'period_1': 2.5,
        'period_2': 2.5,
        'period_3': 2.25,
        'period_4': 2.0,
        'period_5': 2.0,
    }
}

RATE_PRESETS = {
    'conservative': {'period_1': 2.0, 'period_2': 2.0, 'period_3': 2.0, 'period_4': 2.0, 'period_5': 2.0},
    'moderate': {'period_1': 2.5, 'period_2': 2.5, 'period_3': 2.25, 'period_4': 2.0, 'period_5': 2.0},
    'aggressive': {'period_1': 3.0, 'period_2': 2.75, 'period_3': 2.25, 'period_4': 2.0, 'period_5': 2.0},
}


def get_listing_params(listing_id: str) -> dict:
    """Get scenario parameters for a listing, initializing with defaults if needed."""
    key = f'scenario_params_{listing_id}'
    if key not in st.session_state:
        st.session_state[key] = DEFAULT_SCENARIO_PARAMS.copy()
        st.session_state[key]['custom_rates'] = DEFAULT_SCENARIO_PARAMS['custom_rates'].copy()
    return st.session_state[key]


def set_listing_params(listing_id: str, params: dict):
    """Update scenario parameters for a listing."""
    key = f'scenario_params_{listing_id}'
    st.session_state[key] = params


def build_rate_schedule(custom_rates: dict) -> list:
    """Build 25-year rate schedule from 5-period custom rates."""
    return (
        [custom_rates['period_1'] / 100] * 5 +
        [custom_rates['period_2'] / 100] * 5 +
        [custom_rates['period_3'] / 100] * 5 +
        [custom_rates['period_4'] / 100] * 5 +
        [custom_rates['period_5'] / 100] * 5
    )


def build_fixed_params(scenario_params: dict) -> dict:
    """Build the fixed_params dict from scenario parameters."""
    rate_schedule = build_rate_schedule(scenario_params['custom_rates'])
    return {
        'opex_ratio': scenario_params['opex_ratio'],
        'housing_charge_increase': rate_schedule[0],
        'interest_rate_senior': scenario_params['interest_rate_senior'],
        'interest_rate_secondary': PARAM_INFO['interest_rate_secondary']['default'],
        'max_ltv': scenario_params['max_ltv'],
        'risk_free_rate': PARAM_INFO['risk_free_rate']['default'],
        'inflation_assumption': scenario_params['inflation_assumption'],
        'affordability_ratio': 0.30,
        'target_percentile': 50,
        'equity_return_required': scenario_params['equity_return_required'],
        'acquisition_cost_psf': PARAM_INFO['acquisition_cost_psf']['default'],
        'rent_psf': PARAM_INFO['rent_psf']['default'],
    }


# ============================================
# PERSISTENCE - SAVE/LOAD WITH UNDO HISTORY
# ============================================

SAVED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'saved_listings.json')
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'history.json')
MAX_HISTORY = 50  # Keep last 50 changes


from datetime import datetime, date

def json_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_current_state():
    """Get current state as a dictionary."""
    data = {
        'listings': [],
        'rent_rolls': {},
        'income_distributions': {
            'couples': st.session_state.get('income_dist_couples', {}),
            'families': st.session_state.get('income_dist_families', {}),
        }
    }

    for listing in st.session_state.get('listings', []):
        listing_dict = listing.model_dump()
        data['listings'].append(listing_dict)
        rent_roll_key = f'rent_roll_{listing.id}'
        if rent_roll_key in st.session_state:
            data['rent_rolls'][listing.id] = st.session_state[rent_roll_key]

    return data


def load_history():
    """Load history from file."""
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, 'r') as f:
            return json.load(f)
    except:
        return []


def save_history(history):
    """Save history to file."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history[-MAX_HISTORY:], f, indent=2, default=json_serializer)


def save_with_history(description="Change"):
    """Save current state and add to history."""
    state = get_current_state()

    # Load existing history
    history = load_history()

    # Add new entry with timestamp and description
    entry = {
        'timestamp': datetime.now().isoformat(),
        'description': description,
        'state': state
    }
    history.append(entry)

    # Save history (trimmed to MAX_HISTORY)
    save_history(history)

    # Also save current state
    os.makedirs(os.path.dirname(SAVED_DATA_PATH), exist_ok=True)
    with open(SAVED_DATA_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=json_serializer)

    return True


def undo_last_change():
    """Revert to previous state. Returns True if successful."""
    history = load_history()

    if len(history) < 2:
        return False, "No previous state to restore"

    # Remove current state
    history.pop()

    # Get previous state
    previous = history[-1]

    # Save updated history
    save_history(history)

    # Also update current state file
    with open(SAVED_DATA_PATH, 'w') as f:
        json.dump(previous['state'], f, indent=2, default=json_serializer)

    return True, previous['description']


def apply_state(state):
    """Apply a state dictionary to session state."""
    # Increment widget version to force refresh
    st.session_state['_widget_version'] = st.session_state.get('_widget_version', 0) + 1

    # Clear existing rent roll keys AND widget keys
    keys_to_remove = [k for k in st.session_state.keys()
                      if k.startswith('rent_roll_') or k.startswith('rr_') or k.startswith('ld_')]
    for k in keys_to_remove:
        del st.session_state[k]

    # Load listings - handle both dict and MultifamilyListing objects
    listings = []
    for item in state.get('listings', []):
        if isinstance(item, MultifamilyListing):
            listings.append(item)
        else:
            listings.append(MultifamilyListing(**item))
    st.session_state.listings = listings

    # Load rent rolls
    for listing_id, rent_roll in state.get('rent_rolls', {}).items():
        st.session_state[f'rent_roll_{listing_id}'] = rent_roll

    # Load income distributions
    if state.get('income_distributions', {}).get('couples'):
        st.session_state.income_dist_couples = state['income_distributions']['couples']
    if state.get('income_distributions', {}).get('families'):
        st.session_state.income_dist_families = state['income_distributions']['families']


def load_listings_from_file():
    """Load listings and rent rolls from JSON file."""
    if not os.path.exists(SAVED_DATA_PATH):
        return None

    try:
        with open(SAVED_DATA_PATH, 'r') as f:
            data = json.load(f)

        listings = []
        for listing_dict in data.get('listings', []):
            listing = MultifamilyListing(**listing_dict)
            listings.append(listing)

        rent_rolls = data.get('rent_rolls', {})
        income_dists = data.get('income_distributions', {})

        return {
            'listings': listings,
            'rent_rolls': rent_rolls,
            'income_distributions': income_dists
        }
    except Exception as e:
        st.error(f"Error loading saved data: {e}")
        return None


# Alias for backward compatibility
def save_listings_to_file():
    """Save current state (wrapper for save_with_history)."""
    return save_with_history("Auto-save")


# ============================================
# BC HOUSING INCOME LIMITS
# ============================================

@st.cache_data
def load_bc_housing_income_limits():
    """Load BC Housing income limits by year and family type."""
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'bc_housing_income_limits.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def get_bc_housing_income(year: int = 2025, family_type: str = 'couples_without_children',
                          income_level: str = 'median') -> float:
    """
    Get BC Housing income limit.

    Args:
        year: Reference year (2024, 2025, or 2026)
        family_type: 'couples_without_children' or 'families_with_children'
        income_level: 'median' (50th percentile) or '75th_percentile'

    Returns:
        Income limit in dollars
    """
    df = load_bc_housing_income_limits()
    if df.empty:
        # Fallback defaults
        defaults = {
            ('couples_without_children', 'median'): 85870,
            ('couples_without_children', '75th_percentile'): 136210,
            ('families_with_children', 'median'): 138770,
            ('families_with_children', '75th_percentile'): 198230,
        }
        return defaults.get((family_type, income_level), 100000)

    # Find matching row
    mask = (df['year'] == year) & (df['family_type'] == family_type) & (df['income_level'] == income_level)
    matches = df[mask]

    if not matches.empty:
        return float(matches.iloc[0]['income_limit'])

    # Try closest year
    available_years = df['year'].unique()
    closest_year = min(available_years, key=lambda y: abs(y - year))
    mask = (df['year'] == closest_year) & (df['family_type'] == family_type) & (df['income_level'] == income_level)
    matches = df[mask]

    if not matches.empty:
        return float(matches.iloc[0]['income_limit'])

    return 100000  # Ultimate fallback

def get_income_for_unit_type(sqft: float, year: int = 2026, income_level: str = 'median') -> tuple[float, str]:
    """
    Get appropriate income threshold based on unit size.

    BC Housing methodology:
    - Units < 2BR (typically < 700 sqft): couples without children
    - Units >= 2BR (typically >= 700 sqft): families with children

    Args:
        sqft: Unit square footage
        year: Reference year
        income_level: 'median' or '75th_percentile'

    Returns:
        Tuple of (income_limit, family_type_label)
    """
    # Threshold: ~700 sqft is roughly the boundary between 1BR and 2BR
    TWO_BR_THRESHOLD = 700

    if sqft < TWO_BR_THRESHOLD:
        family_type = 'couples_without_children'
        label = 'Couples (no children)'
    else:
        family_type = 'families_with_children'
        label = 'Families (with children)'

    income = get_bc_housing_income(year, family_type, income_level)
    return income, label


# ============================================
# INITIALIZE LISTINGS
# ============================================

if 'listings' not in st.session_state:
    # Try to load saved listings first
    saved_data = load_listings_from_file()
    if saved_data and saved_data['listings']:
        st.session_state.listings = saved_data['listings']
        # Restore rent rolls
        for listing_id, rent_roll in saved_data['rent_rolls'].items():
            st.session_state[f'rent_roll_{listing_id}'] = rent_roll
        # Restore income distributions
        if saved_data.get('income_distributions', {}).get('couples'):
            st.session_state.income_dist_couples = saved_data['income_distributions']['couples']
        if saved_data.get('income_distributions', {}).get('families'):
            st.session_state.income_dist_families = saved_data['income_distributions']['families']
    else:
        # Initialize with Tresah as default
        st.session_state.listings = [create_tresah_listing()]
        st.session_state['rent_roll_TRESAH_001'] = TRESAH_RENT_ROLL


# ============================================
# INITIALIZE INCOME DISTRIBUTION SESSION STATE
# ============================================

# Load BC Housing defaults
BC_HOUSING_DEFAULTS = {
    'couples_without_children': {
        'median': get_bc_housing_income(2026, 'couples_without_children', 'median'),
        '75th_percentile': get_bc_housing_income(2026, 'couples_without_children', '75th_percentile'),
    },
    'families_with_children': {
        'median': get_bc_housing_income(2026, 'families_with_children', 'median'),
        '75th_percentile': get_bc_housing_income(2026, 'families_with_children', '75th_percentile'),
    }
}

# Initialize income distribution parameters in session state
if 'income_dist_couples' not in st.session_state:
    st.session_state.income_dist_couples = BC_HOUSING_DEFAULTS['couples_without_children'].copy()
if 'income_dist_families' not in st.session_state:
    st.session_state.income_dist_families = BC_HOUSING_DEFAULTS['families_with_children'].copy()


# ============================================
# MAIN CONTENT: TABS
# ============================================

tab_viability, tab_income = st.tabs(["Project Viability", "Income Distributions"])

# ============================================
# TAB 2: INCOME DISTRIBUTIONS
# ============================================
with tab_income:
    st.header("BC Household Income Distributions")
    st.caption(
        "Configure income distributions using BC Housing reference points. "
        "These distributions are used to calculate affordability in the Project Viability tab."
    )

    # Explanation of methodology
    with st.expander("How the distribution is computed", expanded=False):
        st.markdown("""
        **Log-Normal Distribution Fitting**

        BC Housing provides two reference points from the StatsCan T1 Family File:
        - **Median (50th percentile)**: Half of households earn less than this
        - **75th percentile**: 75% of households earn less than this

        We fit a log-normal distribution using these two points:

        1. For a log-normal distribution with parameters μ and σ:
           - Median = e^μ
           - Any percentile P = e^(μ + σ × Φ⁻¹(p))

        2. Given median M and 75th percentile P75:
           - μ = ln(M)
           - σ = (ln(P75) - ln(M)) / Φ⁻¹(0.75)
           - Where Φ⁻¹(0.75) ≈ 0.6745

        This allows us to calculate any percentile of the income distribution.
        """)

    st.divider()

    # Two columns for the two family types
    col_couples, col_families = st.columns(2)

    # ---- Couples Without Children ----
    with col_couples:
        st.subheader("Couples Without Children")
        st.caption("Used for Studio & 1BR units (0-1 bedrooms)")

        # Editable inputs
        couples_median = st.number_input(
            "Median Income (50th %ile)",
            min_value=30000,
            max_value=300000,
            value=int(st.session_state.income_dist_couples['median']),
            step=1000,
            key="couples_median_input",
            help="BC Housing Low & Moderate Income threshold"
        )
        couples_p75 = st.number_input(
            "75th Percentile Income",
            min_value=50000,
            max_value=500000,
            value=int(st.session_state.income_dist_couples['75th_percentile']),
            step=1000,
            key="couples_p75_input",
            help="BC Housing Middle Income threshold"
        )

        # Update session state - only save if values changed
        old_median = st.session_state.income_dist_couples.get('median')
        old_p75 = st.session_state.income_dist_couples.get('75th_percentile')
        if couples_median != old_median or couples_p75 != old_p75:
            st.session_state.income_dist_couples['median'] = couples_median
            st.session_state.income_dist_couples['75th_percentile'] = couples_p75
            save_with_history("Updated couples income distribution")

        # Show BC Housing reference
        bc_couples_med = BC_HOUSING_DEFAULTS['couples_without_children']['median']
        bc_couples_p75 = BC_HOUSING_DEFAULTS['couples_without_children']['75th_percentile']
        st.caption(f"BC Housing 2026: Median ${bc_couples_med:,.0f} | 75th %ile ${bc_couples_p75:,.0f}")

        # Reset button
        if st.button("Reset to BC Housing", key="reset_couples"):
            st.session_state.income_dist_couples = BC_HOUSING_DEFAULTS['couples_without_children'].copy()
            save_with_history("Reset couples income to BC Housing defaults")
            st.rerun()

        # Fit distribution and show percentiles
        mu_c, sigma_c = fit_lognormal_from_percentiles(couples_median, couples_p75)

        st.markdown("**Derived Percentiles:**")
        percentiles_to_show = [10, 25, 50, 75, 90, 95]
        perc_values_c = {p: get_percentile_value(mu_c, sigma_c, p) for p in percentiles_to_show}
        perc_df_c = pd.DataFrame({
            'Percentile': [f"{p}th" for p in percentiles_to_show],
            'Income': [f"${perc_values_c[p]:,.0f}" for p in percentiles_to_show]
        })
        st.dataframe(perc_df_c, use_container_width=True, hide_index=True)

        # Distribution chart
        fig_couples = create_income_distribution_figure(
            mu_c, sigma_c,
            "Couples Without Children",
            couples_median, couples_p75
        )
        st.plotly_chart(fig_couples, use_container_width=True)

    # ---- Families With Children ----
    with col_families:
        st.subheader("Families With Children")
        st.caption("Used for 2BR+ units (2+ bedrooms)")

        # Editable inputs
        families_median = st.number_input(
            "Median Income (50th %ile)",
            min_value=30000,
            max_value=400000,
            value=int(st.session_state.income_dist_families['median']),
            step=1000,
            key="families_median_input",
            help="BC Housing Low & Moderate Income threshold"
        )
        families_p75 = st.number_input(
            "75th Percentile Income",
            min_value=50000,
            max_value=600000,
            value=int(st.session_state.income_dist_families['75th_percentile']),
            step=1000,
            key="families_p75_input",
            help="BC Housing Middle Income threshold"
        )

        # Update session state - only save if values changed
        old_median = st.session_state.income_dist_families.get('median')
        old_p75 = st.session_state.income_dist_families.get('75th_percentile')
        if families_median != old_median or families_p75 != old_p75:
            st.session_state.income_dist_families['median'] = families_median
            st.session_state.income_dist_families['75th_percentile'] = families_p75
            save_with_history("Updated families income distribution")

        # Show BC Housing reference
        bc_families_med = BC_HOUSING_DEFAULTS['families_with_children']['median']
        bc_families_p75 = BC_HOUSING_DEFAULTS['families_with_children']['75th_percentile']
        st.caption(f"BC Housing 2026: Median ${bc_families_med:,.0f} | 75th %ile ${bc_families_p75:,.0f}")

        # Reset button
        if st.button("Reset to BC Housing", key="reset_families"):
            st.session_state.income_dist_families = BC_HOUSING_DEFAULTS['families_with_children'].copy()
            save_with_history("Reset families income to BC Housing defaults")
            st.rerun()

        # Fit distribution and show percentiles
        mu_f, sigma_f = fit_lognormal_from_percentiles(families_median, families_p75)

        st.markdown("**Derived Percentiles:**")
        perc_values_f = {p: get_percentile_value(mu_f, sigma_f, p) for p in percentiles_to_show}
        perc_df_f = pd.DataFrame({
            'Percentile': [f"{p}th" for p in percentiles_to_show],
            'Income': [f"${perc_values_f[p]:,.0f}" for p in percentiles_to_show]
        })
        st.dataframe(perc_df_f, use_container_width=True, hide_index=True)

        # Distribution chart
        fig_families = create_income_distribution_figure(
            mu_f, sigma_f,
            "Families With Children",
            families_median, families_p75
        )
        st.plotly_chart(fig_families, use_container_width=True)

    st.divider()

    # ---- Affordability Animation Section ----
    st.subheader("Affordability Over Time")
    st.caption(
        "See how the affordable band shifts over 25 years as incomes grow by inflation "
        "and housing charges increase by the selected rate schedule."
    )

    # Animation uses default parameters (or from first listing if available)
    anim_inflation = DEFAULT_SCENARIO_PARAMS['inflation_assumption']
    anim_rate_schedule = build_rate_schedule(DEFAULT_SCENARIO_PARAMS['custom_rates'])
    affordability_ratio = 0.30

    # If there's a selected listing in the viability tab, use its parameters
    if 'listings' in st.session_state and st.session_state.listings:
        first_listing = st.session_state.listings[0]
        first_params = get_listing_params(first_listing.id)
        anim_inflation = first_params['inflation_assumption']
        anim_rate_schedule = build_rate_schedule(first_params['custom_rates'])

    # Get rent roll data from session state if available
    animation_rent = None
    animation_sqft = 700  # default to families

    if 'listings' in st.session_state and st.session_state.listings:
        # Use first listing's rent roll weighted average
        first_listing = st.session_state.listings[0]
        rent_roll_key = f'rent_roll_{first_listing.id}'
        if rent_roll_key in st.session_state:
            rent_roll_data = st.session_state[rent_roll_key]
            total_units = sum(u['count'] for u in rent_roll_data)
            if total_units > 0:
                weighted_rent = sum(u['monthly_rent'] * u['count'] for u in rent_roll_data) / total_units
                weighted_sqft = sum(u['sqft'] * u['count'] for u in rent_roll_data) / total_units
                animation_rent = weighted_rent * 12  # annual rent
                animation_sqft = weighted_sqft

    # Manual rent input for animation
    anim_col1, anim_col2 = st.columns(2)
    with anim_col1:
        example_rent = st.number_input(
            "Example Monthly Rent ($)",
            min_value=500,
            max_value=10000,
            value=int(animation_rent / 12) if animation_rent else 2500,
            step=100,
            key="animation_rent_input",
            help="The rent to show in the animation"
        )
    with anim_col2:
        unit_size_option = st.radio(
            "Unit Type",
            options=["Studio/1BR (Couples)", "2BR+ (Families)"],
            index=0 if animation_sqft < 700 else 1,
            key="animation_unit_type",
            horizontal=True
        )

    use_families = unit_size_option == "2BR+ (Families)"

    # Get the appropriate distribution
    if use_families:
        anim_median = st.session_state.income_dist_families['median']
        anim_p75 = st.session_state.income_dist_families['75th_percentile']
        dist_title = "Families With Children"
    else:
        anim_median = st.session_state.income_dist_couples['median']
        anim_p75 = st.session_state.income_dist_couples['75th_percentile']
        dist_title = "Couples Without Children"

    # Animation controls
    anim_col1, anim_col2 = st.columns([3, 1])
    with anim_col1:
        year_slider = st.slider(
            "Year",
            min_value=0,
            max_value=25,
            value=0,
            key="affordability_year_slider",
            help="Drag to see affordability change over time, or use Play button"
        )
    with anim_col2:
        play_animation = st.button("▶ Play Animation", key="play_animation_btn", use_container_width=True)

    # Calculate values for the selected year
    annual_rent_year0 = example_rent * 12

    # Rent grows by housing charge schedule
    rent_growth = 1.0
    for y in range(year_slider):
        rent_growth *= (1 + anim_rate_schedule[min(y, len(anim_rate_schedule) - 1)])
    annual_rent_year_n = annual_rent_year0 * rent_growth

    # Income grows by inflation
    income_growth = (1 + anim_inflation) ** year_slider
    median_year_n = anim_median * income_growth
    p75_year_n = anim_p75 * income_growth

    # Fit distribution for this year
    mu_anim, sigma_anim = fit_lognormal_from_percentiles(median_year_n, p75_year_n)

    # Create the visualization
    fig_anim = create_income_distribution_figure(
        mu_anim, sigma_anim,
        f"Year {year_slider}: {dist_title}",
        median_year_n, p75_year_n,
        rent_line=annual_rent_year_n,
        affordability_ratio=affordability_ratio
    )
    chart_placeholder = st.empty()
    chart_placeholder.plotly_chart(fig_anim, use_container_width=True, key="affordability_animation_chart")

    # Show metrics for the selected year
    min_income_required = annual_rent_year_n / affordability_ratio
    pct_can_afford = 100 - get_income_percentile(mu_anim, sigma_anim, min_income_required)

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Year", f"{year_slider}")
    with metric_cols[1]:
        st.metric("Monthly Rent", f"${annual_rent_year_n/12:,.0f}")
    with metric_cols[2]:
        st.metric("Min Income Required", f"${min_income_required:,.0f}")
    with metric_cols[3]:
        st.metric("Median Income", f"${median_year_n:,.0f}")
    with metric_cols[4]:
        st.metric("% Can Afford", f"{pct_can_afford:.1f}%")

    # Play animation
    if play_animation:
        for year in range(26):
            # Calculate values for this year
            rent_growth = 1.0
            for y in range(year):
                rent_growth *= (1 + anim_rate_schedule[min(y, len(anim_rate_schedule) - 1)])
            annual_rent = annual_rent_year0 * rent_growth

            income_growth = (1 + anim_inflation) ** year
            median_y = anim_median * income_growth
            p75_y = anim_p75 * income_growth

            mu_y, sigma_y = fit_lognormal_from_percentiles(median_y, p75_y)

            fig_y = create_income_distribution_figure(
                mu_y, sigma_y,
                f"Year {year}: {dist_title}",
                median_y, p75_y,
                rent_line=annual_rent,
                affordability_ratio=affordability_ratio
            )
            # Use empty() to clear and reuse the placeholder
            chart_placeholder.empty()
            chart_placeholder.plotly_chart(fig_y, use_container_width=True, key=f"anim_frame_{year}")
            time.sleep(0.3)


# ============================================
# TAB 1: PROJECT VIABILITY
# ============================================
with tab_viability:
    # ---- Add Listing Section (Collapsible) ----
    with st.expander("Add New Listing", expanded=False):
        add_cols = st.columns([2, 1.5, 1.5, 1, 1, 1])
        with add_cols[0]:
            ml_address = st.text_input("Address", placeholder="123 Main St", key="add_address")
        with add_cols[1]:
            ml_city = st.text_input("City", placeholder="Vancouver", key="add_city")
        with add_cols[2]:
            ml_price = st.number_input("Asking Price ($)", min_value=100000, max_value=500000000, value=5000000, step=100000, key="add_price")
        with add_cols[3]:
            ml_sqft = st.number_input("Sq Ft", min_value=0, max_value=1000000, value=0, step=1000, help="Leave 0 to estimate from units", key="add_sqft")
        with add_cols[4]:
            ml_units = st.number_input("Units", min_value=1, max_value=1000, value=20, step=1, key="add_units")
        with add_cols[5]:
            st.write("")  # Spacer
            if st.button("Add Listing", type="primary", use_container_width=True):
                if ml_address and ml_city:
                    new_id = f"LISTING_{len(st.session_state.listings) + 1:03d}"
                    new_listing = MultifamilyListing(
                        id=new_id,
                        source='manual',
                        address=ml_address,
                        city=ml_city,
                        asking_price=float(ml_price),
                        building_sqft=float(ml_sqft) if ml_sqft > 0 else None,
                        num_units=int(ml_units),
                        year_built=None,
                        cap_rate=None,
                        url='',
                    )
                    st.session_state.listings.append(new_listing)
                    save_with_history(f"Added listing: {ml_address}")
                    st.session_state['_select_listing'] = f"{ml_address} - {ml_city}"
                    st.rerun()
                else:
                    st.error("Address and City are required")

    if not st.session_state.listings:
        st.info("No listings loaded. Add a listing above to get started.")
    else:
        # Project selector row
        st.subheader("Select Project")
        listing_options = {l.display_name: l for l in st.session_state.listings}
        option_list = list(listing_options.keys())

        # Check if we should select a specific listing (e.g., after adding new one)
        if '_select_listing' in st.session_state:
            target = st.session_state.pop('_select_listing')
            if target in option_list:
                st.session_state['deep_dive_select'] = target

        # Undo history
        history = load_history()
        history_count = len(history)

        select_col, delete_col, undo_col = st.columns([4, 0.8, 0.8])
        with select_col:
            selected_name = st.selectbox(
                "Choose a listing to analyze:",
                options=option_list,
                key="deep_dive_select"
            )
        with delete_col:
            st.write("")  # Spacer for alignment
            if st.button("Delete", type="secondary", use_container_width=True):
                if selected_name:
                    listing_to_delete = listing_options[selected_name]
                    st.session_state.listings = [
                        l for l in st.session_state.listings if l.id != listing_to_delete.id
                    ]
                    save_with_history(f"Deleted listing: {listing_to_delete.address}")
                    st.rerun()
        with undo_col:
            st.write("")  # Spacer for alignment
            if st.button("Undo", type="secondary", use_container_width=True, disabled=(history_count < 2)):
                success, desc = undo_last_change()
                if success:
                    saved_data = load_listings_from_file()
                    if saved_data:
                        apply_state(saved_data)
                    st.rerun()

        if selected_name:
            listing = listing_options[selected_name]

            # Get per-listing scenario parameters
            scenario_params = get_listing_params(listing.id)
            assumptions = {'opex_ratio': scenario_params['opex_ratio']}

            # Get params for this listing
            params = listing_to_params(listing, assumptions)

            st.divider()

            # ---- Row 1: Listing Details (Editable) ----
            st.subheader("Listing Details")
            wv = st.session_state.get('_widget_version', 0)

            detail_cols = st.columns([2, 1.5, 1.5, 1, 1.5, 1])
            with detail_cols[0]:
                new_address = st.text_input("Address", value=listing.address, key=f"ld_addr_{listing.id}_v{wv}")
            with detail_cols[1]:
                new_city = st.text_input("City", value=listing.city, key=f"ld_city_{listing.id}_v{wv}")
            with detail_cols[2]:
                new_price = st.number_input("Price ($)", value=int(listing.asking_price), min_value=100000, max_value=500000000, step=100000, key=f"ld_price_{listing.id}_v{wv}")
            with detail_cols[3]:
                new_units = st.number_input("Units", value=int(listing.num_units or 20), min_value=1, max_value=1000, step=1, key=f"ld_units_{listing.id}_v{wv}")
            with detail_cols[4]:
                new_sqft = st.number_input("Residential Sq Ft", value=int(listing.building_sqft or 0), min_value=0, max_value=1000000, step=1000, key=f"ld_sqft_{listing.id}_v{wv}")
            with detail_cols[5]:
                new_year = st.number_input("Year", value=int(listing.year_built or 2000), min_value=1900, max_value=2030, step=1, key=f"ld_year_{listing.id}_v{wv}")

            # Check if listing details changed
            listing_changed = (
                new_address != listing.address or
                new_city != listing.city or
                new_price != listing.asking_price or
                new_units != listing.num_units or
                new_sqft != (listing.building_sqft or 0) or
                new_year != (listing.year_built or 2000)
            )

            if listing_changed:
                # Update the listing in session state
                for i, l in enumerate(st.session_state.listings):
                    if l.id == listing.id:
                        st.session_state.listings[i] = MultifamilyListing(
                            id=listing.id,
                            source=listing.source,
                            address=new_address,
                            city=new_city,
                            asking_price=float(new_price),
                            building_sqft=float(new_sqft) if new_sqft > 0 else None,
                            num_units=int(new_units),
                            year_built=int(new_year) if new_year else None,
                            cap_rate=listing.cap_rate,
                            url=listing.url,
                        )
                        # Update listing reference for rest of page
                        listing = st.session_state.listings[i]
                        break
                save_with_history(f"Updated listing details: {new_address}")
                # Refresh params with new listing data
                params = listing_to_params(listing, assumptions)

            # ---- Scenario Parameters ----
            st.divider()
            st.subheader("Scenario Parameters")

            import altair as alt

            # ============================================
            # 1. ACQUISITION - Starting constraint
            # ============================================
            st.markdown("**Acquisition**")
            st.caption("Model purchase price negotiation. The effective price drives all downstream calculations.")

            haircut_pct = scenario_params.get('haircut_pct', 0.0)

            new_haircut = st.slider(
                "Haircut %",
                min_value=0.0, max_value=50.0,
                value=haircut_pct,
                step=1.0, format="%.0f%%",
                key=f"haircut_{listing.id}_v{wv}",
            )
            if new_haircut != haircut_pct:
                scenario_params['haircut_pct'] = new_haircut
                haircut_pct = new_haircut

            effective_price = listing.asking_price * (1 - haircut_pct / 100)
            price_reduction = listing.asking_price - effective_price

            acq_metrics = st.columns(3)
            with acq_metrics[0]:
                st.metric("Asking Price", f"${listing.asking_price:,.0f}")
            with acq_metrics[1]:
                st.metric("Effective Price", f"${effective_price:,.0f}",
                         delta=f"-${price_reduction:,.0f}" if price_reduction > 0 else None,
                         delta_color="normal" if price_reduction > 0 else "off")
            with acq_metrics[2]:
                if params.get('building_sqft'):
                    effective_psf = effective_price / params['building_sqft']
                    st.metric("Effective $/SqFt", f"${effective_psf:.0f}")
                else:
                    st.metric("Effective $/SqFt", "N/A")

            # ============================================
            # 2. CAPITAL STRUCTURE - Financing and target return
            # ============================================
            st.markdown("**Capital Structure**")
            st.caption("Debt/equity split, cost of capital, and required return on equity.")

            cap_cols = st.columns(3)
            with cap_cols[0]:
                new_ltv = st.number_input(
                    "LTV (%)",
                    min_value=50.0, max_value=95.0,
                    value=scenario_params['max_ltv'] * 100,
                    step=1.0, format="%.0f",
                    key=f"ltv_{listing.id}_v{wv}",
                    help="Loan-to-value ratio. Determines debt vs equity split."
                ) / 100

            with cap_cols[1]:
                new_interest = st.number_input(
                    "Interest Rate (%)",
                    min_value=2.0, max_value=10.0,
                    value=scenario_params['interest_rate_senior'] * 100,
                    step=0.1, format="%.2f",
                    key=f"interest_{listing.id}_v{wv}",
                    help="Senior debt interest rate (annual)."
                ) / 100

            with cap_cols[2]:
                new_target_return = st.number_input(
                    "Target IRR (%)",
                    min_value=0.0, max_value=15.0,
                    value=scenario_params['equity_return_required'] * 100,
                    step=0.5, format="%.1f",
                    key=f"target_return_{listing.id}_v{wv}",
                    help="Minimum acceptable 25-year IRR on equity. Project viable if DSCR >= 1.0 AND meets target."
                ) / 100

            # Show resulting capital stack
            debt_amount = effective_price * new_ltv
            equity_amount = effective_price * (1 - new_ltv)
            st.markdown(f"""
            | | Amount | % |
            |---|---:|---:|
            | **Debt** | ${debt_amount:,.0f} | {new_ltv:.0%} |
            | **Equity** | ${equity_amount:,.0f} | {1-new_ltv:.0%} |
            | **Total** | ${effective_price:,.0f} | 100% |
            """)

            # ============================================
            # 3. OPERATIONS - How it performs
            # ============================================
            st.markdown("**Operations**")
            st.caption("Vacancy/bad debt and operating expenses. Subtracted from gross income.")

            ops_cols = st.columns(2)
            with ops_cols[0]:
                new_vacancy_bad_debt = st.number_input(
                    "Vacancy & Bad Debt (%)",
                    min_value=0.0, max_value=25.0,
                    value=scenario_params.get('vacancy_bad_debt', 0.025) * 100,
                    step=0.5, format="%.1f",
                    key=f"vacancy_bad_debt_{listing.id}_v{wv}",
                    help="Combined vacancy and bad debt as % of gross income."
                ) / 100

            with ops_cols[1]:
                new_opex = st.number_input(
                    "OpEx Ratio (%)",
                    min_value=15.0, max_value=55.0,
                    value=scenario_params['opex_ratio'] * 100,
                    step=1.0, format="%.0f",
                    key=f"opex_{listing.id}_v{wv}",
                    help="Operating expenses as % of effective gross income."
                ) / 100

            # Update scenario params if changed
            if (new_interest != scenario_params['interest_rate_senior'] or
                new_ltv != scenario_params['max_ltv'] or
                new_opex != scenario_params['opex_ratio'] or
                new_vacancy_bad_debt != scenario_params.get('vacancy_bad_debt', 0.025) or
                new_target_return != scenario_params['equity_return_required']):
                scenario_params['interest_rate_senior'] = new_interest
                scenario_params['max_ltv'] = new_ltv
                scenario_params['opex_ratio'] = new_opex
                scenario_params['vacancy_bad_debt'] = new_vacancy_bad_debt
                scenario_params['equity_return_required'] = new_target_return
                assumptions['opex_ratio'] = new_opex

            # Apply scenario params to calculations
            rate_schedule = build_rate_schedule(scenario_params['custom_rates'])
            fixed_params = build_fixed_params(scenario_params)

            # Extract commonly used params for convenience
            opex_ratio = fixed_params['opex_ratio']
            vacancy_bad_debt = scenario_params.get('vacancy_bad_debt', 0.025)
            interest_rate_senior = fixed_params['interest_rate_senior']
            max_ltv = fixed_params['max_ltv']
            inflation_assumption = fixed_params['inflation_assumption']
            equity_return_required = fixed_params['equity_return_required']

            # Apply haircut to params
            if params.get('acquisition_cost_psf') is not None and params.get('building_sqft'):
                params['acquisition_cost_psf'] = effective_price / params['building_sqft']
                params['effective_price'] = effective_price
                params['haircut_pct'] = haircut_pct

            # Check if we can analyze
            if params.get('acquisition_cost_psf') is None:
                st.error("Cannot analyze this listing - missing required data (sqft)")
            else:
                building_sqft = params.get('building_sqft', 10000)

                st.divider()

                # ---- Rent Roll Section ----
                st.subheader("Rent Roll")
                st.caption("Enter actual rents by unit type. Revenue and affordability are calculated from this data.")

                # Initialize rent roll in session state if needed
                rent_roll_key = f'rent_roll_{listing.id}'
                if rent_roll_key not in st.session_state:
                    num_units = listing.num_units or 20
                    default_roll = create_default_rent_roll(num_units)
                    st.session_state[rent_roll_key] = [ut.model_dump() for ut in default_roll.unit_types]

                # Display editable rent roll table
                rent_roll_data = st.session_state[rent_roll_key]

                # Get income thresholds for quick-set buttons
                couples_income = st.session_state.get('income_dist_couples', BC_HOUSING_DEFAULTS['couples_without_children'])
                families_income = st.session_state.get('income_dist_families', BC_HOUSING_DEFAULTS['families_with_children'])
                affordability_ratio = 0.30

                # Header row
                header_cols = st.columns([2, 0.7, 1, 1.5, 1.2, 1, 1.3])
                with header_cols[0]:
                    st.markdown("**Unit Type**")
                with header_cols[1]:
                    st.markdown("**Beds**")
                with header_cols[2]:
                    st.markdown("**Sq Ft**")
                with header_cols[3]:
                    st.markdown("**Monthly Rent**")
                with header_cols[4]:
                    st.markdown("**Set to**")
                with header_cols[5]:
                    st.markdown("**# Units**")
                with header_cols[6]:
                    st.markdown("**Annual Revenue**")

                # Editable rows - use version in key to force refresh on undo
                updated_units = []
                wv = st.session_state.get('_widget_version', 0)
                for i, ut in enumerate(rent_roll_data):
                    # Handle legacy data without bedrooms field
                    default_beds = ut.get('bedrooms', 1)

                    # Calculate affordable rents based on bedroom count
                    if default_beds < 2:
                        low_mod_rent = int(couples_income['median'] * affordability_ratio / 12)
                        middle_rent = int(couples_income['75th_percentile'] * affordability_ratio / 12)
                    else:
                        low_mod_rent = int(families_income['median'] * affordability_ratio / 12)
                        middle_rent = int(families_income['75th_percentile'] * affordability_ratio / 12)

                    cols = st.columns([2, 0.7, 1, 1.5, 1.2, 1, 1.3])
                    with cols[0]:
                        name = st.text_input("Type", value=ut['name'], key=f"rr_name_{listing.id}_{i}_v{wv}", label_visibility="collapsed")
                    with cols[1]:
                        bedrooms = st.number_input("Beds", value=int(default_beds), min_value=0, max_value=5, step=1, key=f"rr_beds_{listing.id}_{i}_v{wv}", label_visibility="collapsed")
                    with cols[2]:
                        sqft = st.number_input("SqFt", value=int(ut['sqft']), min_value=100, max_value=3000, step=50, key=f"rr_sqft_{listing.id}_{i}_v{wv}", label_visibility="collapsed")
                    with cols[3]:
                        rent = st.number_input("Rent", value=int(ut['monthly_rent']), min_value=0, max_value=10000, step=50, key=f"rr_rent_{listing.id}_{i}_v{wv}", label_visibility="collapsed")
                    with cols[4]:
                        # Quick-set buttons
                        btn_cols = st.columns(2)
                        with btn_cols[0]:
                            if st.button(f"L/M", key=f"set_lm_{listing.id}_{i}_v{wv}", help=f"Set to Low/Mod: ${low_mod_rent:,}"):
                                # Create a deep copy to ensure Streamlit detects the change
                                updated_data = copy.deepcopy(st.session_state[rent_roll_key])
                                updated_data[i]['monthly_rent'] = float(low_mod_rent)
                                st.session_state[rent_roll_key] = updated_data
                                st.session_state['_widget_version'] = wv + 1  # Force widget refresh
                                save_with_history(f"Set {ut['name']} to Low/Mod rate: ${low_mod_rent:,}")
                                st.rerun()
                        with btn_cols[1]:
                            if st.button(f"Mid", key=f"set_mid_{listing.id}_{i}_v{wv}", help=f"Set to Middle: ${middle_rent:,}"):
                                # Create a deep copy to ensure Streamlit detects the change
                                updated_data = copy.deepcopy(st.session_state[rent_roll_key])
                                updated_data[i]['monthly_rent'] = float(middle_rent)
                                st.session_state[rent_roll_key] = updated_data
                                st.session_state['_widget_version'] = wv + 1  # Force widget refresh
                                save_with_history(f"Set {ut['name']} to Middle rate: ${middle_rent:,}")
                                st.rerun()
                    with cols[5]:
                        count = st.number_input("Count", value=int(ut['count']), min_value=0, max_value=500, step=1, key=f"rr_count_{listing.id}_{i}_v{wv}", label_visibility="collapsed")
                    with cols[6]:
                        annual_rev = rent * count * 12
                        st.markdown(f"**${annual_rev:,.0f}**")

                    updated_units.append({'name': name, 'bedrooms': int(bedrooms), 'sqft': float(sqft), 'monthly_rent': float(rent), 'count': int(count)})

                # Update session state - only save if values changed
                old_units = st.session_state.get(rent_roll_key, [])
                if updated_units != old_units:
                    st.session_state[rent_roll_key] = updated_units
                    save_with_history(f"Updated rent roll: {listing.address}")

                # Add unit type button
                if st.button("+ Add Unit Type", key=f"add_unit_{listing.id}"):
                    updated_units.append({'name': 'Custom', 'bedrooms': 1, 'sqft': 700.0, 'monthly_rent': 2000.0, 'count': 0})
                    st.session_state[rent_roll_key] = updated_units
                    save_with_history(f"Added unit type to: {listing.address}")
                    st.rerun()

                # Build RentRoll object from current data
                rent_roll = RentRoll(unit_types=[UnitType(**u) for u in updated_units if u['count'] > 0])

                # Calculate building sqft from rent roll if it has units
                if rent_roll.total_units > 0:
                    building_sqft = rent_roll.total_sqft

                # Show totals - calculate Effective Gross Income
                gross_income = rent_roll.total_annual_revenue
                vacancy_bad_debt_loss = gross_income * vacancy_bad_debt
                effective_gross_income = gross_income - vacancy_bad_debt_loss
                opex_amount = effective_gross_income * opex_ratio
                noi_from_roll = effective_gross_income - opex_amount

                total_cols = st.columns(5)
                with total_cols[0]:
                    st.metric("Total Units", rent_roll.total_units)
                with total_cols[1]:
                    st.metric("Total Sq Ft", f"{rent_roll.total_sqft:,.0f}")
                with total_cols[2]:
                    st.metric("Gross Income", f"${gross_income:,.0f}")
                with total_cols[3]:
                    st.metric("Effective Gross", f"${effective_gross_income:,.0f}",
                             delta=f"-${vacancy_bad_debt_loss:,.0f}",
                             delta_color="off")
                with total_cols[4]:
                    st.metric("NOI", f"${noi_from_roll:,.0f}")

                st.divider()

                # ---- Housing Charge Increase Schedule ----
                st.subheader("Housing Charge Increase Schedule")
                st.caption("Annual rent increases over 25 years. Select a preset or customize by period.")

                # Preset buttons row
                preset_cols = st.columns([1, 1, 1, 1, 2])
                with preset_cols[0]:
                    if st.button("Conservative", key=f"rate_cons_{listing.id}", use_container_width=True,
                                type="primary" if scenario_params['rate_mode'] == 'conservative' else "secondary"):
                        scenario_params['rate_mode'] = 'conservative'
                        scenario_params['custom_rates'] = RATE_PRESETS['conservative'].copy()
                        st.rerun()
                with preset_cols[1]:
                    if st.button("Moderate", key=f"rate_mod_{listing.id}", use_container_width=True,
                                type="primary" if scenario_params['rate_mode'] == 'moderate' else "secondary"):
                        scenario_params['rate_mode'] = 'moderate'
                        scenario_params['custom_rates'] = RATE_PRESETS['moderate'].copy()
                        st.rerun()
                with preset_cols[2]:
                    if st.button("Aggressive", key=f"rate_agg_{listing.id}", use_container_width=True,
                                type="primary" if scenario_params['rate_mode'] == 'aggressive' else "secondary"):
                        scenario_params['rate_mode'] = 'aggressive'
                        scenario_params['custom_rates'] = RATE_PRESETS['aggressive'].copy()
                        st.rerun()
                with preset_cols[3]:
                    if st.button("Custom", key=f"rate_custom_{listing.id}", use_container_width=True,
                                type="primary" if scenario_params['rate_mode'] == 'custom' else "secondary"):
                        scenario_params['rate_mode'] = 'custom'
                        st.rerun()

                # Build rate schedule
                is_custom_rate = scenario_params['rate_mode'] == 'custom'
                rate_schedule = build_rate_schedule(scenario_params['custom_rates'])

                # Period sliders (only in custom mode) and chart
                if is_custom_rate:
                    schedule_cols = st.columns([2, 3])
                    with schedule_cols[0]:
                        periods = ['period_1', 'period_2', 'period_3', 'period_4', 'period_5']
                        period_labels = ['Years 1-5', 'Years 6-10', 'Years 11-15', 'Years 16-20', 'Years 21-25']

                        for period, label in zip(periods, period_labels):
                            new_rate = st.slider(
                                label,
                                min_value=0.0, max_value=5.0,
                                value=scenario_params['custom_rates'][period],
                                step=0.25, format="%.2f%%",
                                key=f"rate_{period}_{listing.id}_v{wv}",
                                label_visibility="visible"
                            )
                            if new_rate != scenario_params['custom_rates'][period]:
                                scenario_params['custom_rates'][period] = new_rate
                                rate_schedule = build_rate_schedule(scenario_params['custom_rates'])
                    chart_col = schedule_cols[1]
                else:
                    chart_col = st.container()

                with chart_col:
                    rate_chart_data = pd.DataFrame({
                        'Year': list(range(1, 26)),
                        'Annual Increase (%)': [r * 100 for r in rate_schedule]
                    })

                    cumulative = [100]
                    for rate in rate_schedule:
                        cumulative.append(cumulative[-1] * (1 + rate))
                    rate_chart_data['Cumulative Rent Index'] = cumulative[1:]

                    base = alt.Chart(rate_chart_data).encode(x=alt.X('Year:Q', scale=alt.Scale(domain=[1, 25]), title='Year'))
                    bars = base.mark_bar(color='#4CAF50', opacity=0.7).encode(
                        y=alt.Y('Annual Increase (%):Q', scale=alt.Scale(domain=[0, 5]), title='Annual Rate (%)')
                    )
                    line = base.mark_line(color='#1976D2', strokeWidth=2).encode(
                        y=alt.Y('Cumulative Rent Index:Q', scale=alt.Scale(domain=[100, 220]), title='Cumulative (100=Year 0)')
                    )
                    chart = alt.layer(bars, line).resolve_scale(y='independent').properties(height=200)
                    st.altair_chart(chart, use_container_width=True)

                    final_rent_index = cumulative[-1]
                    avg_rate = sum(rate_schedule) / len(rate_schedule) * 100
                    st.caption(f"25-year growth: **{final_rent_index:.0f}%** of Year 0 rent | Avg: **{avg_rate:.2f}%**/yr")

                # Income inflation
                st.markdown("<small>**Income inflation:** Annual growth rate for household incomes.</small>", unsafe_allow_html=True)
                infl_cols = st.columns([1, 3])
                with infl_cols[0]:
                    new_inflation = st.slider(
                        "Income Inflation",
                        min_value=-2.0, max_value=6.0,
                        value=scenario_params['inflation_assumption'] * 100,
                        step=0.25, format="%.2f%%",
                        key=f"inflation_{listing.id}_v{wv}"
                    ) / 100

                # Update inflation in scenario params
                if new_inflation != scenario_params['inflation_assumption']:
                    scenario_params['inflation_assumption'] = new_inflation

                st.divider()

                # ---- Affordability by Unit Type ----
                st.subheader("Affordability by Unit Type")
                st.caption("Edit $/SqFt or burden to update rents. All values are linked.")

                if rent_roll.total_units == 0:
                    st.info("Add units to the rent roll to see affordability analysis.")
                else:
                    # Get income thresholds from session state (set in Income Distributions tab)
                    couples_income = st.session_state.get('income_dist_couples', BC_HOUSING_DEFAULTS['couples_without_children'])
                    families_income = st.session_state.get('income_dist_families', BC_HOUSING_DEFAULTS['families_with_children'])

                    # Header row for affordability table
                    aff_header = st.columns([2, 0.6, 0.8, 1.2, 1, 0.8, 1.2, 1.2])
                    with aff_header[0]:
                        st.markdown("**Unit Type**")
                    with aff_header[1]:
                        st.markdown("**Beds**")
                    with aff_header[2]:
                        st.markdown("**Sq Ft**")
                    with aff_header[3]:
                        st.markdown("**Rent**")
                    with aff_header[4]:
                        st.markdown("**$/SqFt**")
                    with aff_header[5]:
                        st.markdown("**Units**")
                    with aff_header[6]:
                        st.markdown("**L/M Burden**")
                    with aff_header[7]:
                        st.markdown("**Mid Burden**")

                    # Build mapping from rent_roll index to display row
                    rent_roll_data = st.session_state[rent_roll_key]
                    affordability_summary = {'total': 0, 'lm_affordable': 0, 'mid_affordable': 0}

                    for i, ut_data in enumerate(rent_roll_data):
                        if ut_data['count'] == 0:
                            continue

                        sqft = ut_data['sqft']
                        monthly_rent = ut_data['monthly_rent']
                        bedrooms = ut_data.get('bedrooms', 1)

                        # Determine income thresholds based on bedrooms
                        if bedrooms < 2:
                            low_mod_income = couples_income['median']
                            middle_income = couples_income['75th_percentile']
                        else:
                            low_mod_income = families_income['median']
                            middle_income = families_income['75th_percentile']

                        # Calculate current values
                        rent_per_sqft = monthly_rent / sqft if sqft > 0 else 0
                        annual_rent = monthly_rent * 12
                        burden_lm = annual_rent / low_mod_income if low_mod_income > 0 else 0
                        burden_mid = annual_rent / middle_income if middle_income > 0 else 0

                        # Track affordability
                        affordability_summary['total'] += ut_data['count']
                        if burden_lm <= affordability_ratio:
                            affordability_summary['lm_affordable'] += ut_data['count']
                        if burden_mid <= affordability_ratio:
                            affordability_summary['mid_affordable'] += ut_data['count']

                        # Editable row
                        aff_cols = st.columns([2, 0.6, 0.8, 1.2, 1, 0.8, 1.2, 1.2])
                        with aff_cols[0]:
                            st.text(ut_data['name'])
                        with aff_cols[1]:
                            st.text(str(bedrooms))
                        with aff_cols[2]:
                            st.text(f"{sqft:,.0f}")
                        with aff_cols[3]:
                            st.text(f"${monthly_rent:,.0f}")
                        with aff_cols[4]:
                            new_psf = st.number_input(
                                "$/sf", value=rent_per_sqft, min_value=0.0, max_value=20.0,
                                step=0.25, format="%.2f", key=f"aff_psf_{listing.id}_{i}_v{wv}",
                                label_visibility="collapsed"
                            )
                        with aff_cols[5]:
                            st.text(str(ut_data['count']))
                        with aff_cols[6]:
                            new_burden_lm = st.number_input(
                                "L/M", value=burden_lm * 100, min_value=0.0, max_value=100.0,
                                step=1.0, format="%.1f", key=f"aff_lm_{listing.id}_{i}_v{wv}",
                                label_visibility="collapsed"
                            ) / 100
                        with aff_cols[7]:
                            new_burden_mid = st.number_input(
                                "Mid", value=burden_mid * 100, min_value=0.0, max_value=100.0,
                                step=1.0, format="%.1f", key=f"aff_mid_{listing.id}_{i}_v{wv}",
                                label_visibility="collapsed"
                            ) / 100

                        # Check which input changed and update rent accordingly
                        new_rent = monthly_rent
                        rent_changed = False

                        # $/SqFt changed
                        if abs(new_psf - rent_per_sqft) > 0.001:
                            new_rent = new_psf * sqft
                            rent_changed = True
                        # L/M Burden changed
                        elif abs(new_burden_lm - burden_lm) > 0.0001:
                            new_rent = (new_burden_lm * low_mod_income) / 12
                            rent_changed = True
                        # Mid Burden changed
                        elif abs(new_burden_mid - burden_mid) > 0.0001:
                            new_rent = (new_burden_mid * middle_income) / 12
                            rent_changed = True

                        if rent_changed:
                            # Update rent roll in session state
                            updated_roll = copy.deepcopy(st.session_state[rent_roll_key])
                            updated_roll[i]['monthly_rent'] = float(new_rent)
                            st.session_state[rent_roll_key] = updated_roll
                            st.session_state['_widget_version'] = wv + 1
                            save_with_history(f"Updated {ut_data['name']} rent via affordability table")
                            st.rerun()

                    # Summary metrics
                    total_units_aff = affordability_summary['total']
                    affordable_low_mod = affordability_summary['lm_affordable']
                    affordable_middle = affordability_summary['mid_affordable']

                    sum_col1, sum_col2 = st.columns(2)
                    with sum_col1:
                        pct_lm = affordable_low_mod / total_units_aff * 100 if total_units_aff > 0 else 0
                        st.metric(
                            "Affordable at Low/Moderate",
                            f"{affordable_low_mod} of {total_units_aff} ({pct_lm:.0f}%)",
                            help=f"Rent burden <= {affordability_ratio:.0%} of median income"
                        )
                    with sum_col2:
                        pct_mid = affordable_middle / total_units_aff * 100 if total_units_aff > 0 else 0
                        st.metric(
                            "Affordable at Middle Income",
                            f"{affordable_middle} of {total_units_aff} ({pct_mid:.0f}%)",
                            help=f"Rent burden <= {affordability_ratio:.0%} of 75th percentile income"
                        )

                st.divider()

                # ---- Viability Analysis ----
                st.subheader("Viability Analysis")

                if rent_roll.total_units == 0:
                    st.warning("Add units to the rent roll to see viability analysis.")
                else:
                    # Calculate financial metrics using rent roll revenue
                    viability_params = fixed_params.copy()
                    viability_params['acquisition_cost_psf'] = params['acquisition_cost_psf']
                    viability_params['rent_psf'] = rent_roll.weighted_avg_rent_psf

                    project = ProjectParameters(**viability_params)
                    model = ProjectViabilityModel(project)

                    # Calculate acquisition and debt service
                    acquisition_cost = params['acquisition_cost_psf'] * building_sqft
                    loan_amount = acquisition_cost * max_ltv
                    equity = acquisition_cost * (1 - max_ltv)

                    # Debt service calculation
                    monthly_rate = interest_rate_senior / 12
                    n_payments = 50 * 12  # 50-year amortization
                    if monthly_rate > 0:
                        payment_factor = monthly_rate * (1 + monthly_rate)**n_payments / ((1 + monthly_rate)**n_payments - 1)
                    else:
                        payment_factor = 1 / n_payments
                    annual_debt_service = loan_amount * payment_factor * 12

                    # NOI and cash flow using Effective Gross Income
                    gross_revenue = rent_roll.total_annual_revenue
                    vacancy_bad_debt_amt = gross_revenue * vacancy_bad_debt
                    eff_gross_income = gross_revenue - vacancy_bad_debt_amt
                    opex_amt = eff_gross_income * opex_ratio
                    noi = eff_gross_income - opex_amt
                    cash_flow = noi - annual_debt_service
                    dscr = noi / annual_debt_service if annual_debt_service > 0 else float('inf')

                    # DSCR check
                    dscr_ok = dscr >= 1.0

                    # Calculate 25-year IRR
                    time_series_for_irr = model.calculate_metrics_over_time(
                        total_sqft=building_sqft,
                        years=25,
                        rate_schedule=rate_schedule
                    )
                    # Override year 0 with actual rent roll values
                    base_rent_psf = rent_roll.weighted_avg_rent_psf
                    base_opex = eff_gross_income * opex_ratio  # Year 0 operating expenses
                    adjusted_time_series = []
                    for year_data in time_series_for_irr:
                        year = year_data['year']
                        if year == 0:
                            year_data['gross_revenue'] = gross_revenue
                            year_data['effective_gross'] = eff_gross_income
                            year_data['opex'] = base_opex
                            year_data['noi'] = noi
                            year_data['cash_flow'] = cash_flow
                            year_data['equity'] = equity
                        else:
                            # Revenue grows by housing charge schedule
                            growth_factor = year_data['rent_psf'] / base_rent_psf if base_rent_psf > 0 else 1
                            year_gross = gross_revenue * growth_factor
                            year_eff_gross = year_gross * (1 - vacancy_bad_debt)
                            # Opex grows by inflation each year
                            year_data['gross_revenue'] = year_gross
                            year_data['effective_gross'] = year_eff_gross
                            year_data['opex'] = base_opex * ((1 + inflation_assumption) ** year)
                            year_data['noi'] = year_eff_gross - year_data['opex']
                            year_data['cash_flow'] = year_data['noi'] - annual_debt_service
                        adjusted_time_series.append(year_data)

                    cash_flows_for_irr = [m['cash_flow'] for m in adjusted_time_series if m['year'] > 0]
                    irr_25yr = calculate_project_irr(equity, cash_flows_for_irr)
                    meets_irr_target = irr_25yr >= equity_return_required - 0.0001  # tolerance for floating point

                    # Key metrics
                    metric_cols = st.columns(6)
                    with metric_cols[0]:
                        acq_label = "Acquisition $/SqFt"
                        if params.get('haircut_pct', 0) > 0:
                            acq_label += f" ({params['haircut_pct']:.0f}% haircut)"
                        st.metric(acq_label, f"${params['acquisition_cost_psf']:.0f}")
                    with metric_cols[1]:
                        st.metric("Rent $/SqFt", f"${rent_roll.weighted_avg_rent_psf:.2f}")
                    with metric_cols[2]:
                        st.metric("DSCR", f"{dscr:.2f}",
                                  delta="OK" if dscr_ok else "Below 1.0",
                                  delta_color="normal" if dscr_ok else "inverse")
                    with metric_cols[3]:
                        st.metric("Year 1 Cash Flow", f"${cash_flow:,.0f}")
                    with metric_cols[4]:
                        st.metric("25-Year IRR", f"{irr_25yr:.2%}",
                                  delta=f"vs {equity_return_required:.2%} target",
                                  delta_color="normal" if meets_irr_target else "inverse")
                    with metric_cols[5]:
                        is_viable = dscr_ok and meets_irr_target
                        status_text = "VIABLE" if is_viable else "NOT VIABLE"
                        st.metric("Status", status_text)

                    # Status messages
                    if dscr_ok:
                        st.success(f"DSCR = {dscr:.2f} - Rents cover debt service")
                    else:
                        st.error(f"DSCR = {dscr:.2f} - Rents do NOT cover debt service (need DSCR >= 1.0)")

                    # Detailed viability calculations
                    with st.expander("View Viability Calculations"):
                        st.markdown("**Acquisition & Financing:**")
                        haircut_applied = params.get('haircut_pct', 0) > 0
                        price_note = f"Price ÷ SqFt (after {params.get('haircut_pct', 0):.0f}% haircut)" if haircut_applied else "Price ÷ SqFt"
                        fin_calc_data = [
                            ("Building Square Footage", f"{building_sqft:,.0f} sqft", "From rent roll"),
                            ("Acquisition Cost per SqFt", f"${params['acquisition_cost_psf']:,.2f}", price_note),
                            ("Total Acquisition Cost", f"${acquisition_cost:,.0f}", f"{building_sqft:,.0f} × ${params['acquisition_cost_psf']:,.2f}"),
                            ("Loan-to-Value (LTV)", f"{max_ltv:.0%}", "Model parameter"),
                            ("Senior Loan Amount", f"${loan_amount:,.0f}", f"${acquisition_cost:,.0f} × {max_ltv:.0%}"),
                            ("Equity Required", f"${equity:,.0f}", f"${acquisition_cost:,.0f} × {(1-max_ltv):.0%}"),
                        ]
                        fin_calc_df = pd.DataFrame(fin_calc_data, columns=["Item", "Value", "Calculation"])
                        st.dataframe(fin_calc_df, use_container_width=True, hide_index=True)

                        st.markdown("**Debt Service Calculation:**")
                        monthly_payment = annual_debt_service / 12
                        debt_calc_data = [
                            ("Interest Rate", f"{interest_rate_senior:.2%}", "Model parameter"),
                            ("Monthly Rate", f"{monthly_rate:.6f}", f"{interest_rate_senior:.2%} ÷ 12"),
                            ("Amortization", "50 years (600 months)", "Fixed"),
                            ("Monthly Payment", f"${monthly_payment:,.2f}", "PMT formula"),
                            ("Annual Debt Service", f"${annual_debt_service:,.0f}", f"${monthly_payment:,.2f} × 12"),
                        ]
                        debt_calc_df = pd.DataFrame(debt_calc_data, columns=["Item", "Value", "Calculation"])
                        st.dataframe(debt_calc_df, use_container_width=True, hide_index=True)

                        st.markdown("**Cash Flow & DSCR:**")
                        cf_calc_data = [
                            ("Gross Revenue", f"${gross_revenue:,.0f}", "Sum of all unit rents × 12"),
                            ("Less: Vacancy & Bad Debt", f"(${vacancy_bad_debt_amt:,.0f})", f"${gross_revenue:,.0f} × {vacancy_bad_debt:.1%}"),
                            ("Effective Gross Income", f"${eff_gross_income:,.0f}", f"${gross_revenue:,.0f} - ${vacancy_bad_debt_amt:,.0f}"),
                            ("Operating Expense Ratio", f"{opex_ratio:.0%}", "Model parameter"),
                            ("Operating Expenses", f"${opex_amt:,.0f}", f"${eff_gross_income:,.0f} × {opex_ratio:.0%}"),
                            ("Net Operating Income (NOI)", f"${noi:,.0f}", f"${eff_gross_income:,.0f} - ${opex_amt:,.0f}"),
                            ("Annual Debt Service", f"${annual_debt_service:,.0f}", "From above"),
                            ("Cash Flow", f"${cash_flow:,.0f}", f"${noi:,.0f} - ${annual_debt_service:,.0f}"),
                            ("DSCR", f"{dscr:.2f}", f"${noi:,.0f} ÷ ${annual_debt_service:,.0f}"),
                        ]
                        cf_calc_df = pd.DataFrame(cf_calc_data, columns=["Item", "Value", "Calculation"])
                        st.dataframe(cf_calc_df, use_container_width=True, hide_index=True)

                        st.markdown("**IRR Calculation:**")
                        st.markdown(f"""
                        - Initial Investment (Equity): ${equity:,.0f}
                        - 25 years of projected cash flows (see projection table)
                        - Cash flows grow as revenue increases by housing charge schedule
                        - Operating expenses grow by inflation ({inflation_assumption:.1%}/year)
                        - Debt service remains constant
                        - **25-Year IRR: {irr_25yr:.2%}**
                        """)

                    # ---- Viability Space Visualization ----
                    st.divider()
                    st.subheader("Viability Space")
                    st.caption("Explore how viability changes across two parameters. Green = viable, red = not viable. Star marks current position.")

                    # Define available parameters for axes
                    VIABILITY_PARAMS = {
                        'Haircut %': {'current': haircut_pct, 'range': (0, 50), 'step': 2, 'format': '{:.0f}%', 'mult': 1},
                        'LTV %': {'current': max_ltv * 100, 'range': (50, 95), 'step': 5, 'format': '{:.0f}%', 'mult': 1},
                        'Interest Rate %': {'current': interest_rate_senior * 100, 'range': (2, 10), 'step': 0.5, 'format': '{:.1f}%', 'mult': 1},
                        'OpEx Ratio %': {'current': opex_ratio * 100, 'range': (15, 55), 'step': 2, 'format': '{:.0f}%', 'mult': 1},
                        'Vacancy & Bad Debt %': {'current': vacancy_bad_debt * 100, 'range': (0, 20), 'step': 0.5, 'format': '{:.1f}%', 'mult': 1},
                        'Rent $/SqFt': {'current': rent_roll.weighted_avg_rent_psf, 'range': (1, 8), 'step': 0.25, 'format': '${:.2f}', 'mult': 1},
                        'Target IRR %': {'current': equity_return_required * 100, 'range': (0, 15), 'step': 0.5, 'format': '{:.1f}%', 'mult': 1},
                    }

                    # Parameter selection
                    param_options = list(VIABILITY_PARAMS.keys())

                    # Track selections and widget version for swap functionality
                    x_sel_key = f"viab_x_sel_{listing.id}"
                    y_sel_key = f"viab_y_sel_{listing.id}"
                    viab_ver_key = f"viab_ver_{listing.id}"

                    # Initialize session state
                    if x_sel_key not in st.session_state:
                        st.session_state[x_sel_key] = param_options[0]
                    if y_sel_key not in st.session_state:
                        st.session_state[y_sel_key] = param_options[1]
                    if viab_ver_key not in st.session_state:
                        st.session_state[viab_ver_key] = 0

                    viab_ver = st.session_state[viab_ver_key]
                    current_x = st.session_state[x_sel_key]
                    current_y = st.session_state[y_sel_key]

                    axis_cols = st.columns([1, 0.3, 1, 1.7])
                    with axis_cols[0]:
                        x_idx = param_options.index(current_x) if current_x in param_options else 0
                        x_param = st.selectbox("X-Axis", param_options, index=x_idx, key=f"viab_x_{listing.id}_v{viab_ver}")
                        if x_param != st.session_state[x_sel_key]:
                            st.session_state[x_sel_key] = x_param
                    with axis_cols[1]:
                        st.write("")  # Spacing
                        if st.button("⇄", key=f"swap_axes_{listing.id}", help="Swap X and Y axes"):
                            # Swap selections and increment version for fresh widgets
                            st.session_state[x_sel_key] = current_y
                            st.session_state[y_sel_key] = current_x
                            st.session_state[viab_ver_key] = viab_ver + 1
                            st.rerun()
                    with axis_cols[2]:
                        y_options = [p for p in param_options if p != x_param]
                        # Ensure current Y selection is valid
                        if current_y == x_param or current_y not in y_options:
                            current_y = y_options[0]
                        y_idx = y_options.index(current_y) if current_y in y_options else 0
                        y_param = st.selectbox("Y-Axis", y_options, index=y_idx, key=f"viab_y_{listing.id}_v{viab_ver}")
                        if y_param != st.session_state[y_sel_key]:
                            st.session_state[y_sel_key] = y_param

                    # Get parameter configs
                    x_cfg = VIABILITY_PARAMS[x_param]
                    y_cfg = VIABILITY_PARAMS[y_param]

                    # Generate grid values
                    x_values = np.arange(x_cfg['range'][0], x_cfg['range'][1] + x_cfg['step'], x_cfg['step'])
                    y_values = np.arange(y_cfg['range'][0], y_cfg['range'][1] + y_cfg['step'], y_cfg['step'])

                    # Function to calculate viability for given parameter values
                    def calc_viability_point(x_val, y_val, x_name, y_name):
                        # Start with current values
                        p_haircut = haircut_pct
                        p_ltv = max_ltv
                        p_interest = interest_rate_senior
                        p_opex = opex_ratio
                        p_vacancy_bad_debt = vacancy_bad_debt
                        p_rent_psf = rent_roll.weighted_avg_rent_psf
                        p_target_irr = equity_return_required

                        # Override X parameter
                        if x_name == 'Haircut %':
                            p_haircut = x_val / 100
                        elif x_name == 'LTV %':
                            p_ltv = x_val / 100
                        elif x_name == 'Interest Rate %':
                            p_interest = x_val / 100
                        elif x_name == 'OpEx Ratio %':
                            p_opex = x_val / 100
                        elif x_name == 'Vacancy & Bad Debt %':
                            p_vacancy_bad_debt = x_val / 100
                        elif x_name == 'Rent $/SqFt':
                            p_rent_psf = x_val
                        elif x_name == 'Target IRR %':
                            p_target_irr = x_val / 100

                        # Override Y parameter
                        if y_name == 'Haircut %':
                            p_haircut = y_val / 100
                        elif y_name == 'LTV %':
                            p_ltv = y_val / 100
                        elif y_name == 'Interest Rate %':
                            p_interest = y_val / 100
                        elif y_name == 'OpEx Ratio %':
                            p_opex = y_val / 100
                        elif y_name == 'Vacancy & Bad Debt %':
                            p_vacancy_bad_debt = y_val / 100
                        elif y_name == 'Rent $/SqFt':
                            p_rent_psf = y_val
                        elif y_name == 'Target IRR %':
                            p_target_irr = y_val / 100

                        # Calculate financials
                        eff_price = listing.asking_price * (1 - p_haircut)
                        loan_amt = eff_price * p_ltv
                        eq = eff_price * (1 - p_ltv)

                        # Debt service
                        m_rate = p_interest / 12
                        n_pmt = 50 * 12
                        if m_rate > 0:
                            pmt_factor = m_rate * (1 + m_rate)**n_pmt / ((1 + m_rate)**n_pmt - 1)
                        else:
                            pmt_factor = 1 / n_pmt
                        ann_debt_svc = loan_amt * pmt_factor * 12

                        # Revenue with rent adjustment
                        if p_rent_psf != rent_roll.weighted_avg_rent_psf and rent_roll.weighted_avg_rent_psf > 0:
                            rent_factor = p_rent_psf / rent_roll.weighted_avg_rent_psf
                        else:
                            rent_factor = 1.0

                        # Year 0 values
                        base_gross = rent_roll.total_annual_revenue * rent_factor
                        base_eff_gross = base_gross * (1 - p_vacancy_bad_debt)
                        base_opex_amt = base_eff_gross * p_opex
                        base_noi = base_eff_gross - base_opex_amt
                        base_cf = base_noi - ann_debt_svc

                        # DSCR (Year 1)
                        pt_dscr = base_noi / ann_debt_svc if ann_debt_svc > 0 else float('inf')

                        # Build 25-year cash flows for IRR
                        cash_flows = []
                        for yr in range(1, 26):
                            # Revenue grows by housing charge schedule
                            rev_growth = 1.0
                            for y in range(yr):
                                rev_growth *= (1 + rate_schedule[min(y, len(rate_schedule) - 1)])
                            yr_gross = base_gross * rev_growth
                            yr_eff_gross = yr_gross * (1 - p_vacancy_bad_debt)
                            # OpEx grows by inflation
                            yr_opex = base_opex_amt * ((1 + inflation_assumption) ** yr)
                            yr_noi = yr_eff_gross - yr_opex
                            yr_cf = yr_noi - ann_debt_svc
                            cash_flows.append(yr_cf)

                        # Calculate 25-year IRR
                        pt_irr = calculate_project_irr(eq, cash_flows)

                        # Viability check
                        dscr_ok = pt_dscr >= 1.0
                        irr_ok = pt_irr >= p_target_irr - 0.0001  # tolerance for floating point
                        viable = dscr_ok and irr_ok

                        return {'dscr': pt_dscr, 'irr': pt_irr, 'viable': viable, 'dscr_ok': dscr_ok, 'irr_ok': irr_ok}

                    # Build grid data
                    grid_data = []
                    for x_val in x_values:
                        for y_val in y_values:
                            result = calc_viability_point(x_val, y_val, x_param, y_param)
                            grid_data.append({
                                'x': x_val,
                                'y': y_val,
                                'viable': 1 if result['viable'] else 0,
                                'dscr': result['dscr'],
                                'irr': result['irr'] * 100,
                                'status': 'Viable' if result['viable'] else ('DSCR < 1' if not result['dscr_ok'] else 'IRR Below Target'),
                            })

                    grid_df = pd.DataFrame(grid_data)

                    # Create heatmap
                    heatmap = alt.Chart(grid_df).mark_rect().encode(
                        x=alt.X('x:O', title=x_param, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('y:O', title=y_param, sort='descending'),
                        color=alt.Color('viable:Q',
                                        scale=alt.Scale(domain=[0, 1], range=['#ffcccc', '#ccffcc']),
                                        legend=None),
                        tooltip=[
                            alt.Tooltip('x:Q', title=x_param, format='.1f'),
                            alt.Tooltip('y:Q', title=y_param, format='.1f'),
                            alt.Tooltip('dscr:Q', title='DSCR', format='.2f'),
                            alt.Tooltip('irr:Q', title='25-Year IRR %', format='.1f'),
                            alt.Tooltip('status:N', title='Status'),
                        ]
                    )

                    # Add current position marker
                    current_point = pd.DataFrame([{
                        'x': x_cfg['current'],
                        'y': y_cfg['current'],
                    }])

                    # Find closest grid point for current position
                    current_x_grid = min(x_values, key=lambda v: abs(v - x_cfg['current']))
                    current_y_grid = min(y_values, key=lambda v: abs(v - y_cfg['current']))

                    current_marker = alt.Chart(pd.DataFrame([{'x': current_x_grid, 'y': current_y_grid}])).mark_point(
                        shape='diamond',
                        size=200,
                        color='black',
                        strokeWidth=2,
                        filled=True,
                    ).encode(
                        x='x:O',
                        y='y:O',
                    )

                    # Combine chart
                    viability_chart = (heatmap + current_marker).properties(
                        height=400,
                        title=f'Viability Space: {x_param} vs {y_param}'
                    )

                    st.altair_chart(viability_chart, use_container_width=True)

                    # Show all parameters table
                    st.markdown("**Parameter Values**")
                    param_table_data = []
                    for param_name, param_cfg in VIABILITY_PARAMS.items():
                        if param_name == x_param:
                            role = "X-Axis"
                            value_str = f"{x_cfg['format'].format(param_cfg['current'])} (varying)"
                        elif param_name == y_param:
                            role = "Y-Axis"
                            value_str = f"{y_cfg['format'].format(param_cfg['current'])} (varying)"
                        else:
                            role = "Fixed"
                            value_str = param_cfg['format'].format(param_cfg['current'])
                        param_table_data.append({
                            'Parameter': param_name,
                            'Current Value': value_str,
                            'Role': role,
                        })

                    param_table_df = pd.DataFrame(param_table_data)

                    # Display as columns for compactness
                    fixed_params_display = [p for p in param_table_data if p['Role'] == 'Fixed']
                    axis_params_display = [p for p in param_table_data if p['Role'] != 'Fixed']

                    st.caption(f"**Axes:** {x_param} = {x_cfg['format'].format(x_cfg['current'])} (X), {y_param} = {y_cfg['format'].format(y_cfg['current'])} (Y)")

                    # Show fixed parameters in a compact row
                    fixed_cols = st.columns(len(fixed_params_display))
                    for idx, param in enumerate(fixed_params_display):
                        with fixed_cols[idx]:
                            st.metric(param['Parameter'], param['Current Value'], delta=None)

                    # ---- Investment Product Structure ----
                    if is_viable:
                        st.divider()
                        st.subheader("Investment Product Structure")
                        st.caption("Structuring the equity gap as subordinated debt for non-profit financing.")

                        # Calculate subordinated debt structures
                        sub_debt_principal = equity
                        hold_period = 25  # years
                        target_irr = equity_return_required

                        # Get cash flows available for sub debt (after senior debt service)
                        available_cash_flows = [m['cash_flow'] for m in adjusted_time_series if m['year'] > 0]

                        # Structure 1: Full Deferral (Soft Second)
                        # No payments during hold, balloon at exit
                        # Rate = target IRR to deliver same return as equity
                        soft_second_rate = target_irr
                        soft_second_balloon = sub_debt_principal * ((1 + soft_second_rate) ** hold_period)

                        # Structure 2: Deferred Start Interest-Only
                        # Find optimal deferral period based on cash flow capacity
                        # After deferral, pay interest only from available cash flow
                        def calc_deferred_io_structure(principal, target_irr, cash_flows, hold_years):
                            """
                            Find optimal deferral period and rate for deferred interest-only structure.
                            Returns: (deferral_years, rate, annual_payment, balloon, achieved_irr)
                            """
                            best_structure = None

                            for deferral_years in range(1, hold_years):
                                payment_years = hold_years - deferral_years
                                if payment_years < 1:
                                    continue

                                # Available cash flow in payment years (use minimum to be conservative)
                                payment_period_cfs = cash_flows[deferral_years:]
                                if not payment_period_cfs:
                                    continue
                                min_available_cf = min(payment_period_cfs)

                                # Max we can pay as interest only (leave some buffer)
                                max_annual_payment = min_available_cf * 0.8  # 80% of available

                                if max_annual_payment <= 0:
                                    continue

                                # Interest rate implied by this payment
                                implied_rate = max_annual_payment / principal

                                # Accrued interest during deferral (compounding)
                                accrued_at_deferral_end = principal * ((1 + implied_rate) ** deferral_years)

                                # Balloon = accrued principal + remaining after IO payments
                                # During IO period, we pay interest only, so principal stays as accrued amount
                                balloon = accrued_at_deferral_end

                                # Calculate IRR for this structure
                                # Cash flows: -principal at year 0, then 0 during deferral,
                                # then annual_payment during IO, then balloon at exit
                                structure_cfs = [-principal]
                                for yr in range(1, hold_years + 1):
                                    if yr <= deferral_years:
                                        structure_cfs.append(0)
                                    elif yr < hold_years:
                                        structure_cfs.append(max_annual_payment)
                                    else:
                                        structure_cfs.append(max_annual_payment + balloon)

                                try:
                                    achieved_irr = npf.irr(structure_cfs)
                                    if np.isnan(achieved_irr):
                                        continue
                                except:
                                    continue

                                # Check if this meets target IRR
                                if achieved_irr >= target_irr * 0.95:  # Within 5% of target
                                    if best_structure is None or deferral_years < best_structure[0]:
                                        best_structure = (deferral_years, implied_rate, max_annual_payment, balloon, achieved_irr)

                            return best_structure

                        deferred_io = calc_deferred_io_structure(sub_debt_principal, target_irr, available_cash_flows, hold_period)

                        # Structure 3: Residual Receipts
                        # Pay whatever cash flow is available each year, balloon at exit
                        def calc_residual_receipts_structure(principal, target_irr, cash_flows, hold_years):
                            """
                            Structure where sub debt receives available cash flow each year.
                            Calculate implied rate and balloon needed to hit target IRR.
                            """
                            # Use 50% of available cash flow for sub debt payments
                            annual_payments = [max(0, cf * 0.5) for cf in cash_flows]
                            total_payments = sum(annual_payments)

                            # Target total return
                            target_total = principal * ((1 + target_irr) ** hold_years)
                            balloon_needed = target_total - total_payments

                            # Verify IRR
                            structure_cfs = [-principal] + annual_payments[:-1] + [annual_payments[-1] + balloon_needed]
                            try:
                                achieved_irr = npf.irr(structure_cfs)
                            except:
                                achieved_irr = None

                            return annual_payments, balloon_needed, achieved_irr

                        residual_payments, residual_balloon, residual_irr = calc_residual_receipts_structure(
                            sub_debt_principal, target_irr, available_cash_flows, hold_period
                        )

                        # Display capital stack summary
                        st.markdown("**Optimized Capital Stack**")

                        stack_cols = st.columns(3)
                        with stack_cols[0]:
                            st.metric("Senior Debt", f"${loan_amount:,.0f}", delta=f"{max_ltv:.0%} LTV")
                        with stack_cols[1]:
                            st.metric("Subordinated Debt", f"${sub_debt_principal:,.0f}", delta=f"{(1-max_ltv):.0%} of capital")
                        with stack_cols[2]:
                            st.metric("Total Capital", f"${acquisition_cost:,.0f}")

                        st.markdown("---")

                        # Recommended structure based on analysis
                        st.markdown("**Subordinated Debt Structure Options**")

                        # Option 1: Soft Second (always available)
                        with st.expander("Option A: Soft Second (Full Deferral)", expanded=True):
                            st.markdown(f"""
                            **Structure:** Fully deferred subordinated loan - no payments during hold period.

                            | Term | Value |
                            |------|-------|
                            | Principal | ${sub_debt_principal:,.0f} |
                            | Interest Rate | {soft_second_rate:.2%} (accruing) |
                            | Deferral Type | Full P&I Deferral |
                            | Deferral Period | {hold_period} years (entire hold) |
                            | Annual Payments | $0 |
                            | Balloon at Exit | ${soft_second_balloon:,.0f} |
                            | Investor IRR | {target_irr:.2%} |

                            **How it works:** Interest accrues and compounds over the hold period.
                            The entire principal plus accrued interest is repaid at exit (sale or refinance).
                            This maximizes cash flow during operations while delivering target returns to investors.
                            """)

                            # Show year-by-year for soft second
                            soft_second_schedule = []
                            balance = sub_debt_principal
                            for yr in range(hold_period + 1):
                                if yr == 0:
                                    soft_second_schedule.append({
                                        'Year': yr, 'Payment': 0, 'Interest Accrued': 0,
                                        'Balance': balance, 'Project Cash Flow': 0
                                    })
                                else:
                                    interest = balance * soft_second_rate
                                    balance += interest
                                    project_cf = available_cash_flows[yr-1] if yr <= len(available_cash_flows) else 0
                                    soft_second_schedule.append({
                                        'Year': yr, 'Payment': 0, 'Interest Accrued': interest,
                                        'Balance': balance, 'Project Cash Flow': project_cf
                                    })

                            ss_df = pd.DataFrame(soft_second_schedule)
                            ss_df['Payment'] = ss_df['Payment'].apply(lambda x: f"${x:,.0f}")
                            ss_df['Interest Accrued'] = ss_df['Interest Accrued'].apply(lambda x: f"${x:,.0f}")
                            ss_df['Balance'] = ss_df['Balance'].apply(lambda x: f"${x:,.0f}")
                            ss_df['Project Cash Flow'] = ss_df['Project Cash Flow'].apply(lambda x: f"${x:,.0f}")

                            with st.expander("View Payment Schedule"):
                                st.dataframe(ss_df, use_container_width=True, hide_index=True)

                        # Option 2: Deferred Interest-Only (if feasible)
                        if deferred_io:
                            defer_yrs, dio_rate, dio_payment, dio_balloon, dio_irr = deferred_io
                            with st.expander("Option B: Deferred Start Interest-Only"):
                                st.markdown(f"""
                                **Structure:** Payment holiday followed by interest-only payments.

                                | Term | Value |
                                |------|-------|
                                | Principal | ${sub_debt_principal:,.0f} |
                                | Interest Rate | {dio_rate:.2%} |
                                | Deferral Type | Deferred Interest |
                                | Deferral Period | {defer_yrs} years |
                                | Payment Period | Years {defer_yrs + 1}-{hold_period} |
                                | Annual Payment | ${dio_payment:,.0f} |
                                | Balloon at Exit | ${dio_balloon:,.0f} |
                                | Investor IRR | {dio_irr:.2%} |

                                **How it works:** No payments for the first {defer_yrs} years while the project stabilizes.
                                Interest accrues during deferral. After year {defer_yrs}, annual interest payments
                                of ${dio_payment:,.0f} begin. Remaining balance due at exit.
                                """)

                                # Payment schedule for Deferred IO
                                dio_schedule = []
                                dio_balance = sub_debt_principal
                                for yr in range(hold_period + 1):
                                    if yr == 0:
                                        dio_schedule.append({
                                            'Year': yr, 'Payment': 0, 'Interest Accrued': 0,
                                            'Balance': dio_balance, 'Project Cash Flow': 0
                                        })
                                    elif yr <= defer_yrs:
                                        # Deferral period - interest accrues
                                        interest = dio_balance * dio_rate
                                        dio_balance += interest
                                        project_cf = available_cash_flows[yr-1] if yr <= len(available_cash_flows) else 0
                                        dio_schedule.append({
                                            'Year': yr, 'Payment': 0, 'Interest Accrued': interest,
                                            'Balance': dio_balance, 'Project Cash Flow': project_cf
                                        })
                                    else:
                                        # Payment period - interest only
                                        interest = dio_balance * dio_rate
                                        payment = dio_payment
                                        project_cf = available_cash_flows[yr-1] if yr <= len(available_cash_flows) else 0
                                        dio_schedule.append({
                                            'Year': yr, 'Payment': payment, 'Interest Accrued': interest - payment,
                                            'Balance': dio_balance, 'Project Cash Flow': project_cf
                                        })

                                dio_df = pd.DataFrame(dio_schedule)
                                dio_df['Payment'] = dio_df['Payment'].apply(lambda x: f"${x:,.0f}")
                                dio_df['Interest Accrued'] = dio_df['Interest Accrued'].apply(lambda x: f"${x:,.0f}")
                                dio_df['Balance'] = dio_df['Balance'].apply(lambda x: f"${x:,.0f}")
                                dio_df['Project Cash Flow'] = dio_df['Project Cash Flow'].apply(lambda x: f"${x:,.0f}")

                                with st.expander("View Payment Schedule"):
                                    st.dataframe(dio_df, use_container_width=True, hide_index=True)

                        # Option 3: Residual Receipts
                        if residual_irr and not np.isnan(residual_irr) and residual_balloon > 0:
                            avg_residual_payment = sum(residual_payments) / len(residual_payments)
                            with st.expander("Option C: Residual Receipts"):
                                st.markdown(f"""
                                **Structure:** Payments tied to available cash flow each year.

                                | Term | Value |
                                |------|-------|
                                | Principal | ${sub_debt_principal:,.0f} |
                                | Deferral Type | Residual Receipts |
                                | Annual Payments | ~${avg_residual_payment:,.0f} avg (50% of cash flow) |
                                | Total Distributions | ${sum(residual_payments):,.0f} over {hold_period} years |
                                | Balloon at Exit | ${residual_balloon:,.0f} |
                                | Investor IRR | {residual_irr:.2%} |

                                **How it works:** Investor receives 50% of available project cash flow each year.
                                Remaining return delivered via balloon payment at exit. This shares both
                                upside and risk with the project.
                                """)

                                # Payment schedule for Residual Receipts
                                rr_schedule = []
                                cumulative_paid = 0
                                for yr in range(hold_period + 1):
                                    if yr == 0:
                                        rr_schedule.append({
                                            'Year': yr, 'Project Cash Flow': 0, 'Payment (50%)': 0,
                                            'Cumulative Paid': 0, 'Remaining to Target': sub_debt_principal * ((1 + target_irr) ** hold_period)
                                        })
                                    else:
                                        project_cf = available_cash_flows[yr-1] if yr <= len(available_cash_flows) else 0
                                        payment = residual_payments[yr-1] if yr <= len(residual_payments) else 0
                                        cumulative_paid += payment
                                        target_total = sub_debt_principal * ((1 + target_irr) ** hold_period)
                                        remaining = target_total - cumulative_paid
                                        rr_schedule.append({
                                            'Year': yr, 'Project Cash Flow': project_cf, 'Payment (50%)': payment,
                                            'Cumulative Paid': cumulative_paid, 'Remaining to Target': remaining
                                        })

                                rr_df = pd.DataFrame(rr_schedule)
                                rr_df['Project Cash Flow'] = rr_df['Project Cash Flow'].apply(lambda x: f"${x:,.0f}")
                                rr_df['Payment (50%)'] = rr_df['Payment (50%)'].apply(lambda x: f"${x:,.0f}")
                                rr_df['Cumulative Paid'] = rr_df['Cumulative Paid'].apply(lambda x: f"${x:,.0f}")
                                rr_df['Remaining to Target'] = rr_df['Remaining to Target'].apply(lambda x: f"${x:,.0f}")

                                with st.expander("View Payment Schedule"):
                                    st.dataframe(rr_df, use_container_width=True, hide_index=True)

