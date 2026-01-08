"""
Affordable Housing Viability Analyzer

Streamlit app for detailed analysis of multifamily housing project viability
and income affordability in British Columbia.
"""

import streamlit as st
import pandas as pd
import numpy as np
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
# SIDEBAR
# ============================================

with st.sidebar:
    # ============================================
    # Add Listing Section
    # ============================================
    st.header("Add Listing")

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

    with st.form("listing_form"):
        ml_address = st.text_input("Address", placeholder="123 Main St")
        ml_city = st.text_input("City", placeholder="Vancouver")
        ml_price = st.number_input("Asking Price ($)", min_value=100000, max_value=500000000, value=5000000, step=100000)
        ml_sqft = st.number_input("Building Sq Ft", min_value=0, max_value=1000000, value=0, step=1000, help="Leave 0 to estimate from units")
        ml_units = st.number_input("Number of Units", min_value=1, max_value=1000, value=20, step=1)

        submitted = st.form_submit_button("Analyze", type="primary", use_container_width=True)
        if submitted:
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
                # Select the new listing in the dropdown
                st.session_state['_select_listing'] = f"{ml_address} - {ml_city}"
                st.rerun()
            else:
                st.error("Address and City are required")

    # Undo button
    history = load_history()
    history_count = len(history)

    if st.button("↶ Undo", use_container_width=True, disabled=(history_count < 2)):
        success, desc = undo_last_change()
        if success:
            # Load the restored state
            saved_data = load_listings_from_file()
            if saved_data:
                apply_state(saved_data)
            st.rerun()

    st.caption(f"History: {history_count} state{'s' if history_count != 1 else ''} saved")

    st.divider()

    # ============================================
    # Model Parameters
    # ============================================
    st.header("Model Parameters")
    st.caption("Adjust financing, economic, and affordability assumptions. Changes apply to all analyses.")

    st.subheader("Assumptions")
    opex_ratio = create_param_input('opex_ratio', 'sidebar_')

    st.subheader("Lender")
    interest_rate_senior = create_param_input('interest_rate_senior', 'sidebar_')
    max_ltv = create_param_input('max_ltv', 'sidebar_')

    st.subheader("Economic")
    inflation_assumption = create_param_input('inflation_assumption', 'sidebar_')
    st.caption("Income thresholds use BC Housing data (StatsCan T1 Family File)")

    st.subheader("Target")
    equity_return_required = create_param_input('equity_return_required', 'sidebar_')

    st.subheader("Housing Charge Scenario")
    scenario_descriptions = {
        'conservative': 'Conservative (2% flat)',
        'moderate': 'Moderate (2.5% → 2%)',
        'aggressive': 'Aggressive (3% → 2%)'
    }
    selected_scenario = st.selectbox(
        "Rate Schedule",
        options=list(HOUSING_CHARGE_SCENARIOS.keys()),
        format_func=lambda x: scenario_descriptions[x],
        index=1,  # Default to 'moderate'
        key="sidebar_scenario",
        help="Housing charge increase rates over 25 years. Rates typically moderate over time."
    )
    # Show the rate schedule
    rates = HOUSING_CHARGE_SCENARIOS[selected_scenario]
    st.caption(f"Yrs 1-5: {rates[0]:.1%} | Yrs 6-10: {rates[5]:.1%} | Yrs 11-15: {rates[10]:.1%} | Yrs 16-25: {rates[15]:.1%}")


# ============================================
# BUILD PARAMS DICT
# ============================================

# Affordability parameters (fixed - income thresholds are managed in Income Distributions tab)
affordability_ratio = 0.30  # 30% rent burden threshold
target_percentile = 50  # Median income

# Fixed parameters for viability calculations (not from listings)
fixed_params = {
    'opex_ratio': opex_ratio,
    'housing_charge_increase': HOUSING_CHARGE_SCENARIOS[selected_scenario][0],  # Year 1 rate for display
    'interest_rate_senior': interest_rate_senior,
    'interest_rate_secondary': PARAM_INFO['interest_rate_secondary']['default'],
    'max_ltv': max_ltv,
    'risk_free_rate': PARAM_INFO['risk_free_rate']['default'],
    'inflation_assumption': inflation_assumption,
    'affordability_ratio': affordability_ratio,
    'target_percentile': target_percentile,
    'equity_return_required': equity_return_required,
    # Placeholders - will be overwritten per listing
    'acquisition_cost_psf': PARAM_INFO['acquisition_cost_psf']['default'],
    'rent_psf': PARAM_INFO['rent_psf']['default'],
    # median_income is looked up per listing city
}

# Rate schedule for the selected scenario
rate_schedule = HOUSING_CHARGE_SCENARIOS[selected_scenario]

assumptions = {
    'opex_ratio': opex_ratio
}


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
        rent_growth *= (1 + rate_schedule[min(y, len(rate_schedule) - 1)])
    annual_rent_year_n = annual_rent_year0 * rent_growth

    # Income grows by inflation
    income_growth = (1 + inflation_assumption) ** year_slider
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
                rent_growth *= (1 + rate_schedule[min(y, len(rate_schedule) - 1)])
            annual_rent = annual_rent_year0 * rent_growth

            income_growth = (1 + inflation_assumption) ** year
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
    if not st.session_state.listings:
        st.info("No listings loaded. Upload a CSV or load sample data from the sidebar.")
    else:
        # Project selector
        st.subheader("Select Project")
        listing_options = {l.display_name: l for l in st.session_state.listings}
        option_list = list(listing_options.keys())

        # Check if we should select a specific listing (e.g., after adding new one)
        if '_select_listing' in st.session_state:
            target = st.session_state.pop('_select_listing')
            if target in option_list:
                st.session_state['deep_dive_select'] = target

        select_col, delete_col = st.columns([4, 1])
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

        if selected_name:
            listing = listing_options[selected_name]

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
                new_sqft = st.number_input("Sq Ft", value=int(listing.building_sqft or 0), min_value=0, max_value=1000000, step=1000, key=f"ld_sqft_{listing.id}_v{wv}")
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

                # Show totals
                total_cols = st.columns(5)
                with total_cols[0]:
                    st.metric("Total Units", rent_roll.total_units)
                with total_cols[1]:
                    st.metric("Total Sq Ft", f"{rent_roll.total_sqft:,.0f}")
                with total_cols[2]:
                    st.metric("Annual Revenue", f"${rent_roll.total_annual_revenue:,.0f}")
                with total_cols[3]:
                    st.metric("Avg Rent/SqFt", f"${rent_roll.weighted_avg_rent_psf:.2f}")
                with total_cols[4]:
                    noi_from_roll = rent_roll.total_annual_revenue * (1 - opex_ratio)
                    st.metric("NOI", f"${noi_from_roll:,.0f}")

                # Detailed rent roll calculations
                with st.expander("View Rent Roll Calculations"):
                    st.markdown("**Revenue by Unit Type:**")
                    if rent_roll.total_units > 0:
                        rr_calc_data = []
                        for ut in rent_roll.unit_types:
                            if ut.count > 0:
                                annual_rev = ut.monthly_rent * ut.count * 12
                                total_sqft = ut.sqft * ut.count
                                rr_calc_data.append({
                                    'Unit Type': ut.name,
                                    'Beds': ut.bedrooms,
                                    'Sq Ft/Unit': f"{ut.sqft:,.0f}",
                                    'Monthly Rent': f"${ut.monthly_rent:,.0f}",
                                    '# Units': ut.count,
                                    'Total Sq Ft': f"{total_sqft:,.0f}",
                                    'Monthly Revenue': f"${ut.monthly_rent * ut.count:,.0f}",
                                    'Annual Revenue': f"${annual_rev:,.0f}",
                                    'Rent/SqFt': f"${ut.monthly_rent / ut.sqft:.2f}",
                                })
                        rr_calc_df = pd.DataFrame(rr_calc_data)
                        st.dataframe(rr_calc_df, use_container_width=True, hide_index=True)

                        st.markdown("**Totals Calculation:**")
                        totals_data = [
                            ("Total Units", f"{rent_roll.total_units}", "Sum of # Units"),
                            ("Total Sq Ft", f"{rent_roll.total_sqft:,.0f}", "Sum of (Sq Ft/Unit × # Units)"),
                            ("Gross Annual Revenue", f"${rent_roll.total_annual_revenue:,.0f}", "Sum of Annual Revenue"),
                            ("Weighted Avg Rent/SqFt", f"${rent_roll.weighted_avg_rent_psf:.2f}", "Total Revenue ÷ Total SqFt ÷ 12"),
                            ("Avg Unit Size", f"{rent_roll.avg_unit_sqft:,.0f} sqft", "Total SqFt ÷ Total Units"),
                            ("Operating Expenses", f"${rent_roll.total_annual_revenue * opex_ratio:,.0f}", f"Revenue × {opex_ratio:.0%}"),
                            ("Net Operating Income", f"${noi_from_roll:,.0f}", f"Revenue × {(1-opex_ratio):.0%}"),
                        ]
                        totals_df = pd.DataFrame(totals_data, columns=["Metric", "Value", "Calculation"])
                        st.dataframe(totals_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("Add units to the rent roll to see calculations.")

                st.divider()

                # ---- Affordability by Unit Type ----
                st.subheader("Affordability by Unit Type")
                st.caption("Income thresholds from Income Distributions tab (based on BC Housing data)")

                if rent_roll.total_units == 0:
                    st.info("Add units to the rent roll to see affordability analysis.")
                else:
                    # Get income thresholds from session state (set in Income Distributions tab)
                    couples_income = st.session_state.get('income_dist_couples', BC_HOUSING_DEFAULTS['couples_without_children'])
                    families_income = st.session_state.get('income_dist_families', BC_HOUSING_DEFAULTS['families_with_children'])

                    # Calculate affordability for each unit type
                    affordability_rows = []
                    for ut in rent_roll.unit_types:
                        if ut.count == 0:
                            continue

                        # Determine family type based on number of bedrooms
                        # 0-1 bedrooms (Studio/1BR): Couples without children
                        # 2+ bedrooms: Families with children
                        if ut.bedrooms < 2:
                            low_mod_income = couples_income['median']
                            middle_income = couples_income['75th_percentile']
                            family_type = 'Couples (no children)'
                        else:
                            low_mod_income = families_income['median']
                            middle_income = families_income['75th_percentile']
                            family_type = 'Families (with children)'

                        annual_rent = ut.monthly_rent * 12
                        burden_low_mod = annual_rent / low_mod_income if low_mod_income > 0 else 0
                        burden_middle = annual_rent / middle_income if middle_income > 0 else 0

                        affordability_rows.append({
                            'unit_type': ut.name,
                            'bedrooms': ut.bedrooms,
                            'sqft': ut.sqft,
                            'monthly_rent': ut.monthly_rent,
                            'count': ut.count,
                            'family_type': family_type,
                            'low_mod_income': low_mod_income,
                            'middle_income': middle_income,
                            'burden_low_mod': burden_low_mod,
                            'burden_middle': burden_middle,
                            'affordable_low_mod': burden_low_mod <= affordability_ratio,
                            'affordable_middle': burden_middle <= affordability_ratio,
                        })

                    if affordability_rows:
                        aff_df = pd.DataFrame(affordability_rows)

                        # Show income thresholds being used (from Income Distributions tab)
                        st.markdown("**Income Thresholds Used** *(editable in Income Distributions tab)*:")
                        thresh_col1, thresh_col2 = st.columns(2)
                        with thresh_col1:
                            st.markdown(f"**Low & Moderate (Median)**")
                            st.text(f"Studio/1BR (0-1 beds): ${couples_income['median']:,.0f}")
                            st.text(f"2BR+ (2+ beds): ${families_income['median']:,.0f}")
                        with thresh_col2:
                            st.markdown(f"**Middle Income (75th %ile)**")
                            st.text(f"Studio/1BR (0-1 beds): ${couples_income['75th_percentile']:,.0f}")
                            st.text(f"2BR+ (2+ beds): ${families_income['75th_percentile']:,.0f}")

                        st.markdown("")

                        # Format for display
                        display_df = pd.DataFrame({
                            'Unit Type': aff_df['unit_type'],
                            'Beds': aff_df['bedrooms'],
                            'Sq Ft': aff_df['sqft'].astype(int),
                            'Rent': aff_df['monthly_rent'].apply(lambda x: f"${x:,.0f}"),
                            '# Units': aff_df['count'],
                            'Low/Mod Burden': aff_df['burden_low_mod'].apply(lambda x: f"{x:.1%}"),
                            'Middle Burden': aff_df['burden_middle'].apply(lambda x: f"{x:.1%}"),
                        })

                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                        # Summary metrics
                        total_units_aff = sum(r['count'] for r in affordability_rows)
                        affordable_low_mod = sum(r['count'] for r in affordability_rows if r['affordable_low_mod'])
                        affordable_middle = sum(r['count'] for r in affordability_rows if r['affordable_middle'])

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

                        # Detailed calculation table
                        with st.expander("View Affordability Calculations"):
                            st.markdown(f"""
                            **Methodology:**
                            - Rent Burden = Annual Rent ÷ Income Threshold
                            - Affordable if Rent Burden ≤ {affordability_ratio:.0%}
                            - Studio/1BR (0-1 bedrooms) use "Couples without Children" thresholds
                            - 2BR+ (2+ bedrooms) use "Families with Children" thresholds
                            """)

                            calc_df = pd.DataFrame({
                                'Unit Type': aff_df['unit_type'],
                                'Beds': aff_df['bedrooms'],
                                'Sq Ft': aff_df['sqft'].astype(int),
                                'Monthly Rent': aff_df['monthly_rent'].apply(lambda x: f"${x:,.0f}"),
                                'Annual Rent': (aff_df['monthly_rent'] * 12).apply(lambda x: f"${x:,.0f}"),
                                'Family Type': aff_df['family_type'],
                                'Low/Mod Income': aff_df['low_mod_income'].apply(lambda x: f"${x:,.0f}"),
                                'Middle Income': aff_df['middle_income'].apply(lambda x: f"${x:,.0f}"),
                                'Low/Mod Burden': aff_df['burden_low_mod'].apply(lambda x: f"{x:.1%}"),
                                'Middle Burden': aff_df['burden_middle'].apply(lambda x: f"{x:.1%}"),
                                'Affordable (L/M)': aff_df['affordable_low_mod'].apply(lambda x: "Yes" if x else "No"),
                                'Affordable (Mid)': aff_df['affordable_middle'].apply(lambda x: "Yes" if x else "No"),
                            })
                            st.dataframe(calc_df, use_container_width=True, hide_index=True)

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

                    # NOI and cash flow
                    noi = rent_roll.total_annual_revenue * (1 - opex_ratio)
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
                    base_opex = rent_roll.total_annual_revenue * opex_ratio  # Year 0 operating expenses
                    adjusted_time_series = []
                    for year_data in time_series_for_irr:
                        year = year_data['year']
                        if year == 0:
                            year_data['annual_revenue'] = rent_roll.total_annual_revenue
                            year_data['opex'] = base_opex
                            year_data['noi'] = noi
                            year_data['cash_flow'] = cash_flow
                            year_data['equity'] = equity
                        else:
                            # Revenue grows by housing charge schedule
                            growth_factor = year_data['rent_psf'] / base_rent_psf if base_rent_psf > 0 else 1
                            year_data['annual_revenue'] = rent_roll.total_annual_revenue * growth_factor
                            # Opex grows by inflation each year
                            year_data['opex'] = base_opex * ((1 + inflation_assumption) ** year)
                            year_data['noi'] = year_data['annual_revenue'] - year_data['opex']
                            year_data['cash_flow'] = year_data['noi'] - annual_debt_service
                        adjusted_time_series.append(year_data)

                    cash_flows_for_irr = [m['cash_flow'] for m in adjusted_time_series if m['year'] > 0]
                    irr_25yr = calculate_project_irr(equity, cash_flows_for_irr)
                    meets_irr_target = irr_25yr >= equity_return_required

                    # Key metrics
                    metric_cols = st.columns(5)
                    with metric_cols[0]:
                        st.metric("Acquisition $/SqFt", f"${params['acquisition_cost_psf']:.0f}")
                    with metric_cols[1]:
                        st.metric("DSCR", f"{dscr:.2f}",
                                  delta="OK" if dscr_ok else "Below 1.0",
                                  delta_color="normal" if dscr_ok else "inverse")
                    with metric_cols[2]:
                        st.metric("Year 1 Cash Flow", f"${cash_flow:,.0f}")
                    with metric_cols[3]:
                        st.metric("25-Year IRR", f"{irr_25yr:.1%}",
                                  delta=f"vs {equity_return_required:.1%} target",
                                  delta_color="normal" if meets_irr_target else "inverse")
                    with metric_cols[4]:
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
                        fin_calc_data = [
                            ("Building Square Footage", f"{building_sqft:,.0f} sqft", "From rent roll"),
                            ("Acquisition Cost per SqFt", f"${params['acquisition_cost_psf']:,.2f}", "Price ÷ SqFt"),
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
                        opex_amount = rent_roll.total_annual_revenue * opex_ratio
                        cf_calc_data = [
                            ("Gross Revenue", f"${rent_roll.total_annual_revenue:,.0f}", "Sum of all unit rents × 12"),
                            ("Operating Expense Ratio", f"{opex_ratio:.0%}", "Model parameter"),
                            ("Operating Expenses", f"${opex_amount:,.0f}", f"${rent_roll.total_annual_revenue:,.0f} × {opex_ratio:.0%}"),
                            ("Net Operating Income (NOI)", f"${noi:,.0f}", f"${rent_roll.total_annual_revenue:,.0f} - ${opex_amount:,.0f}"),
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

