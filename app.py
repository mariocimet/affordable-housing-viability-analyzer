"""
Affordable Housing Viability Analyzer

Single-page Streamlit app for analyzing affordable housing project viability
and income affordability bands.
"""

import streamlit as st
import pandas as pd
import numpy as np

from models.income_distribution import BCIncomeDistribution, calculate_band_over_time
from models.project_viability import PARAM_INFO, ProjectParameters, ProjectViabilityModel
from visualizations.income_band import create_income_band_chart, create_band_summary_chart
from visualizations.viability_space import create_single_viability_chart

st.set_page_config(
    page_title="Affordable Housing Analyzer",
    page_icon="🏠",
    layout="wide"
)

st.title("Affordable Housing Viability Analyzer")


# ============================================
# SIDEBAR: All Parameters by Category
# ============================================

def create_ratio_input(label, numerator_label, denominator_label,
                       default_ratio, default_num, default_denom,
                       ratio_format="%.2f", num_format="%.0f", denom_format="%.0f",
                       key_prefix=""):
    """
    Create flexible ratio input where user can specify:
    - The ratio directly
    - OR the numerator and denominator (system calculates ratio)
    """
    st.markdown(f"**{label}**")

    input_mode = st.radio(
        f"Input mode for {label}",
        ["Ratio", "Components"],
        horizontal=True,
        key=f"{key_prefix}_mode",
        label_visibility="collapsed"
    )

    if input_mode == "Ratio":
        ratio = st.number_input(
            label,
            value=float(default_ratio),
            format=ratio_format,
            key=f"{key_prefix}_ratio",
            label_visibility="collapsed"
        )
        # Calculate implied components
        numerator = ratio * default_denom
        denominator = default_denom
    else:
        col1, col2 = st.columns(2)
        with col1:
            numerator = st.number_input(
                numerator_label,
                value=float(default_num),
                format=num_format,
                key=f"{key_prefix}_num"
            )
        with col2:
            denominator = st.number_input(
                denominator_label,
                value=float(default_denom),
                format=denom_format,
                key=f"{key_prefix}_denom"
            )
        ratio = numerator / denominator if denominator > 0 else 0

    return ratio, numerator, denominator


def create_param_input(param_key, prefix="", allow_unspecified=False):
    """Create appropriate input for a parameter with optional unspecified state."""
    info = PARAM_INFO[param_key]
    key = f"{prefix}{param_key}"

    if allow_unspecified:
        is_specified = st.checkbox(
            f"Specify {info['name']}",
            value=True,
            key=f"{key}_specified"
        )
        if not is_specified:
            return None

    # Check if this is a precise input parameter (rates with 0.1% precision)
    if info.get('precise_input'):
        # Use number_input with 0.1% precision for rates
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
        # Percentile inputs (0-100 integer)
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


# Initialize session state for synchronized values
if 'acquisition_cost_psf' not in st.session_state:
    st.session_state.acquisition_cost_psf = 400.0
if 'total_sqft' not in st.session_state:
    st.session_state.total_sqft = 50000.0
if 'total_acquisition' not in st.session_state:
    st.session_state.total_acquisition = 400.0 * 50000.0
if 'rent_psf' not in st.session_state:
    st.session_state.rent_psf = 2.50
if 'sqft_per_unit' not in st.session_state:
    st.session_state.sqft_per_unit = 800.0
if 'monthly_unit_rent' not in st.session_state:
    st.session_state.monthly_unit_rent = 2.50 * 800.0


def update_acquisition_from_psf():
    """Update total acquisition when cost per sqft changes."""
    st.session_state.total_acquisition = st.session_state.acq_psf_input * st.session_state.total_sqft


def update_acquisition_from_total():
    """Update cost per sqft when total acquisition changes."""
    if st.session_state.total_sqft > 0:
        st.session_state.acquisition_cost_psf = st.session_state.total_acq_input / st.session_state.total_sqft


def update_acquisition_from_sqft_psf():
    """Update total acquisition when sqft changes (in psf mode)."""
    st.session_state.total_sqft = st.session_state.total_sqft_input
    st.session_state.total_acquisition = st.session_state.acquisition_cost_psf * st.session_state.total_sqft


def update_acquisition_from_sqft_total():
    """Update cost per sqft when sqft changes (in total mode)."""
    st.session_state.total_sqft = st.session_state.total_sqft_input_2
    if st.session_state.total_sqft > 0:
        st.session_state.acquisition_cost_psf = st.session_state.total_acquisition / st.session_state.total_sqft


def update_rent_from_psf():
    """Update monthly rent when rent per sqft changes."""
    st.session_state.monthly_unit_rent = st.session_state.rent_psf_input * st.session_state.sqft_per_unit


def update_rent_from_unit():
    """Update rent per sqft when monthly unit rent changes."""
    if st.session_state.sqft_per_unit > 0:
        st.session_state.rent_psf = st.session_state.unit_rent_input / st.session_state.sqft_per_unit


def update_rent_from_sqft_psf_mode():
    """Update rent values when sqft per unit changes (in Per Sq Ft mode)."""
    st.session_state.sqft_per_unit = st.session_state.sqft_per_unit_input
    st.session_state.monthly_unit_rent = st.session_state.rent_psf * st.session_state.sqft_per_unit


def update_rent_from_sqft_unit_mode():
    """Update rent values when sqft per unit changes (in Unit Rent mode)."""
    st.session_state.sqft_per_unit = st.session_state.sqft_per_unit_input_2
    if st.session_state.sqft_per_unit > 0:
        st.session_state.rent_psf = st.session_state.monthly_unit_rent / st.session_state.sqft_per_unit


with st.sidebar:
    st.header("Model Parameters")

    # ============================================
    # Project Parameters with Ratio Inputs
    # ============================================
    st.subheader("Project")

    # Acquisition Cost - ratio input
    st.markdown("**Cost per Sq Ft**")
    acq_mode = st.radio("Acquisition input", ["Per Sq Ft", "Total Cost"],
                        horizontal=True, key="acq_mode", label_visibility="collapsed")

    if acq_mode == "Per Sq Ft":
        st.number_input(
            "$/sq ft",
            value=st.session_state.acquisition_cost_psf,
            step=10.0,
            key="acq_psf_input",
            on_change=update_acquisition_from_psf
        )
        st.session_state.acquisition_cost_psf = st.session_state.acq_psf_input

        st.number_input(
            "Total Sq Ft",
            value=st.session_state.total_sqft,
            step=1000.0,
            key="total_sqft_input",
            on_change=update_acquisition_from_sqft_psf
        )
        st.session_state.total_sqft = st.session_state.total_sqft_input
        st.session_state.total_acquisition = st.session_state.acquisition_cost_psf * st.session_state.total_sqft
    else:
        st.number_input(
            "Total Acquisition Cost ($)",
            value=st.session_state.total_acquisition,
            step=100000.0,
            format="%.0f",
            key="total_acq_input",
            on_change=update_acquisition_from_total
        )
        st.session_state.total_acquisition = st.session_state.total_acq_input

        st.number_input(
            "Total Sq Ft",
            value=st.session_state.total_sqft,
            step=1000.0,
            key="total_sqft_input_2",
            on_change=update_acquisition_from_sqft_total
        )
        st.session_state.total_sqft = st.session_state.total_sqft_input_2
        if st.session_state.total_sqft > 0:
            st.session_state.acquisition_cost_psf = st.session_state.total_acquisition / st.session_state.total_sqft

    acquisition_cost_psf = st.session_state.acquisition_cost_psf
    total_sqft = st.session_state.total_sqft
    total_acquisition = st.session_state.total_acquisition

    st.caption(f"= ${acquisition_cost_psf:,.2f}/sq ft | ${total_acquisition:,.0f} total")

    # Rent - ratio input
    st.markdown("**Rent per Sq Ft (Monthly)**")
    rent_mode = st.radio("Rent input", ["Per Sq Ft", "Unit Rent"],
                         horizontal=True, key="rent_mode", label_visibility="collapsed")

    if rent_mode == "Per Sq Ft":
        st.number_input(
            "$/sq ft/mo",
            value=st.session_state.rent_psf,
            step=0.10,
            key="rent_psf_input",
            on_change=update_rent_from_psf
        )
        st.session_state.rent_psf = st.session_state.rent_psf_input

        st.number_input(
            "Sq Ft per Unit",
            value=st.session_state.sqft_per_unit,
            step=10.0,
            key="sqft_per_unit_input",
            on_change=update_rent_from_sqft_psf_mode
        )
        st.session_state.sqft_per_unit = st.session_state.sqft_per_unit_input
        st.session_state.monthly_unit_rent = st.session_state.rent_psf * st.session_state.sqft_per_unit
    else:
        st.number_input(
            "Monthly Unit Rent ($)",
            value=st.session_state.monthly_unit_rent,
            step=50.0,
            key="unit_rent_input",
            on_change=update_rent_from_unit
        )
        st.session_state.monthly_unit_rent = st.session_state.unit_rent_input

        st.number_input(
            "Sq Ft per Unit",
            value=st.session_state.sqft_per_unit,
            step=10.0,
            key="sqft_per_unit_input_2",
            on_change=update_rent_from_sqft_unit_mode
        )
        st.session_state.sqft_per_unit = st.session_state.sqft_per_unit_input_2
        if st.session_state.sqft_per_unit > 0:
            st.session_state.rent_psf = st.session_state.monthly_unit_rent / st.session_state.sqft_per_unit

    rent_psf = st.session_state.rent_psf
    sqft_per_unit = st.session_state.sqft_per_unit
    monthly_unit_rent = st.session_state.monthly_unit_rent

    st.caption(f"= ${rent_psf:,.2f}/sq ft | ${monthly_unit_rent:,.0f}/unit")

    opex_ratio = create_param_input('opex_ratio', 'sidebar_')
    housing_charge_increase = create_param_input('housing_charge_increase', 'sidebar_')

    # ============================================
    # Lender Parameters
    # ============================================
    st.subheader("Lender")
    interest_rate_senior = create_param_input('interest_rate_senior', 'sidebar_')
    interest_rate_secondary = create_param_input('interest_rate_secondary', 'sidebar_')
    max_ltv = create_param_input('max_ltv', 'sidebar_')

    # ============================================
    # Economic Parameters
    # ============================================
    st.subheader("Economic")
    median_income = create_param_input('median_income', 'sidebar_')
    risk_free_rate = create_param_input('risk_free_rate', 'sidebar_')
    inflation_assumption = create_param_input('inflation_assumption', 'sidebar_')

    # ============================================
    # Conventional Parameters
    # ============================================
    st.subheader("Conventional")
    affordability_ratio_lower = create_param_input('affordability_ratio_lower', 'sidebar_')
    affordability_ratio_upper = create_param_input('affordability_ratio_upper', 'sidebar_')
    # sqft_per_unit already captured above

    # ============================================
    # Target Returns
    # ============================================
    st.subheader("Target")
    equity_return_required = create_param_input('equity_return_required', 'sidebar_')


# ============================================
# MAIN CONTENT
# ============================================

# Calculate derived values
monthly_rent = rent_psf * sqft_per_unit
distribution = BCIncomeDistribution(median=median_income)

# Build params dict for viability calculations
all_params = {
    'acquisition_cost_psf': acquisition_cost_psf,
    'rent_psf': rent_psf,
    'opex_ratio': opex_ratio,
    'housing_charge_increase': housing_charge_increase,
    'interest_rate_senior': interest_rate_senior,
    'interest_rate_secondary': interest_rate_secondary,
    'max_ltv': max_ltv,
    'median_income': median_income,
    'risk_free_rate': risk_free_rate,
    'inflation_assumption': inflation_assumption,
    'affordability_ratio_lower': affordability_ratio_lower,
    'affordability_ratio_upper': affordability_ratio_upper,
    'sqft_per_unit': sqft_per_unit,
    'equity_return_required': equity_return_required,
}

# Project metrics
params = ProjectParameters(**all_params)
model = ProjectViabilityModel(params)
is_viable, viability_metrics = model.is_viable()
metrics = model.calculate_metrics(total_sqft=total_sqft)

# Affordability band - percentiles are calculated from the affordability ratio bounds
min_income, max_income_band, lower_pct, upper_pct = distribution.affordability_band(
    monthly_rent, affordability_ratio_lower, affordability_ratio_upper
)
band_width = max(0, upper_pct - lower_pct)

# ============================================
# ROW 1: Key Metrics
# ============================================
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Monthly Unit Rent", f"${monthly_rent:,.0f}")
with col2:
    st.metric("Min Income Required", f"${min_income:,.0f}")
with col3:
    st.metric("Households Served", f"{band_width:.1f}%", delta=f"{lower_pct:.0f}-{upper_pct:.0f}th %ile")
with col4:
    st.metric("DSCR", f"{metrics['dscr']:.2f}")
with col5:
    cash_on_cash = metrics['cash_flow'] / metrics['equity'] if metrics['equity'] > 0 else 0
    st.metric("Cash-on-Cash", f"{cash_on_cash:.1%}")
with col6:
    viable_text = "Viable" if is_viable else "Not Viable"
    st.metric("Status", viable_text)

st.divider()

# ============================================
# ROW 2: Two Main Visualizations
# ============================================
viz_col1, viz_col2 = st.columns(2)

# LEFT: Income Band Analysis
with viz_col1:
    st.subheader("Income Distribution & Affordability Band")

    # Escalation for animation
    years = 25
    escalation_rates = [housing_charge_increase] * years

    fig1 = create_income_band_chart(
        distribution=distribution,
        monthly_rent=monthly_rent,
        affordability_ratio_lower=affordability_ratio_lower,
        affordability_ratio_upper=affordability_ratio_upper,
        escalation_rates=escalation_rates,
        show_animation=True,
        income_growth_rate=inflation_assumption
    )
    st.plotly_chart(fig1, width="stretch")

# RIGHT: Project Viability Space
with viz_col2:
    st.subheader("Project Viability Space")

    # Axis selection
    variable_params = list(PARAM_INFO.keys())
    param_names = {k: f"{v['name']}" for k, v in PARAM_INFO.items()}

    ax_col1, ax_col2 = st.columns(2)
    with ax_col1:
        x_param = st.selectbox(
            "X-Axis",
            variable_params,
            index=variable_params.index('acquisition_cost_psf'),
            format_func=lambda x: param_names[x],
            key="x_axis_select"
        )
    with ax_col2:
        y_options = [p for p in variable_params if p != x_param]
        y_param = st.selectbox(
            "Y-Axis",
            y_options,
            index=y_options.index('rent_psf') if 'rent_psf' in y_options else 0,
            format_func=lambda x: param_names[x],
            key="y_axis_select"
        )

    # Get ranges
    x_info = PARAM_INFO[x_param]
    y_info = PARAM_INFO[y_param]
    x_range = (x_info['default_range'][0], x_info['default_range'][1])
    y_range = (y_info['default_range'][0], y_info['default_range'][1])

    fig2 = create_single_viability_chart(
        x_param=x_param,
        y_param=y_param,
        fixed_params=all_params,
        x_range=x_range,
        y_range=y_range,
        projects=None
    )
    st.plotly_chart(fig2, width="stretch")

st.divider()

# ============================================
# ROW 3: Band Evolution Over Time
# ============================================
st.subheader("Affordability Band Evolution Over Time")

bands = calculate_band_over_time(
    distribution, monthly_rent, escalation_rates,
    affordability_ratio_lower, affordability_ratio_upper, inflation_assumption
)
summary_fig = create_band_summary_chart(bands)
st.plotly_chart(summary_fig, width="stretch")

# ============================================
# ROW 4: Financial Summary
# ============================================
with st.expander("Financial Summary"):
    fin_col1, fin_col2 = st.columns(2)

    with fin_col1:
        st.markdown("**Project Costs**")
        st.text(f"Total Acquisition: ${metrics['acquisition_cost']:,.0f}")
        st.text(f"Senior Loan: ${metrics['loan_amount']:,.0f}")
        st.text(f"Equity Required: ${metrics['equity']:,.0f}")

    with fin_col2:
        st.markdown("**Annual Cash Flows**")
        st.text(f"Gross Revenue: ${metrics['annual_revenue']:,.0f}")
        st.text(f"Net Operating Income: ${metrics['noi']:,.0f}")
        st.text(f"Debt Service: ${metrics['annual_debt_service']:,.0f}")
        st.text(f"Cash Flow: ${metrics['cash_flow']:,.0f}")
