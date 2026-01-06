"""
Affordable Housing Viability Analyzer

Two-tab Streamlit app for analyzing affordable housing project viability:
- Tab 1: Portfolio Overview - see all listings with viability status
- Tab 2: Project Deep-Dive - detailed analysis of a single project
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from models.income_distribution import BCIncomeDistribution, calculate_band_over_time
from models.project_viability import PARAM_INFO, ProjectParameters, ProjectViabilityModel, HOUSING_CHARGE_SCENARIOS, calculate_project_irr
from models.param_mapper import analyze_portfolio, listing_to_params
from models.listing import MultifamilyListing
from visualizations.income_band import create_income_band_chart, create_band_summary_chart
from visualizations.viability_space import (
    create_portfolio_scatter,
    create_time_series_chart,
    create_cash_flow_chart,
    create_cost_vs_burden_chart,
    create_max_viable_cost_chart,
    create_subsidy_required_chart,
    create_income_percentile_required_chart,
    create_sensitivity_tornado_chart,
    create_affordability_frontier_chart
)

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


def load_listings_from_csv(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of MultifamilyListing objects."""
    listings = []
    for idx, row in df.iterrows():
        try:
            listing = MultifamilyListing(
                id=str(row.get('id', f"listing_{idx}")),
                source=str(row.get('source', 'csv')),
                address=str(row['address']),
                city=str(row['city']),
                asking_price=float(row['asking_price']),
                building_sqft=float(row['building_sqft']) if pd.notna(row.get('building_sqft')) else None,
                num_units=int(row['num_units']) if pd.notna(row.get('num_units')) else None,
                year_built=int(row['year_built']) if pd.notna(row.get('year_built')) else None,
                cap_rate=float(row['cap_rate']) if pd.notna(row.get('cap_rate')) else None,
                noi=float(row['noi']) if pd.notna(row.get('noi')) else None,
                url=str(row['url']) if pd.notna(row.get('url')) else None,
                listing_status=str(row.get('listing_status', 'active'))
            )
            listings.append(listing)
        except Exception as e:
            st.warning(f"Skipped row {idx}: {e}")
    return listings


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    # ============================================
    # Listings Section (Top Priority)
    # ============================================
    st.header("Listings")
    st.caption("BC multifamily properties (20+ units)")

    if 'listings' not in st.session_state:
        st.session_state.listings = []

    uploaded_file = st.file_uploader("Upload listings CSV", type=['csv'], key="csv_upload")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.listings = load_listings_from_csv(df)
            st.success(f"Loaded {len(st.session_state.listings)} listings")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")

    default_csv = os.path.join(os.path.dirname(__file__), 'data', 'synthetic_listings.csv')
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(default_csv):
            if st.button("Load Sample Data", help="Synthetic listings for demonstration"):
                df = pd.read_csv(default_csv)
                st.session_state.listings = load_listings_from_csv(df)
                st.success(f"Loaded {len(st.session_state.listings)} listings")
    with col2:
        if st.button("Clear"):
            st.session_state.listings = []

    st.caption(f"**{len(st.session_state.listings)} listings loaded**")

    # Manual listing entry
    with st.expander("Add Listing Manually"):
        with st.form("manual_listing_form"):
            ml_address = st.text_input("Address*", placeholder="123 Main St")
            ml_city = st.text_input("City*", placeholder="Vancouver")
            ml_price = st.number_input("Asking Price ($)*", min_value=100000, max_value=500000000, value=5000000, step=100000)
            ml_sqft = st.number_input("Building Sq Ft", min_value=0, max_value=1000000, value=0, step=1000, help="Leave 0 to estimate from units")
            ml_units = st.number_input("Number of Units", min_value=1, max_value=1000, value=20, step=1)
            ml_year = st.number_input("Year Built", min_value=1900, max_value=2025, value=1980, step=1)
            ml_cap = st.number_input("Cap Rate (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.1, help="Leave 0 to use market rent")

            submitted = st.form_submit_button("Add Listing")
            if submitted:
                if ml_address and ml_city:
                    new_id = f"MANUAL_{len(st.session_state.listings) + 1:03d}"
                    new_listing = MultifamilyListing(
                        id=new_id,
                        source='manual',
                        address=ml_address,
                        city=ml_city,
                        asking_price=float(ml_price),
                        building_sqft=float(ml_sqft) if ml_sqft > 0 else None,
                        num_units=int(ml_units),
                        year_built=int(ml_year),
                        cap_rate=float(ml_cap) / 100 if ml_cap > 0 else None,
                        url='',
                    )
                    st.session_state.listings.append(new_listing)
                    st.success(f"Added: {ml_address}")
                    st.rerun()
                else:
                    st.error("Address and City are required")

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
    median_income = create_param_input('median_income', 'sidebar_')
    inflation_assumption = create_param_input('inflation_assumption', 'sidebar_')

    st.subheader("Target")
    equity_return_required = create_param_input('equity_return_required', 'sidebar_')

    st.subheader("Affordability")
    affordability_ratio = create_param_input('affordability_ratio', 'sidebar_')
    target_percentile = create_param_input('target_percentile', 'sidebar_')

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

# Fixed parameters for viability calculations (not from listings)
fixed_params = {
    'opex_ratio': opex_ratio,
    'housing_charge_increase': HOUSING_CHARGE_SCENARIOS[selected_scenario][0],  # Year 1 rate for display
    'interest_rate_senior': interest_rate_senior,
    'interest_rate_secondary': PARAM_INFO['interest_rate_secondary']['default'],
    'max_ltv': max_ltv,
    'median_income': median_income,
    'risk_free_rate': PARAM_INFO['risk_free_rate']['default'],
    'inflation_assumption': inflation_assumption,
    'affordability_ratio': affordability_ratio,
    'target_percentile': target_percentile,
    'equity_return_required': equity_return_required,
    # Placeholders - will be overwritten per listing
    'acquisition_cost_psf': PARAM_INFO['acquisition_cost_psf']['default'],
    'rent_psf': PARAM_INFO['rent_psf']['default'],
}

# Rate schedule for the selected scenario
rate_schedule = HOUSING_CHARGE_SCENARIOS[selected_scenario]

assumptions = {
    'opex_ratio': opex_ratio
}


# ============================================
# MAIN CONTENT: TWO TABS
# ============================================

tab1, tab2 = st.tabs(["Portfolio Overview", "Project Deep-Dive"])


# ============================================
# TAB 1: PORTFOLIO OVERVIEW
# ============================================

with tab1:
    if not st.session_state.listings:
        st.info("No listings loaded. Upload a CSV or load sample data from the sidebar.")
    else:
        # Analyze full portfolio with selected housing charge scenario
        portfolio_df = analyze_portfolio(st.session_state.listings, fixed_params, assumptions, scenario=selected_scenario)

        # ---- Summary Metrics Row ----
        st.subheader("Portfolio Summary")

        total = len(portfolio_df)
        viable_count = len(portfolio_df[portfolio_df['is_viable'] == True])
        not_viable_count = len(portfolio_df[portfolio_df['is_viable'] == False])
        missing_data = total - viable_count - not_viable_count

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Listings", total)
        with col2:
            pct = f"{viable_count/total*100:.0f}%" if total > 0 else "0%"
            st.metric("Viable", viable_count, delta=pct)
        with col3:
            st.metric("Not Viable", not_viable_count)
        with col4:
            avg_burden = portfolio_df['rent_burden'].mean()
            st.metric("Avg Rent Burden", f"{avg_burden:.0%}" if pd.notna(avg_burden) else "N/A")
        with col5:
            avg_irr = portfolio_df['irr_25yr'].mean() if 'irr_25yr' in portfolio_df.columns else None
            st.metric("Avg 25-Yr IRR", f"{avg_irr:.1%}" if pd.notna(avg_irr) else "N/A")
        with col6:
            if total > 0:
                price_min = portfolio_df['asking_price'].min() / 1e6
                price_max = portfolio_df['asking_price'].max() / 1e6
                st.metric("Price Range", f"${price_min:.1f}M - ${price_max:.1f}M")
            else:
                st.metric("Price Range", "N/A")

        st.divider()

        # ---- Two Column Layout: Table + Chart ----
        table_col, chart_col = st.columns([1, 1])

        with table_col:
            st.subheader("Listings Table")

            # Prepare display DataFrame with ID for deletion
            display_cols = ['id', 'address', 'city', 'asking_price', 'num_units', 'building_sqft',
                           'acquisition_cost_psf', 'min_rent_psf', 'rent_burden', 'irr_25yr', 'viability_status']
            display_df = portfolio_df[[c for c in display_cols if c in portfolio_df.columns]].copy()

            # Add selection column
            display_df.insert(0, 'Select', False)

            # Format columns for display
            format_df = display_df.copy()
            format_df['Price'] = display_df['asking_price'].apply(lambda x: f"${x/1e6:.1f}M" if pd.notna(x) else "N/A")
            format_df['SqFt'] = display_df['building_sqft'].apply(lambda x: f"{x/1000:.0f}K" if pd.notna(x) else "N/A")
            format_df['$/SqFt'] = display_df['acquisition_cost_psf'].apply(lambda x: f"${x:.0f}" if pd.notna(x) else "N/A")
            format_df['Min Rent'] = display_df['min_rent_psf'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            format_df['Burden'] = display_df['rent_burden'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
            format_df['IRR'] = display_df['irr_25yr'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A") if 'irr_25yr' in display_df.columns else "N/A"

            # Select columns for display
            show_df = format_df[['Select', 'id', 'city', 'Price', 'num_units', '$/SqFt', 'Min Rent', 'IRR', 'viability_status']].copy()
            show_df = show_df.rename(columns={
                'city': 'City',
                'num_units': 'Units',
                'viability_status': 'Status',
                'id': 'ID'
            })

            edited_df = st.data_editor(
                show_df,
                use_container_width=True,
                height=350,
                hide_index=True,
                column_config={
                    'Select': st.column_config.CheckboxColumn(label='🗑️', width='small'),
                    'ID': st.column_config.TextColumn(width='small'),
                    'City': st.column_config.TextColumn(width='small'),
                    'Units': st.column_config.TextColumn(width='small'),
                    'IRR': st.column_config.TextColumn(width='small'),
                    'Status': st.column_config.TextColumn(width='medium'),
                },
                disabled=['ID', 'City', 'Price', 'Units', '$/SqFt', 'Min Rent', 'IRR', 'Status'],
                key="portfolio_table"
            )

            # Delete selected listings
            selected_ids = edited_df[edited_df['Select'] == True]['ID'].tolist()
            if selected_ids:
                if st.button(f"Delete {len(selected_ids)} selected", type="secondary"):
                    st.session_state.listings = [
                        l for l in st.session_state.listings if l.id not in selected_ids
                    ]
                    st.rerun()

        with chart_col:
            st.subheader("Viability Space")

            # Axis selection for viability space
            axis_options = {
                'acquisition_cost_psf': 'Acquisition Cost ($/sqft)',
                'rent_psf': 'Rent ($/sqft/month)',
                'sqft_per_unit': 'Unit Size (sqft)',
                'opex_ratio': 'Operating Expense Ratio',
                'interest_rate_senior': 'Interest Rate',
                'max_ltv': 'Loan-to-Value',
                'affordability_ratio': 'Affordability Ratio',
            }

            axis_col1, axis_col2 = st.columns(2)
            with axis_col1:
                x_param = st.selectbox(
                    "X-Axis",
                    options=list(axis_options.keys()),
                    format_func=lambda x: axis_options[x],
                    index=0,
                    key="viability_x_axis"
                )
            with axis_col2:
                y_param = st.selectbox(
                    "Y-Axis",
                    options=list(axis_options.keys()),
                    format_func=lambda x: axis_options[x],
                    index=1,
                    key="viability_y_axis"
                )

            fig = create_portfolio_scatter(
                portfolio_df,
                x_param=x_param,
                y_param=y_param,
                fixed_params=fixed_params
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- Analysis Charts ----
        st.divider()
        st.subheader("Analysis Charts")

        chart_options = {
            'cost_vs_burden': 'Cost vs Rent Burden',
            'max_viable_cost': 'Max Viable Acquisition Cost',
            'subsidy_required': 'Subsidy Required',
            'income_percentile': 'Income Percentile Required',
            'sensitivity': 'Sensitivity Analysis',
            'frontier': 'Affordability Frontier'
        }

        selected_chart = st.selectbox(
            "Select analysis view:",
            options=list(chart_options.keys()),
            format_func=lambda x: chart_options[x],
            key="analysis_chart_selector"
        )

        if selected_chart == 'cost_vs_burden':
            fig = create_cost_vs_burden_chart(portfolio_df, fixed_params)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows the direct relationship between acquisition cost and rent burden. "
                      "Higher cost requires higher rent to service debt, pushing up burden.")

        elif selected_chart == 'max_viable_cost':
            fig = create_max_viable_cost_chart(portfolio_df, fixed_params)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Compares each listing's acquisition cost to the maximum cost that would still be "
                      "affordable at target income/burden settings. Green = viable, Red = needs subsidy.")

        elif selected_chart == 'subsidy_required':
            fig = create_subsidy_required_chart(portfolio_df, fixed_params)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows how much acquisition cost reduction (subsidy per sqft) would be needed "
                      "to make each listing affordable. Percentages show subsidy as % of cost.")

        elif selected_chart == 'income_percentile':
            fig = create_income_percentile_required_chart(portfolio_df, fixed_params)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows what income percentile is required to afford each listing. "
                      "Lower = more accessible. Green listings meet the target percentile.")

        elif selected_chart == 'sensitivity':
            fig = create_sensitivity_tornado_chart(fixed_params)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Tornado chart showing which parameters have the biggest impact on rent burden. "
                      "Each parameter varied ±20% from current settings.")

        elif selected_chart == 'frontier':
            fig = create_affordability_frontier_chart(fixed_params)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The fundamental trade-off: higher acquisition costs require higher income "
                      "percentiles for affordability. The green star marks the maximum viable cost at target percentile.")

        st.divider()

        # ---- Data Source Notes ----
        with st.expander("About the Model"):
            st.markdown(f"""
**How it works:** The model takes a "patient investor" approach:
1. Calculate minimum rent for DSCR = 1.0 on Day 1 (covers debt service)
2. Project cash flows over 25 years using the selected housing charge scenario
3. Calculate IRR (Internal Rate of Return) from the cash flow stream
4. Check affordability at target income percentile

**Viability Criteria (all must be met):**
- DSCR ≥ 1.0 on Day 1 (rent covers debt service)
- 25-year IRR ≥ {equity_return_required:.1%} target
- Rent is ≤ {affordability_ratio:.0%} of income at {target_percentile:.0f}th percentile

**Housing Charge Scenario:** {scenario_descriptions[selected_scenario]}
- Even projects starting at DSCR=1.0 (breakeven) can generate returns through housing charge growth

**Loan Structure:** 20-year term, 50-year amortization (typical for affordable housing with CMHC MLI Select).

**Income Data:** BC household income distribution modeled as log-normal (median ~$95K CAD).
            """)
            if missing_data > 0:
                st.write(f"Note: {missing_data} listings have missing data and cannot be evaluated.")


# ============================================
# TAB 2: PROJECT DEEP-DIVE
# ============================================

with tab2:
    if not st.session_state.listings:
        st.info("No listings loaded. Upload a CSV or load sample data from the sidebar.")
    else:
        # Project selector
        st.subheader("Select Project")
        listing_options = {l.display_name: l for l in st.session_state.listings}

        select_col, delete_col = st.columns([4, 1])
        with select_col:
            selected_name = st.selectbox(
                "Choose a listing to analyze:",
                options=list(listing_options.keys()),
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
                    st.rerun()

        if selected_name:
            listing = listing_options[selected_name]

            # Get params for this listing
            params = listing_to_params(listing, assumptions)

            st.divider()

            # ---- Row 1: Listing Details ----
            st.subheader("Listing Details")
            detail_cols = st.columns(6)
            with detail_cols[0]:
                st.metric("Address", listing.address)
            with detail_cols[1]:
                st.metric("City", listing.city)
            with detail_cols[2]:
                st.metric("Price", f"${listing.asking_price/1e6:.2f}M")
            with detail_cols[3]:
                st.metric("Units", listing.num_units or "N/A")
            with detail_cols[4]:
                sqft_display = f"{listing.building_sqft:,.0f}" if listing.building_sqft else "Est."
                st.metric("Sq Ft", sqft_display)
            with detail_cols[5]:
                st.metric("Year Built", listing.year_built or "N/A")

            # Check if we can analyze
            if params.get('acquisition_cost_psf') is None:
                st.error("Cannot analyze this listing - missing required data (sqft)")
            else:
                # Import the function to calculate min rent
                from models.param_mapper import calculate_min_rent_for_dscr

                # Calculate minimum rent for DSCR = 1.0 on Day 1
                # Note: equity_return_required=0 because IRR is calculated over time
                min_rent_psf = calculate_min_rent_for_dscr(
                    acquisition_cost_psf=params['acquisition_cost_psf'],
                    interest_rate=interest_rate_senior,
                    max_ltv=max_ltv,
                    opex_ratio=opex_ratio,
                    amortization_years=50,
                    equity_return_required=0.0  # DSCR=1 only; IRR handles return target
                )

                # Check affordability at minimum rent
                viability_params = fixed_params.copy()
                viability_params['acquisition_cost_psf'] = params['acquisition_cost_psf']
                viability_params['rent_psf'] = min_rent_psf

                project = ProjectParameters(**viability_params)
                model = ProjectViabilityModel(project)
                is_affordable, aff_metrics = model.check_affordability()
                building_sqft = params.get('building_sqft', 10000)

                rent_burden = aff_metrics.get('rent_burden', 0)
                target_income = aff_metrics.get('target_income', 0)
                monthly_rent = aff_metrics.get('monthly_rent', 0)

                # Calculate 25-year IRR using selected scenario
                time_series_for_irr = model.calculate_metrics_over_time(
                    total_sqft=building_sqft,
                    years=25,
                    rate_schedule=rate_schedule
                )
                cash_flows_for_irr = [m['cash_flow'] for m in time_series_for_irr if m['year'] > 0]
                equity_for_irr = time_series_for_irr[0]['equity']
                irr_25yr = calculate_project_irr(equity_for_irr, cash_flows_for_irr)
                meets_irr_target = irr_25yr >= equity_return_required
                is_viable = is_affordable and meets_irr_target

                st.divider()

                # ---- Row 2: Viability Metrics ----
                st.subheader("Viability Analysis")

                # Key metrics
                metric_cols = st.columns(6)
                with metric_cols[0]:
                    st.metric("Acquisition $/SqFt", f"${params['acquisition_cost_psf']:.0f}")
                with metric_cols[1]:
                    st.metric("Min Rent $/SqFt", f"${min_rent_psf:.2f}",
                              help="Minimum rent needed for DSCR = 1.0")
                with metric_cols[2]:
                    unit_sqft = PARAM_INFO['sqft_per_unit']['default']
                    st.metric("Monthly Rent", f"${monthly_rent:,.0f}",
                              help=f"For {unit_sqft:.0f} sqft unit")
                with metric_cols[3]:
                    st.metric("Rent Burden", f"{rent_burden:.0%}",
                              delta=f"vs {affordability_ratio:.0%} max",
                              delta_color="inverse" if rent_burden > affordability_ratio else "normal")
                with metric_cols[4]:
                    st.metric("25-Year IRR", f"{irr_25yr:.1%}",
                              delta=f"vs {equity_return_required:.1%} target",
                              delta_color="normal" if meets_irr_target else "inverse")
                with metric_cols[5]:
                    status_text = "VIABLE" if is_viable else "NOT VIABLE"
                    st.metric("Status", status_text)

                # Viability check explanation
                aff_ok = "✓" if is_affordable else "✗"
                irr_ok = "✓" if meets_irr_target else "✗"
                st.write(f"{aff_ok} **Affordability:** At {target_percentile:.0f}th percentile income (${target_income:,.0f}), "
                         f"rent burden is {rent_burden:.0%} (threshold: {affordability_ratio:.0%})")
                st.write(f"{irr_ok} **Return:** 25-year IRR of {irr_25yr:.1%} (target: {equity_return_required:.1%})")

                st.divider()

                # ---- Row 3: Two-column layout ----
                left_col, right_col = st.columns(2)

                with left_col:
                    st.subheader("Viability Space")

                    # Axis selection
                    axis_options_dd = {
                        'acquisition_cost_psf': 'Acquisition Cost ($/sqft)',
                        'rent_psf': 'Rent ($/sqft/month)',
                        'sqft_per_unit': 'Unit Size (sqft)',
                        'opex_ratio': 'Operating Expense Ratio',
                        'interest_rate_senior': 'Interest Rate',
                        'max_ltv': 'Loan-to-Value',
                    }
                    dd_col1, dd_col2 = st.columns(2)
                    with dd_col1:
                        x_param_dd = st.selectbox("X", list(axis_options_dd.keys()),
                                                   format_func=lambda x: axis_options_dd[x],
                                                   index=0, key="dd_x_axis")
                    with dd_col2:
                        y_param_dd = st.selectbox("Y", list(axis_options_dd.keys()),
                                                   format_func=lambda x: axis_options_dd[x],
                                                   index=1, key="dd_y_axis")

                    # Analyze portfolio to show all listings
                    portfolio_df = analyze_portfolio(st.session_state.listings, fixed_params, assumptions, scenario=selected_scenario)

                    fig = create_portfolio_scatter(
                        portfolio_df,
                        x_param=x_param_dd,
                        y_param=y_param_dd,
                        fixed_params=fixed_params,
                        highlight_listing_id=listing.id
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with right_col:
                    st.subheader("Financial Breakdown (at DSCR=1)")

                    # Calculate metrics at min rent
                    full_metrics = model.calculate_metrics(total_sqft=building_sqft)

                    fin_col1, fin_col2 = st.columns(2)
                    with fin_col1:
                        st.markdown("**Project Costs**")
                        st.text(f"Total Acquisition: ${full_metrics['acquisition_cost']:,.0f}")
                        st.text(f"Senior Loan: ${full_metrics['loan_amount']:,.0f}")
                        st.text(f"Equity Required: ${full_metrics['equity']:,.0f}")

                    with fin_col2:
                        st.markdown("**Annual Cash Flows**")
                        st.text(f"Gross Revenue: ${full_metrics['annual_revenue']:,.0f}")
                        st.text(f"Net Operating Income: ${full_metrics['noi']:,.0f}")
                        st.text(f"Debt Service: ${full_metrics['annual_debt_service']:,.0f}")
                        st.text(f"Cash Flow (at DSCR=1): $0")

                st.divider()

                # ---- Row 4: Affordability Analysis ----
                st.subheader("Affordability Analysis")

                distribution = BCIncomeDistribution(median=median_income)

                # Calculate minimum income required at the affordability ratio
                annual_rent = monthly_rent * 12
                min_income_required = annual_rent / affordability_ratio
                min_income_percentile = distribution.percentile_at_income(min_income_required)

                aff_cols = st.columns(3)
                with aff_cols[0]:
                    st.metric("Min Monthly Rent", f"${monthly_rent:,.0f}")
                with aff_cols[1]:
                    st.metric("Min Income Required", f"${min_income_required:,.0f}",
                              help=f"Income where rent = {affordability_ratio:.0%} of income")
                with aff_cols[2]:
                    pct_can_afford = 100 - min_income_percentile
                    st.metric("Households Who Can Afford", f"{pct_can_afford:.1f}%")

                # Income band chart - use selected scenario rate schedule
                fig_income = create_income_band_chart(
                    distribution=distribution,
                    monthly_rent=monthly_rent,
                    affordability_ratio_lower=0.0,
                    affordability_ratio_upper=affordability_ratio,
                    escalation_rates=rate_schedule,
                    show_animation=True,
                    income_growth_rate=inflation_assumption
                )
                st.plotly_chart(fig_income, use_container_width=True)

                # Band evolution
                bands = calculate_band_over_time(
                    distribution, monthly_rent, rate_schedule,
                    0.0, affordability_ratio, inflation_assumption
                )
                summary_fig = create_band_summary_chart(bands)
                st.plotly_chart(summary_fig, use_container_width=True)

                st.divider()

                # ---- Row 5: Time-Series Financial Projection ----
                st.subheader("25-Year Financial Projection")

                st.markdown(f"""
                Projects starting at DSCR=1.0 can reward patient investors through housing charge growth.
                Using **{scenario_descriptions[selected_scenario]}** scenario.
                """)

                # Calculate time series with selected scenario
                time_series = model.calculate_metrics_over_time(
                    total_sqft=building_sqft,
                    years=25,
                    income_growth_rate=inflation_assumption,
                    rate_schedule=rate_schedule
                )

                year_25 = time_series[-1]
                ts_cols = st.columns(5)
                with ts_cols[0]:
                    st.metric("Year 1 DSCR", "1.00", help="DSCR starts at 1.0 by construction")
                with ts_cols[1]:
                    st.metric("Year 25 DSCR", f"{year_25['dscr']:.2f}")
                with ts_cols[2]:
                    st.metric("Year 25 Cash Flow", f"${year_25['cash_flow']:,.0f}")
                with ts_cols[3]:
                    st.metric("25-Year IRR", f"{irr_25yr:.1%}")
                with ts_cols[4]:
                    st.metric("Year 25 Rent Burden", f"{year_25['rent_burden']:.1%}")

                # Time series charts
                ts_chart = create_time_series_chart(time_series)
                st.plotly_chart(ts_chart, use_container_width=True)

                # Cash flow chart
                cf_chart = create_cash_flow_chart(time_series)
                st.plotly_chart(cf_chart, use_container_width=True)

                # Data table
                with st.expander("View Yearly Data"):
                    ts_df = pd.DataFrame(time_series)
                    ts_df['rent_psf'] = ts_df['rent_psf'].apply(lambda x: f"${x:.2f}")
                    ts_df['monthly_rent'] = ts_df['monthly_rent'].apply(lambda x: f"${x:,.0f}")
                    ts_df['annual_revenue'] = ts_df['annual_revenue'].apply(lambda x: f"${x:,.0f}")
                    ts_df['noi'] = ts_df['noi'].apply(lambda x: f"${x:,.0f}")
                    ts_df['debt_service'] = ts_df['debt_service'].apply(lambda x: f"${x:,.0f}")
                    ts_df['cash_flow'] = ts_df['cash_flow'].apply(lambda x: f"${x:,.0f}")
                    ts_df['cumulative_cash_flow'] = ts_df['cumulative_cash_flow'].apply(lambda x: f"${x:,.0f}")
                    ts_df['dscr'] = ts_df['dscr'].apply(lambda x: f"{x:.2f}")
                    ts_df['cash_on_cash'] = ts_df['cash_on_cash'].apply(lambda x: f"{x:.1%}")
                    ts_df['cumulative_return'] = ts_df['cumulative_return'].apply(lambda x: f"{x:.1%}")
                    ts_df['rent_burden'] = ts_df['rent_burden'].apply(lambda x: f"{x:.1%}")
                    ts_df['median_income'] = ts_df['median_income'].apply(lambda x: f"${x:,.0f}")

                    st.dataframe(
                        ts_df[['year', 'rent_psf', 'noi', 'debt_service', 'cash_flow',
                               'dscr', 'cumulative_return', 'rent_burden', 'is_viable']],
                        use_container_width=True,
                        height=300
                    )
