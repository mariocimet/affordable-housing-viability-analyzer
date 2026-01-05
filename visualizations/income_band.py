"""
Income Band Visualization

Creates interactive bell curve showing household income distribution
with shaded affordability band that can animate over time.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List

from models.income_distribution import BCIncomeDistribution, calculate_band_over_time


def create_income_band_chart(
    distribution: BCIncomeDistribution,
    monthly_rent: float,
    affordability_ratio_lower: float = 0.25,
    affordability_ratio_upper: float = 0.30,
    escalation_rates: List[float] = None,
    show_animation: bool = True,
    income_growth_rate: float = 0.02
) -> go.Figure:
    """
    Create interactive income distribution chart with affordability band.

    Args:
        distribution: BC income distribution model
        monthly_rent: Initial monthly rent
        affordability_ratio_lower: Min rent as fraction of income (program floor)
        affordability_ratio_upper: Max rent as fraction of income (affordability cap)
        escalation_rates: Optional list of annual rent increases for animation
        show_animation: Whether to include time animation
        income_growth_rate: Annual income growth rate (inflation)

    Returns:
        Plotly Figure object
    """
    # Get curve data
    incomes, densities = distribution.get_curve_data()

    # Calculate initial band
    min_income, max_income, lower_pct, upper_pct = distribution.affordability_band(
        monthly_rent, affordability_ratio_lower, affordability_ratio_upper
    )

    if show_animation and escalation_rates:
        return _create_animated_chart(
            distribution, incomes, densities, monthly_rent,
            affordability_ratio_lower, affordability_ratio_upper, escalation_rates,
            income_growth_rate
        )
    else:
        return _create_static_chart(
            distribution, incomes, densities,
            min_income, max_income, lower_pct, upper_pct,
            monthly_rent, affordability_ratio_lower, affordability_ratio_upper
        )


def _create_static_chart(
    distribution: BCIncomeDistribution,
    incomes: np.ndarray,
    densities: np.ndarray,
    min_income: float,
    max_income: float,
    lower_pct: float,
    upper_pct: float,
    monthly_rent: float,
    affordability_ratio_lower: float,
    affordability_ratio_upper: float
) -> go.Figure:
    """Create static income band chart."""

    fig = go.Figure()

    # Main distribution curve
    fig.add_trace(go.Scatter(
        x=incomes,
        y=densities,
        mode='lines',
        name='Income Distribution',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='Income: $%{x:,.0f}<br>Density: %{y:.6f}<extra></extra>'
    ))

    # Affordability band (shaded area)
    band_mask = (incomes >= min_income) & (incomes <= max_income)
    band_incomes = incomes[band_mask]
    band_densities = densities[band_mask]

    if len(band_incomes) > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([band_incomes, band_incomes[::-1]]),
            y=np.concatenate([band_densities, np.zeros_like(band_densities)]),
            fill='toself',
            fillcolor='rgba(46, 204, 113, 0.4)',
            line=dict(color='rgba(46, 204, 113, 0)'),
            name=f'Target Band ({lower_pct:.1f}th - {upper_pct:.1f}th %ile)',
            hoverinfo='skip'
        ))

    # Vertical lines for band boundaries
    max_density = max(densities)

    # Lower boundary line (min income needed - from upper affordability ratio)
    fig.add_trace(go.Scatter(
        x=[min_income, min_income],
        y=[0, distribution.pdf(min_income)],
        mode='lines',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        name=f'Min Income ({affordability_ratio_upper:.0%} burden): ${min_income:,.0f}',
        hovertemplate=f'Min income (rent = {affordability_ratio_upper:.0%} of income)<br>${min_income:,.0f}<br>({lower_pct:.1f}th percentile)<extra></extra>'
    ))

    # Upper boundary line (max income - from lower affordability ratio)
    if affordability_ratio_lower > 0:
        fig.add_trace(go.Scatter(
            x=[max_income, max_income],
            y=[0, distribution.pdf(max_income)],
            mode='lines',
            line=dict(color='#9B59B6', width=2, dash='dash'),
            name=f'Max Income ({affordability_ratio_lower:.0%} burden): ${max_income:,.0f}',
            hovertemplate=f'Max income (rent = {affordability_ratio_lower:.0%} of income)<br>${max_income:,.0f}<br>({upper_pct:.1f}th percentile)<extra></extra>'
        ))

    # Add percentile annotations
    fig.add_annotation(
        x=min_income, y=distribution.pdf(min_income) * 1.1,
        text=f'{lower_pct:.1f}th %ile',
        showarrow=False,
        font=dict(color='#E74C3C', size=10)
    )

    if affordability_ratio_lower > 0:
        fig.add_annotation(
            x=max_income, y=distribution.pdf(max_income) * 1.1,
            text=f'{upper_pct:.1f}th %ile',
            showarrow=False,
            font=dict(color='#9B59B6', size=10)
        )

    band_width = max(0, upper_pct - lower_pct)

    fig.update_layout(
        title=dict(
            text=f'BC Household Income Distribution<br><sub>Rent: ${monthly_rent:,.0f}/mo | '
                 f'Target Band: {band_width:.1f}% of households ({lower_pct:.0f}th-{upper_pct:.0f}th %ile)</sub>',
            x=0.5
        ),
        xaxis_title='Annual Household Income (CAD)',
        yaxis_title='Probability Density',
        xaxis=dict(
            tickformat='$,.0f',
            range=[0, distribution.income_at_percentile(99)]
        ),
        yaxis=dict(showticklabels=False),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99
        ),
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def _create_animated_chart(
    distribution: BCIncomeDistribution,
    incomes: np.ndarray,
    densities: np.ndarray,
    initial_rent: float,
    affordability_ratio_lower: float,
    affordability_ratio_upper: float,
    escalation_rates: List[float],
    income_growth_rate: float = 0.02
) -> go.Figure:
    """Create animated chart showing band evolution over time with income growth."""

    bands = calculate_band_over_time(
        distribution, initial_rent, escalation_rates,
        affordability_ratio_lower, affordability_ratio_upper, income_growth_rate
    )

    # Create frames for animation
    frames = []

    # Get max income for consistent x-axis across all frames
    final_median = bands[-1]['median_income']
    final_dist = BCIncomeDistribution(median=final_median, sigma=distribution.sigma)
    max_x = final_dist.income_at_percentile(99)

    for band_data in bands:
        year = band_data['year']
        rent = band_data['rent']
        median_income = band_data['median_income']
        min_income = band_data['min_income']
        max_income = band_data['max_income']
        lower_pct = band_data['lower_percentile']
        upper_pct = band_data['upper_percentile']
        band_width = band_data['band_width']

        # Create distribution for this year
        year_dist = BCIncomeDistribution(median=median_income, sigma=distribution.sigma)
        year_incomes, year_densities = year_dist.get_curve_data(num_points=500, max_income=max_x)

        # Create band mask
        band_mask = (year_incomes >= min_income) & (year_incomes <= max_income)
        band_incomes = year_incomes[band_mask]
        band_densities = year_densities[band_mask]

        frame_data = [
            # Distribution curve (shifts right each year)
            go.Scatter(
                x=year_incomes,
                y=year_densities,
                mode='lines',
                line=dict(color='#2E86AB', width=2),
            ),
        ]

        # Affordability band
        if len(band_incomes) > 0:
            frame_data.append(go.Scatter(
                x=np.concatenate([band_incomes, band_incomes[::-1]]),
                y=np.concatenate([band_densities, np.zeros_like(band_densities)]),
                fill='toself',
                fillcolor='rgba(46, 204, 113, 0.4)',
                line=dict(color='rgba(46, 204, 113, 0)'),
            ))

        # Min income line
        frame_data.append(go.Scatter(
            x=[min_income, min_income],
            y=[0, year_dist.pdf(min_income)],
            mode='lines',
            line=dict(color='#E74C3C', width=2, dash='dash'),
        ))

        # Max income line
        frame_data.append(go.Scatter(
            x=[max_income, max_income],
            y=[0, year_dist.pdf(max_income)],
            mode='lines',
            line=dict(color='#9B59B6', width=2, dash='dash'),
        ))

        frames.append(go.Frame(
            data=frame_data,
            name=str(year),
            layout=go.Layout(
                title=dict(
                    text=f'Household Income Distribution - Year {year}<br>'
                         f'<sub>Rent: ${rent:,.0f}/mo | Median Income: ${median_income:,.0f} | '
                         f'Band: {band_width:.1f}% of households</sub>'
                )
            )
        ))

    # Initial frame data
    initial = bands[0]
    initial_dist = BCIncomeDistribution(median=initial['median_income'], sigma=distribution.sigma)
    init_incomes, init_densities = initial_dist.get_curve_data(num_points=500, max_income=max_x)

    band_mask = (init_incomes >= initial['min_income']) & (init_incomes <= initial['max_income'])
    band_incomes = init_incomes[band_mask]
    band_densities = init_densities[band_mask]

    fig = go.Figure(
        data=[
            go.Scatter(x=init_incomes, y=init_densities, mode='lines',
                      line=dict(color='#2E86AB', width=2), name='Income Distribution'),
            go.Scatter(
                x=np.concatenate([band_incomes, band_incomes[::-1]]) if len(band_incomes) > 0 else [],
                y=np.concatenate([band_densities, np.zeros_like(band_densities)]) if len(band_incomes) > 0 else [],
                fill='toself', fillcolor='rgba(46, 204, 113, 0.4)',
                line=dict(color='rgba(46, 204, 113, 0)'), name='Affordability Band'
            ),
            go.Scatter(x=[initial['min_income']]*2, y=[0, initial_dist.pdf(initial['min_income'])],
                      mode='lines', line=dict(color='#E74C3C', width=2, dash='dash'), name='Min Income'),
            go.Scatter(x=[initial['max_income']]*2, y=[0, initial_dist.pdf(initial['max_income'])],
                      mode='lines', line=dict(color='#9B59B6', width=2, dash='dash'), name='75th %ile Cap'),
        ],
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        title=dict(
            text=f'Household Income Distribution - Year 0<br>'
                 f'<sub>Rent: ${initial["rent"]:,.0f}/mo | Median Income: ${initial["median_income"]:,.0f} | '
                 f'Band: {initial["band_width"]:.1f}% of households</sub>',
            x=0.5
        ),
        xaxis_title='Annual Household Income (CAD)',
        yaxis_title='Probability Density',
        xaxis=dict(tickformat='$,.0f', range=[0, max_x]),
        yaxis=dict(showticklabels=False),
        template='plotly_white',
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=500, redraw=True),
                                         fromcurrent=True,
                                         transition=dict(duration=300))]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate',
                                           transition=dict(duration=0))])
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(
                    font=dict(size=12),
                    prefix='Year: ',
                    visible=True,
                    xanchor='center'
                ),
                transition=dict(duration=300),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.05,
                y=0,
                steps=[
                    dict(args=[[str(i)],
                              dict(frame=dict(duration=300, redraw=True),
                                  mode='immediate',
                                  transition=dict(duration=300))],
                         label=str(i),
                         method='animate')
                    for i in range(len(bands))
                ]
            )
        ]
    )

    return fig


def create_band_summary_chart(bands: List[dict]) -> go.Figure:
    """
    Create summary chart showing band width evolution over time.

    Args:
        bands: List of band data from calculate_band_over_time

    Returns:
        Plotly Figure
    """
    years = [b['year'] for b in bands]
    band_widths = [b['band_width'] for b in bands]
    rents = [b['rent'] for b in bands]
    min_incomes = [b['min_income'] for b in bands]
    median_incomes = [b.get('median_income', bands[0].get('median_income', 90000)) for b in bands]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Affordability Band Width Over Time', 'Rent, Income & Required Income'),
        vertical_spacing=0.15
    )

    # Band width over time
    fig.add_trace(
        go.Scatter(
            x=years, y=band_widths,
            mode='lines+markers',
            name='% of Households Served',
            line=dict(color='#2ECC71', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

    # Rent and income over time
    fig.add_trace(
        go.Scatter(
            x=years, y=rents,
            mode='lines+markers',
            name='Monthly Rent',
            line=dict(color='#3498DB', width=2),
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=years, y=min_incomes,
            mode='lines+markers',
            name='Min Income Required',
            line=dict(color='#E74C3C', width=2),
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=years, y=median_incomes,
            mode='lines+markers',
            name='Median Income',
            line=dict(color='#9B59B6', width=2, dash='dash'),
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        template='plotly_white',
        showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99)
    )

    fig.update_xaxes(title_text='Year', row=1, col=1)
    fig.update_xaxes(title_text='Year', row=2, col=1)
    fig.update_yaxes(title_text='% of Households', row=1, col=1, ticksuffix='%')
    fig.update_yaxes(title_text='CAD ($)', tickformat='$,.0f', row=2, col=1)

    return fig
