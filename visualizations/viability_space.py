"""
Project Viability Space Visualization

Creates 2D heatmap/contour plot showing viable project parameter combinations.
User selects 2 parameters to vary, fixes the rest.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List, Optional
import pandas as pd

from models.project_viability import (
    calculate_viability_grid,
    PARAM_INFO,
    ProjectParameters,
    ProjectViabilityModel
)


def create_viability_space_chart(
    x_param: str,
    y_param: str,
    fixed_params: dict,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    resolution: int = 50,
    projects: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Create 2D viability space visualization.

    Args:
        x_param: Parameter name for X axis
        y_param: Parameter name for Y axis
        fixed_params: Dictionary of fixed parameter values
        x_range: Optional (min, max) for X axis
        y_range: Optional (min, max) for Y axis
        resolution: Grid resolution
        projects: Optional DataFrame with project data to overlay

    Returns:
        Plotly Figure object
    """
    # Get default ranges if not specified
    if x_range is None:
        x_range = PARAM_INFO[x_param]['default_range']
    if y_range is None:
        y_range = PARAM_INFO[y_param]['default_range']

    # Calculate viability grid
    X, Y, viability, margin = calculate_viability_grid(
        x_param, y_param, x_range, y_range, fixed_params, resolution
    )

    # Get display names
    x_name = PARAM_INFO[x_param]['name']
    y_name = PARAM_INFO[y_param]['name']

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Viability Score', 'Viability Margin'),
        horizontal_spacing=0.12
    )

    # Viability score heatmap (0, 1, or 2 criteria met)
    fig.add_trace(
        go.Heatmap(
            x=X[0],
            y=Y[:, 0],
            z=viability,
            colorscale=[
                [0, '#E74C3C'],      # Red: 0 criteria (not viable)
                [0.5, '#F39C12'],    # Yellow: 1 criterion
                [1, '#2ECC71']       # Green: 2 criteria (viable)
            ],
            zmin=0,
            zmax=2,
            colorbar=dict(
                title='Criteria Met',
                tickvals=[0, 1, 2],
                ticktext=['None', 'One', 'Both'],
                x=0.45
            ),
            hovertemplate=(
                f'{x_name}: %{{x:.2f}}<br>'
                f'{y_name}: %{{y:.2f}}<br>'
                'Criteria Met: %{z}<extra></extra>'
            )
        ),
        row=1, col=1
    )

    # Margin heatmap (continuous scale)
    fig.add_trace(
        go.Heatmap(
            x=X[0],
            y=Y[:, 0],
            z=margin,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(
                title='Margin',
                x=1.02
            ),
            hovertemplate=(
                f'{x_name}: %{{x:.2f}}<br>'
                f'{y_name}: %{{y:.2f}}<br>'
                'Margin: %{z:.2%}<extra></extra>'
            )
        ),
        row=1, col=2
    )

    # Add viability boundary contour
    fig.add_trace(
        go.Contour(
            x=X[0],
            y=Y[:, 0],
            z=viability,
            contours=dict(
                start=1.5,
                end=1.5,
                size=1,
                coloring='none'
            ),
            line=dict(color='black', width=3),
            showscale=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Contour(
            x=X[0],
            y=Y[:, 0],
            z=margin,
            contours=dict(
                start=0,
                end=0,
                size=0.01,
                coloring='none'
            ),
            line=dict(color='black', width=3),
            showscale=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )

    # Overlay projects if provided
    if projects is not None and len(projects) > 0:
        for _, row in projects.iterrows():
            x_val = row.get(x_param)
            y_val = row.get(y_param)
            name = row.get('name', 'Project')

            if x_val is not None and y_val is not None:
                for col in [1, 2]:
                    fig.add_trace(
                        go.Scatter(
                            x=[x_val],
                            y=[y_val],
                            mode='markers+text',
                            marker=dict(
                                size=15,
                                color='white',
                                line=dict(color='black', width=2),
                                symbol='star'
                            ),
                            text=[name],
                            textposition='top center',
                            name=name,
                            showlegend=(col == 1),
                            hovertemplate=(
                                f'{name}<br>'
                                f'{x_name}: %{{x:.2f}}<br>'
                                f'{y_name}: %{{y:.2f}}<extra></extra>'
                            )
                        ),
                        row=1, col=col
                    )

    # Format axes based on parameter types
    x_format = PARAM_INFO[x_param]['format']
    y_format = PARAM_INFO[y_param]['format']

    x_tickformat = '$,.0f' if '$' in x_format else '.1%' if '%' in x_format else '.2f'
    y_tickformat = '$,.0f' if '$' in y_format else '.1%' if '%' in y_format else '.2f'

    fig.update_xaxes(title_text=x_name, tickformat=x_tickformat, row=1, col=1)
    fig.update_xaxes(title_text=x_name, tickformat=x_tickformat, row=1, col=2)
    fig.update_yaxes(title_text=y_name, tickformat=y_tickformat, row=1, col=1)
    fig.update_yaxes(title_text=y_name, tickformat=y_tickformat, row=1, col=2)

    # Build fixed params description
    fixed_desc = []
    for param, value in fixed_params.items():
        if param not in [x_param, y_param] and param in PARAM_INFO:
            fmt = PARAM_INFO[param]['format']
            fixed_desc.append(f"{PARAM_INFO[param]['name']}: {fmt.format(value)}")

    fig.update_layout(
        title=dict(
            text=f'Project Viability Space<br><sub>Fixed: {" | ".join(fixed_desc[:3])}</sub>',
            x=0.5
        ),
        height=500,
        template='plotly_white',
        showlegend=True,
        legend=dict(yanchor='top', y=1.15, xanchor='left', x=0)
    )

    return fig


def create_single_viability_chart(
    x_param: str,
    y_param: str,
    fixed_params: dict,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    resolution: int = 75,
    projects: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Create single panel viability chart (cleaner for main display).
    """
    if x_range is None:
        x_range = PARAM_INFO[x_param]['default_range']
    if y_range is None:
        y_range = PARAM_INFO[y_param]['default_range']

    X, Y, viability, margin = calculate_viability_grid(
        x_param, y_param, x_range, y_range, fixed_params, resolution
    )

    x_name = PARAM_INFO[x_param]['name']
    y_name = PARAM_INFO[y_param]['name']

    fig = go.Figure()

    # Main heatmap using margin for color intensity
    fig.add_trace(
        go.Heatmap(
            x=X[0],
            y=Y[:, 0],
            z=margin,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(
                title='Viability<br>Margin',
                tickformat='.0%'
            ),
            hovertemplate=(
                f'{x_name}: %{{x:.2f}}<br>'
                f'{y_name}: %{{y:.2f}}<br>'
                'Margin: %{z:.1%}<extra></extra>'
            )
        )
    )

    # Add viability boundary
    fig.add_trace(
        go.Contour(
            x=X[0],
            y=Y[:, 0],
            z=margin,
            contours=dict(
                start=0,
                end=0,
                size=0.01,
                coloring='none'
            ),
            line=dict(color='black', width=3, dash='solid'),
            showscale=False,
            name='Viability Boundary',
            hoverinfo='skip'
        )
    )

    # Overlay projects
    if projects is not None and len(projects) > 0:
        for _, row in projects.iterrows():
            x_val = row.get(x_param)
            y_val = row.get(y_param)
            name = row.get('name', 'Project')

            if x_val is not None and y_val is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode='markers+text',
                        marker=dict(
                            size=18,
                            color='white',
                            line=dict(color='black', width=2),
                            symbol='star'
                        ),
                        text=[name],
                        textposition='top center',
                        textfont=dict(size=12, color='black'),
                        name=name,
                        hovertemplate=(
                            f'<b>{name}</b><br>'
                            f'{x_name}: %{{x:.2f}}<br>'
                            f'{y_name}: %{{y:.2f}}<extra></extra>'
                        )
                    )
                )

    x_format = PARAM_INFO[x_param]['format']
    y_format = PARAM_INFO[y_param]['format']
    x_tickformat = '$,.0f' if '$' in x_format else '.1%' if '%' in x_format else '.2f'
    y_tickformat = '$,.0f' if '$' in y_format else '.1%' if '%' in y_format else '.2f'

    # Fixed params for subtitle
    fixed_desc = []
    for param, value in fixed_params.items():
        if param not in [x_param, y_param] and param in PARAM_INFO:
            fmt = PARAM_INFO[param]['format']
            fixed_desc.append(f"{PARAM_INFO[param]['name']}: {fmt.format(value)}")

    fig.update_layout(
        title=dict(
            text=f'Project Viability Space<br><sub>Green = viable | Black line = viability boundary</sub>',
            x=0.5
        ),
        xaxis_title=x_name,
        yaxis_title=y_name,
        xaxis=dict(tickformat=x_tickformat),
        yaxis=dict(tickformat=y_tickformat),
        height=600,
        template='plotly_white'
    )

    # Add annotation for fixed params
    if fixed_desc:
        fig.add_annotation(
            text='<br>'.join(fixed_desc),
            xref='paper', yref='paper',
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            align='left',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )

    return fig


def evaluate_project(params: dict) -> dict:
    """
    Evaluate a single project's viability.

    Args:
        params: Dictionary of project parameters

    Returns:
        Dictionary with viability metrics
    """
    project_params = ProjectParameters(**params)
    model = ProjectViabilityModel(project_params)
    is_viable, metrics = model.is_viable()

    return {
        'is_viable': is_viable,
        'dscr': metrics['dscr'],
        'cash_on_cash': metrics['cash_on_cash'],
        'noi': metrics['noi'],
        'debt_service': metrics['annual_debt_service'],
        'cash_flow': metrics['cash_flow'],
        'positive_cash_flow': metrics['positive_cash_flow'],
        'meets_equity_return': metrics['meets_equity_return']
    }


def create_time_series_chart(time_series_data: List[dict]) -> go.Figure:
    """
    Create time-series visualization showing project financials over time.

    Shows how cash flow, DSCR, and cumulative returns evolve as housing
    charges increase while debt service stays constant.

    Args:
        time_series_data: List of dicts from calculate_metrics_over_time()

    Returns:
        Plotly Figure with multiple subplots
    """
    years = [d['year'] for d in time_series_data]
    cash_flow = [d['cash_flow'] for d in time_series_data]
    noi = [d['noi'] for d in time_series_data]
    debt_service = [d['debt_service'] for d in time_series_data]
    dscr = [d['dscr'] for d in time_series_data]
    cumulative_return = [d['cumulative_return'] for d in time_series_data]
    rent_burden = [d['rent_burden'] for d in time_series_data]
    is_viable = [d['is_viable'] for d in time_series_data]

    # Find first viable year
    first_viable_year = None
    for d in time_series_data:
        if d['is_viable']:
            first_viable_year = d['year']
            break

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'NOI vs Debt Service',
            'Debt Service Coverage Ratio (DSCR)',
            'Cumulative Return on Equity',
            'Rent Burden at Target Percentile'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Panel 1: NOI vs Debt Service
    fig.add_trace(
        go.Scatter(
            x=years, y=noi,
            mode='lines',
            name='NOI',
            line=dict(color='#2ECC71', width=2),
            hovertemplate='Year %{x}<br>NOI: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=years, y=debt_service,
            mode='lines',
            name='Debt Service',
            line=dict(color='#E74C3C', width=2, dash='dash'),
            hovertemplate='Year %{x}<br>Debt Service: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )
    # Fill between NOI and debt service
    fig.add_trace(
        go.Scatter(
            x=years + years[::-1],
            y=noi + debt_service[::-1],
            fill='toself',
            fillcolor='rgba(46, 204, 113, 0.2)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Panel 2: DSCR over time
    fig.add_trace(
        go.Scatter(
            x=years, y=dscr,
            mode='lines+markers',
            name='DSCR',
            line=dict(color='#3498DB', width=2),
            marker=dict(size=4),
            hovertemplate='Year %{x}<br>DSCR: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    # Add threshold line at 1.0
    fig.add_hline(
        y=1.0, line_dash='dash', line_color='red',
        annotation_text='Min DSCR = 1.0',
        annotation_position='bottom right',
        row=1, col=2
    )

    # Panel 3: Cumulative return
    fig.add_trace(
        go.Scatter(
            x=years, y=[r * 100 for r in cumulative_return],
            mode='lines+markers',
            name='Cumulative Return',
            line=dict(color='#9B59B6', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)',
            hovertemplate='Year %{x}<br>Cumulative Return: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    fig.add_hline(
        y=0, line_dash='dash', line_color='gray',
        row=2, col=1
    )

    # Panel 4: Rent burden
    fig.add_trace(
        go.Scatter(
            x=years, y=[r * 100 for r in rent_burden],
            mode='lines+markers',
            name='Rent Burden',
            line=dict(color='#F39C12', width=2),
            marker=dict(size=4),
            hovertemplate='Year %{x}<br>Rent Burden: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    # Add affordability threshold (30%)
    fig.add_hline(
        y=30, line_dash='dash', line_color='red',
        annotation_text='Max 30%',
        annotation_position='bottom right',
        row=2, col=2
    )

    # Add vertical line at first viable year if exists
    if first_viable_year is not None and first_viable_year > 0:
        for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            fig.add_vline(
                x=first_viable_year, line_dash='dot', line_color='green',
                row=row, col=col
            )

    # Update axes labels
    fig.update_xaxes(title_text='Year', row=2, col=1)
    fig.update_xaxes(title_text='Year', row=2, col=2)
    fig.update_yaxes(title_text='$ / Year', tickformat='$,.0f', row=1, col=1)
    fig.update_yaxes(title_text='DSCR', row=1, col=2)
    fig.update_yaxes(title_text='Cumulative Return (%)', row=2, col=1)
    fig.update_yaxes(title_text='Rent Burden (%)', row=2, col=2)

    # Title with viability summary
    if first_viable_year is not None:
        if first_viable_year == 0:
            title_text = 'Project Financial Projection (25 Years)<br><sub style="color:green">✓ Viable from Year 0</sub>'
        else:
            title_text = f'Project Financial Projection (25 Years)<br><sub style="color:orange">Becomes viable in Year {first_viable_year}</sub>'
    else:
        title_text = 'Project Financial Projection (25 Years)<br><sub style="color:red">✗ Not viable within 25 years</sub>'

    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        height=600,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    return fig


def create_cash_flow_chart(time_series_data: List[dict]) -> go.Figure:
    """
    Create a focused cash flow chart showing annual and cumulative cash flow.

    Args:
        time_series_data: List of dicts from calculate_metrics_over_time()

    Returns:
        Plotly Figure with bar chart for annual and line for cumulative
    """
    years = [d['year'] for d in time_series_data]
    cash_flow = [d['cash_flow'] for d in time_series_data]
    cumulative = [d['cumulative_cash_flow'] for d in time_series_data]
    is_viable = [d['is_viable'] for d in time_series_data]

    # Color bars by viability
    colors = ['#2ECC71' if v else '#E74C3C' for v in is_viable]

    fig = go.Figure()

    # Annual cash flow bars
    fig.add_trace(
        go.Bar(
            x=years,
            y=cash_flow,
            name='Annual Cash Flow',
            marker_color=colors,
            hovertemplate='Year %{x}<br>Cash Flow: $%{y:,.0f}<extra></extra>'
        )
    )

    # Cumulative cash flow line
    fig.add_trace(
        go.Scatter(
            x=years,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative Cash Flow',
            line=dict(color='#3498DB', width=3),
            marker=dict(size=6),
            yaxis='y2',
            hovertemplate='Year %{x}<br>Cumulative: $%{y:,.0f}<extra></extra>'
        )
    )

    # Find breakeven year
    breakeven_year = None
    for i, c in enumerate(cumulative):
        if c >= 0:
            breakeven_year = years[i]
            break

    fig.update_layout(
        title=dict(
            text='Cash Flow Over Time<br><sub>Green = viable year, Red = not viable</sub>',
            x=0.5
        ),
        xaxis=dict(title='Year', dtick=5),
        yaxis=dict(
            title='Annual Cash Flow ($)',
            tickformat='$,.0f',
            side='left'
        ),
        yaxis2=dict(
            title='Cumulative Cash Flow ($)',
            tickformat='$,.0f',
            overlaying='y',
            side='right'
        ),
        height=400,
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        hovermode='x unified'
    )

    # Add breakeven annotation
    if breakeven_year is not None:
        fig.add_vline(
            x=breakeven_year, line_dash='dot', line_color='green',
            annotation_text=f'Breakeven Year {breakeven_year}',
            annotation_position='top'
        )

    return fig


def create_portfolio_scatter(
    portfolio_df: pd.DataFrame,
    x_param: str = 'acquisition_cost_psf',
    y_param: str = 'rent_psf',
    fixed_params: dict = None,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    resolution: int = 50,
    highlight_listing_id: str = None
) -> go.Figure:
    """
    Create viability space with all portfolio listings overlaid.

    Args:
        portfolio_df: DataFrame from analyze_portfolio() with viability metrics
        x_param, y_param: Axes parameters
        fixed_params: Fixed parameters for viability grid
        x_range, y_range: Axis ranges
        resolution: Grid resolution
        highlight_listing_id: Optional ID to highlight with gold star

    Returns:
        Plotly Figure with heatmap background and scatter overlay
    """
    if fixed_params is None:
        fixed_params = {}

    # Get default ranges if not specified
    if x_range is None:
        x_range = PARAM_INFO[x_param]['default_range']
    if y_range is None:
        y_range = PARAM_INFO[y_param]['default_range']

    # Calculate viability grid
    X, Y, viability, margin = calculate_viability_grid(
        x_param, y_param, x_range, y_range, fixed_params, resolution
    )

    x_name = PARAM_INFO[x_param]['name']
    y_name = PARAM_INFO[y_param]['name']

    fig = go.Figure()

    # Background heatmap
    fig.add_trace(
        go.Heatmap(
            x=X[0],
            y=Y[:, 0],
            z=margin,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title='Viability<br>Margin', tickformat='.0%'),
            opacity=0.6,
            hoverinfo='skip'
        )
    )

    # Viability boundary
    fig.add_trace(
        go.Contour(
            x=X[0],
            y=Y[:, 0],
            z=margin,
            contours=dict(start=0, end=0, size=0.01, coloring='none'),
            line=dict(color='black', width=3, dash='solid'),
            showscale=False,
            name='Viability Boundary',
            hoverinfo='skip'
        )
    )

    # Map param names for portfolio_df (min_rent_psf in data = rent_psf in model)
    data_param_mapping = {'rent_psf': 'min_rent_psf'}
    x_data_col = data_param_mapping.get(x_param, x_param)
    y_data_col = data_param_mapping.get(y_param, y_param)

    # Add fixed params to portfolio_df if not present (for plotting)
    for param, data_col in [(x_param, x_data_col), (y_param, y_data_col)]:
        if data_col not in portfolio_df.columns and param in fixed_params:
            portfolio_df[data_col] = fixed_params[param]

    # Filter to plottable listings
    plottable = portfolio_df[
        portfolio_df[x_data_col].notna() &
        portfolio_df[y_data_col].notna()
    ].copy()

    # Rename columns for consistent access
    if x_data_col != x_param:
        plottable[x_param] = plottable[x_data_col]
    if y_data_col != y_param:
        plottable[y_param] = plottable[y_data_col]

    # Prepare hover data columns (with fallbacks)
    hover_cols = []
    for col in ['annualized_return', 'payback_year']:
        if col in plottable.columns:
            hover_cols.append(col)
        else:
            plottable[col] = None
            hover_cols.append(col)

    # Viable listings (green circles)
    viable = plottable[plottable['is_viable'] == True]
    if len(viable) > 0:
        fig.add_trace(
            go.Scatter(
                x=viable[x_param],
                y=viable[y_param],
                mode='markers',
                marker=dict(
                    size=12,
                    color='#2ECC71',
                    line=dict(color='white', width=2),
                    symbol='circle'
                ),
                name=f'Viable ({len(viable)})',
                text=viable['name'],
                customdata=viable[hover_cols].values,
                hovertemplate='<b>%{text}</b><br>' +
                              f'{x_name}: %{{x:.2f}}<br>' +
                              f'{y_name}: %{{y:.1%}}<br>' +
                              'Return: %{customdata[0]:.1%}<br>' +
                              'Payback: Yr %{customdata[1]:.0f}<extra></extra>'
            )
        )

    # Not viable listings (red circles)
    not_viable = plottable[plottable['is_viable'] == False]
    if len(not_viable) > 0:
        fig.add_trace(
            go.Scatter(
                x=not_viable[x_param],
                y=not_viable[y_param],
                mode='markers',
                marker=dict(
                    size=12,
                    color='#E74C3C',
                    line=dict(color='white', width=2),
                    symbol='circle'
                ),
                name=f'Not Viable ({len(not_viable)})',
                text=not_viable['name'],
                customdata=not_viable[hover_cols].values,
                hovertemplate='<b>%{text}</b><br>' +
                              f'{x_name}: %{{x:.2f}}<br>' +
                              f'{y_name}: %{{y:.1%}}<br>' +
                              'Return: %{customdata[0]:.1%}<br>' +
                              'Payback: N/A<extra></extra>'
            )
        )

    # Highlight specific listing if requested
    if highlight_listing_id:
        highlight = plottable[plottable['id'] == highlight_listing_id]
        if len(highlight) > 0:
            fig.add_trace(
                go.Scatter(
                    x=highlight[x_param],
                    y=highlight[y_param],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='gold',
                        line=dict(color='black', width=3),
                        symbol='star'
                    ),
                    name='Selected',
                    text=highlight['name'],
                    hovertemplate='<b>%{text}</b> (Selected)<extra></extra>'
                )
            )

    # Layout
    x_format = PARAM_INFO[x_param]['format']
    y_format = PARAM_INFO[y_param]['format']
    x_tickformat = '$,.0f' if '$' in x_format else '.1%' if '%' in x_format else '.2f'
    y_tickformat = '$,.0f' if '$' in y_format else '.1%' if '%' in y_format else '.2f'

    fig.update_layout(
        title='Portfolio Viability Analysis',
        xaxis_title=x_name,
        yaxis_title=y_name,
        xaxis=dict(tickformat=x_tickformat),
        yaxis=dict(tickformat=y_tickformat),
        height=500,
        template='plotly_white',
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99)
    )

    return fig


# ============================================
# NEW VISUALIZATION OPTIONS
# ============================================

def create_cost_vs_burden_chart(
    portfolio_df: pd.DataFrame,
    fixed_params: dict,
) -> go.Figure:
    """
    1. Acquisition Cost vs Rent Burden scatter plot.

    Shows how rent burden increases with acquisition cost - the fundamental
    relationship driving affordability. Each listing is a point, colored by
    whether it's affordable.
    """
    if portfolio_df.empty:
        return go.Figure().update_layout(title="No data available")

    df = portfolio_df.copy()
    affordability_ratio = fixed_params.get('affordability_ratio', 0.30)

    fig = go.Figure()

    # Affordable listings
    affordable = df[df['is_viable'] == True]
    if len(affordable) > 0:
        fig.add_trace(go.Scatter(
            x=affordable['acquisition_cost_psf'],
            y=affordable['rent_burden'],
            mode='markers',
            marker=dict(size=12, color='#2ECC71', line=dict(color='white', width=2)),
            name=f'Affordable ({len(affordable)})',
            text=affordable['name'],
            hovertemplate='<b>%{text}</b><br>Cost: $%{x:.0f}/sqft<br>Burden: %{y:.1%}<extra></extra>'
        ))

    # Not affordable listings
    not_affordable = df[df['is_viable'] == False]
    if len(not_affordable) > 0:
        fig.add_trace(go.Scatter(
            x=not_affordable['acquisition_cost_psf'],
            y=not_affordable['rent_burden'],
            mode='markers',
            marker=dict(size=12, color='#E74C3C', line=dict(color='white', width=2)),
            name=f'Not Affordable ({len(not_affordable)})',
            text=not_affordable['name'],
            hovertemplate='<b>%{text}</b><br>Cost: $%{x:.0f}/sqft<br>Burden: %{y:.1%}<extra></extra>'
        ))

    # Affordability threshold line
    fig.add_hline(
        y=affordability_ratio,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Max Burden ({affordability_ratio:.0%})',
        annotation_position='top right'
    )

    # Add trendline
    if len(df) > 2:
        x_vals = df['acquisition_cost_psf'].dropna()
        y_vals = df['rent_burden'].dropna()
        if len(x_vals) == len(y_vals) and len(x_vals) > 2:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line, y=p(x_line),
                mode='lines',
                line=dict(color='gray', dash='dot'),
                name='Trend',
                hoverinfo='skip'
            ))

    fig.update_layout(
        title='Acquisition Cost vs Rent Burden<br><sub>Higher cost → Higher rent → Higher burden</sub>',
        xaxis_title='Acquisition Cost ($/sqft)',
        yaxis_title='Rent Burden (% of Income)',
        xaxis=dict(tickformat='$,.0f'),
        yaxis=dict(tickformat='.0%'),
        height=450,
        template='plotly_white',
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99)
    )

    return fig


def create_max_viable_cost_chart(
    portfolio_df: pd.DataFrame,
    fixed_params: dict,
) -> go.Figure:
    """
    2. Maximum Viable Acquisition Cost.

    Shows what the maximum acquisition cost could be for each listing
    to remain affordable. Compares actual cost to max viable cost.
    """
    from models.param_mapper import calculate_min_rent_for_dscr
    from models.income_distribution import BCIncomeDistribution
    from scipy import stats

    if portfolio_df.empty:
        return go.Figure().update_layout(title="No data available")

    df = portfolio_df.copy()

    # Get parameters
    affordability_ratio = fixed_params.get('affordability_ratio', 0.30)
    target_percentile = fixed_params.get('target_percentile', 50)
    median_income = fixed_params.get('median_income', 95000)
    max_ltv = fixed_params.get('max_ltv', 0.95)
    interest_rate = fixed_params.get('interest_rate_senior', 0.045)
    opex_ratio = fixed_params.get('opex_ratio', 0.35)
    equity_return = fixed_params.get('equity_return_required', 0.0)
    sqft_per_unit = PARAM_INFO.get('sqft_per_unit', {}).get('default', 800)

    # Calculate target income
    mu = np.log(median_income)
    sigma = 0.65
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    target_income = dist.ppf(target_percentile / 100)

    # Max affordable rent at target income
    max_annual_rent = target_income * affordability_ratio
    max_monthly_rent = max_annual_rent / 12
    max_rent_psf = max_monthly_rent / sqft_per_unit

    # Reverse the min_rent formula to get max acquisition cost:
    # min_rent = cost × (LTV × payment_factor + equity_return × (1 - LTV)) / (12 × (1 - opex))
    # cost = min_rent × 12 × (1 - opex) / (LTV × payment_factor + equity_return × (1 - LTV))

    # Calculate payment factor
    r = interest_rate
    n = 50  # amortization
    monthly_rate = r / 12
    num_payments = n * 12
    monthly_factor = (monthly_rate * (1 + monthly_rate)**num_payments) / \
                    ((1 + monthly_rate)**num_payments - 1)
    annual_payment_factor = monthly_factor * 12

    # Include equity return in denominator
    cost_factor = max_ltv * annual_payment_factor + equity_return * (1 - max_ltv)
    max_viable_cost = max_rent_psf * 12 * (1 - opex_ratio) / cost_factor

    # Prepare data for chart
    df = df.sort_values('acquisition_cost_psf', ascending=True).reset_index(drop=True)

    fig = go.Figure()

    # Actual cost bars
    colors = ['#2ECC71' if c <= max_viable_cost else '#E74C3C'
              for c in df['acquisition_cost_psf']]

    fig.add_trace(go.Bar(
        x=df['name'],
        y=df['acquisition_cost_psf'],
        name='Actual Cost',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Actual: $%{y:.0f}/sqft<extra></extra>'
    ))

    # Max viable cost line
    fig.add_hline(
        y=max_viable_cost,
        line_dash='dash',
        line_color='blue',
        line_width=3,
        annotation_text=f'Max Viable: ${max_viable_cost:.0f}/sqft',
        annotation_position='top left'
    )

    fig.update_layout(
        title=f'Acquisition Cost vs Maximum Viable<br><sub>Max viable cost at {target_percentile:.0f}th percentile, {affordability_ratio:.0%} burden</sub>',
        xaxis_title='Listing',
        yaxis_title='Acquisition Cost ($/sqft)',
        yaxis=dict(tickformat='$,.0f'),
        height=450,
        template='plotly_white',
        xaxis_tickangle=-45
    )

    return fig


def create_subsidy_required_chart(
    portfolio_df: pd.DataFrame,
    fixed_params: dict,
) -> go.Figure:
    """
    3. Subsidy Required to Make Viable.

    Shows how much acquisition cost reduction (subsidy) would be needed
    to make each listing affordable. Listings already viable show $0.
    """
    from scipy import stats

    if portfolio_df.empty:
        return go.Figure().update_layout(title="No data available")

    df = portfolio_df.copy()

    # Get parameters
    affordability_ratio = fixed_params.get('affordability_ratio', 0.30)
    target_percentile = fixed_params.get('target_percentile', 50)
    median_income = fixed_params.get('median_income', 95000)
    max_ltv = fixed_params.get('max_ltv', 0.95)
    interest_rate = fixed_params.get('interest_rate_senior', 0.045)
    opex_ratio = fixed_params.get('opex_ratio', 0.35)
    equity_return = fixed_params.get('equity_return_required', 0.0)
    sqft_per_unit = PARAM_INFO.get('sqft_per_unit', {}).get('default', 800)

    # Calculate target income and max viable cost
    mu = np.log(median_income)
    sigma = 0.65
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    target_income = dist.ppf(target_percentile / 100)

    max_annual_rent = target_income * affordability_ratio
    max_monthly_rent = max_annual_rent / 12
    max_rent_psf = max_monthly_rent / sqft_per_unit

    r = interest_rate
    n = 50
    monthly_rate = r / 12
    num_payments = n * 12
    monthly_factor = (monthly_rate * (1 + monthly_rate)**num_payments) / \
                    ((1 + monthly_rate)**num_payments - 1)
    annual_payment_factor = monthly_factor * 12

    # Include equity return in denominator
    cost_factor = max_ltv * annual_payment_factor + equity_return * (1 - max_ltv)
    max_viable_cost = max_rent_psf * 12 * (1 - opex_ratio) / cost_factor

    # Calculate subsidy needed
    df['subsidy_psf'] = np.maximum(0, df['acquisition_cost_psf'] - max_viable_cost)
    df['subsidy_pct'] = df['subsidy_psf'] / df['acquisition_cost_psf']

    # Sort by subsidy needed
    df = df.sort_values('subsidy_psf', ascending=False).reset_index(drop=True)

    fig = go.Figure()

    # Subsidy bars
    fig.add_trace(go.Bar(
        x=df['name'],
        y=df['subsidy_psf'],
        name='Subsidy Needed',
        marker_color=['#E74C3C' if s > 0 else '#2ECC71' for s in df['subsidy_psf']],
        text=[f'{p:.0%}' if p > 0 else 'Viable' for p in df['subsidy_pct']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Subsidy: $%{y:.0f}/sqft<br>(%{text})<extra></extra>'
    ))

    fig.update_layout(
        title=f'Subsidy Required for Affordability<br><sub>Cost reduction needed to meet {affordability_ratio:.0%} burden at {target_percentile:.0f}th percentile</sub>',
        xaxis_title='Listing',
        yaxis_title='Subsidy Required ($/sqft)',
        yaxis=dict(tickformat='$,.0f'),
        height=450,
        template='plotly_white',
        xaxis_tickangle=-45
    )

    return fig


def create_income_percentile_required_chart(
    portfolio_df: pd.DataFrame,
    fixed_params: dict,
) -> go.Figure:
    """
    4. Income Percentile Required for each listing.

    Shows what income percentile a household needs to be at to afford each listing.
    Lower is better (more inclusive).
    """
    from scipy import stats

    if portfolio_df.empty:
        return go.Figure().update_layout(title="No data available")

    df = portfolio_df.copy()

    # Get parameters
    affordability_ratio = fixed_params.get('affordability_ratio', 0.30)
    target_percentile = fixed_params.get('target_percentile', 50)
    median_income = fixed_params.get('median_income', 95000)
    sqft_per_unit = PARAM_INFO.get('sqft_per_unit', {}).get('default', 800)

    # Calculate required income percentile for each listing
    mu = np.log(median_income)
    sigma = 0.65
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    def calc_required_percentile(min_rent_psf):
        if pd.isna(min_rent_psf):
            return None
        monthly_rent = min_rent_psf * sqft_per_unit
        annual_rent = monthly_rent * 12
        income_required = annual_rent / affordability_ratio
        percentile = dist.cdf(income_required) * 100
        return percentile

    df['required_percentile'] = df['min_rent_psf'].apply(calc_required_percentile)
    df = df.dropna(subset=['required_percentile'])
    df = df.sort_values('required_percentile', ascending=True).reset_index(drop=True)

    fig = go.Figure()

    # Color based on whether meets target percentile
    colors = ['#2ECC71' if p <= target_percentile else '#E74C3C'
              for p in df['required_percentile']]

    fig.add_trace(go.Bar(
        x=df['name'],
        y=df['required_percentile'],
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br>Required: %{y:.0f}th percentile<extra></extra>'
    ))

    # Target percentile line
    fig.add_hline(
        y=target_percentile,
        line_dash='dash',
        line_color='blue',
        line_width=3,
        annotation_text=f'Target: {target_percentile:.0f}th',
        annotation_position='top right'
    )

    fig.update_layout(
        title=f'Income Percentile Required for Each Listing<br><sub>Lower = more affordable (green = meets {target_percentile:.0f}th percentile target)</sub>',
        xaxis_title='Listing',
        yaxis_title='Income Percentile Required',
        yaxis=dict(tickformat='.0f', range=[0, 100]),
        height=450,
        template='plotly_white',
        xaxis_tickangle=-45
    )

    return fig


def create_sensitivity_tornado_chart(
    base_params: dict,
    variations: dict = None,
) -> go.Figure:
    """
    5. Sensitivity Tornado Chart.

    Shows which parameters have the biggest impact on rent burden.
    Each parameter is varied ±20% and the impact on burden is measured.
    """
    from models.param_mapper import calculate_min_rent_for_dscr
    from scipy import stats

    if variations is None:
        variations = {
            'acquisition_cost_psf': ('Acquisition Cost', 0.20),
            'interest_rate_senior': ('Interest Rate', 0.20),
            'max_ltv': ('Loan-to-Value', 0.10),
            'opex_ratio': ('Operating Expenses', 0.20),
            'median_income': ('Median Income', 0.20),
            'affordability_ratio': ('Affordability Ratio', 0.20),
        }

    # Base case calculation
    base_cost = base_params.get('acquisition_cost_psf', 400)
    interest = base_params.get('interest_rate_senior', 0.045)
    ltv = base_params.get('max_ltv', 0.95)
    opex = base_params.get('opex_ratio', 0.35)
    median_income = base_params.get('median_income', 95000)
    affordability_ratio = base_params.get('affordability_ratio', 0.30)
    target_percentile = base_params.get('target_percentile', 50)
    sqft_per_unit = PARAM_INFO.get('sqft_per_unit', {}).get('default', 800)

    def calculate_burden(params_override):
        p = {**base_params, **params_override}
        min_rent = calculate_min_rent_for_dscr(
            p['acquisition_cost_psf'],
            p['interest_rate_senior'],
            p['max_ltv'],
            p['opex_ratio'],
            50,
            p.get('equity_return_required', 0.0)
        )
        monthly_rent = min_rent * sqft_per_unit
        annual_rent = monthly_rent * 12

        mu = np.log(p['median_income'])
        sigma = 0.65
        dist = stats.lognorm(s=sigma, scale=np.exp(mu))
        target_income = dist.ppf(p['target_percentile'] / 100)

        return annual_rent / target_income if target_income > 0 else 1.0

    base_burden = calculate_burden({})

    results = []
    for param, (label, variation) in variations.items():
        base_val = base_params.get(param, PARAM_INFO.get(param, {}).get('default', 1))
        if base_val is None or base_val == 0:
            continue

        # Low and high values
        low_val = base_val * (1 - variation)
        high_val = base_val * (1 + variation)

        burden_low = calculate_burden({param: low_val})
        burden_high = calculate_burden({param: high_val})

        # Store impact (high - low gives the swing)
        results.append({
            'param': label,
            'low_impact': (burden_low - base_burden) * 100,
            'high_impact': (burden_high - base_burden) * 100,
            'total_swing': abs(burden_high - burden_low) * 100
        })

    # Sort by total swing
    results = sorted(results, key=lambda x: x['total_swing'], reverse=True)

    fig = go.Figure()

    params = [r['param'] for r in results]
    low_impacts = [r['low_impact'] for r in results]
    high_impacts = [r['high_impact'] for r in results]

    # Low impact (left side)
    fig.add_trace(go.Bar(
        y=params,
        x=low_impacts,
        orientation='h',
        name='-20%',
        marker_color='#3498DB',
        hovertemplate='%{y}: %{x:+.1f}pp<extra></extra>'
    ))

    # High impact (right side)
    fig.add_trace(go.Bar(
        y=params,
        x=high_impacts,
        orientation='h',
        name='+20%',
        marker_color='#E74C3C',
        hovertemplate='%{y}: %{x:+.1f}pp<extra></extra>'
    ))

    fig.add_vline(x=0, line_color='black', line_width=2)

    fig.update_layout(
        title=f'Sensitivity Analysis: Impact on Rent Burden<br><sub>Base burden: {base_burden:.1%} | Each parameter varied ±20%</sub>',
        xaxis_title='Change in Rent Burden (percentage points)',
        yaxis_title='',
        xaxis=dict(tickformat='+.1f', ticksuffix='pp'),
        barmode='overlay',
        height=400,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    return fig


def create_affordability_frontier_chart(
    fixed_params: dict,
    cost_range: Tuple[float, float] = (100, 800),
    n_points: int = 50
) -> go.Figure:
    """
    6. Affordability Frontier.

    Shows the relationship between acquisition cost and the minimum income
    percentile required for affordability. This is the fundamental trade-off
    curve for affordable housing policy.
    """
    from scipy import stats

    # Get parameters
    affordability_ratio = fixed_params.get('affordability_ratio', 0.30)
    median_income = fixed_params.get('median_income', 95000)
    max_ltv = fixed_params.get('max_ltv', 0.95)
    interest_rate = fixed_params.get('interest_rate_senior', 0.045)
    opex_ratio = fixed_params.get('opex_ratio', 0.35)
    equity_return = fixed_params.get('equity_return_required', 0.0)
    target_percentile = fixed_params.get('target_percentile', 50)
    sqft_per_unit = PARAM_INFO.get('sqft_per_unit', {}).get('default', 800)

    # Set up income distribution
    mu = np.log(median_income)
    sigma = 0.65
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    # Calculate payment factor
    r = interest_rate
    n = 50
    monthly_rate = r / 12
    num_payments = n * 12
    monthly_factor = (monthly_rate * (1 + monthly_rate)**num_payments) / \
                    ((1 + monthly_rate)**num_payments - 1)
    annual_payment_factor = monthly_factor * 12

    # Cost factor includes equity return requirement
    cost_factor = max_ltv * annual_payment_factor + equity_return * (1 - max_ltv)

    # Calculate for each acquisition cost
    costs = np.linspace(cost_range[0], cost_range[1], n_points)
    percentiles = []

    for cost in costs:
        # Min rent for DSCR >= 1 AND equity return
        min_rent_psf = (cost * cost_factor) / (12 * (1 - opex_ratio))
        monthly_rent = min_rent_psf * sqft_per_unit
        annual_rent = monthly_rent * 12
        income_required = annual_rent / affordability_ratio
        percentile = dist.cdf(income_required) * 100
        percentiles.append(percentile)

    fig = go.Figure()

    # Main frontier curve
    fig.add_trace(go.Scatter(
        x=costs,
        y=percentiles,
        mode='lines',
        name='Affordability Frontier',
        line=dict(color='#3498DB', width=3),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)',
        hovertemplate='Cost: $%{x:.0f}/sqft<br>Requires: %{y:.0f}th percentile<extra></extra>'
    ))

    # Fill "affordable zone" below
    fig.add_trace(go.Scatter(
        x=costs,
        y=[0] * len(costs),
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Target percentile reference
    fig.add_hline(
        y=target_percentile,
        line_dash='dash',
        line_color='green',
        annotation_text=f'Target: {target_percentile:.0f}th percentile',
        annotation_position='right'
    )

    # Find the max viable cost at target percentile
    target_income = dist.ppf(target_percentile / 100)
    max_annual_rent = target_income * affordability_ratio
    max_monthly_rent = max_annual_rent / 12
    max_rent_psf = max_monthly_rent / sqft_per_unit
    max_viable_cost = max_rent_psf * 12 * (1 - opex_ratio) / cost_factor

    # Mark the intersection point
    fig.add_trace(go.Scatter(
        x=[max_viable_cost],
        y=[target_percentile],
        mode='markers',
        marker=dict(size=15, color='green', symbol='star'),
        name=f'Max Viable: ${max_viable_cost:.0f}/sqft',
        hovertemplate=f'Max Viable Cost<br>${max_viable_cost:.0f}/sqft<extra></extra>'
    ))

    fig.add_vline(
        x=max_viable_cost,
        line_dash='dot',
        line_color='green',
        annotation_text=f'Max: ${max_viable_cost:.0f}/sqft',
        annotation_position='top'
    )

    # Add shaded regions
    fig.add_vrect(
        x0=cost_range[0], x1=max_viable_cost,
        fillcolor='rgba(46, 204, 113, 0.1)',
        layer='below',
        line_width=0,
    )
    fig.add_vrect(
        x0=max_viable_cost, x1=cost_range[1],
        fillcolor='rgba(231, 76, 60, 0.1)',
        layer='below',
        line_width=0,
    )

    # Add zone labels
    fig.add_annotation(
        x=(cost_range[0] + max_viable_cost) / 2,
        y=10,
        text="Affordable Zone",
        showarrow=False,
        font=dict(size=14, color='green')
    )
    fig.add_annotation(
        x=(max_viable_cost + cost_range[1]) / 2,
        y=10,
        text="Subsidy Required",
        showarrow=False,
        font=dict(size=14, color='red')
    )

    fig.update_layout(
        title=f'Affordability Frontier<br><sub>Acquisition cost vs minimum income percentile needed ({affordability_ratio:.0%} burden threshold)</sub>',
        xaxis_title='Acquisition Cost ($/sqft)',
        yaxis_title='Minimum Income Percentile Required',
        xaxis=dict(tickformat='$,.0f'),
        yaxis=dict(tickformat='.0f', range=[0, 100]),
        height=500,
        template='plotly_white',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )

    return fig
