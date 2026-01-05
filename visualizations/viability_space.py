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
