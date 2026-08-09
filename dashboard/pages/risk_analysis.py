"""
risk_analysis.py — dashboard page stub.
"""

import dash
from dash import html

dash.register_page(__name__, path="/risk-analysis")

layout = html.Div([
    html.H1("Risk Analysis"),
])
