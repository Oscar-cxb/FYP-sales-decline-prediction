"""
interventions.py — dashboard page stub.
"""

import dash
from dash import html

dash.register_page(__name__, path="/interventions")

layout = html.Div([
    html.H1("Interventions"),
])
