"""
segments.py — dashboard page stub.
"""

import dash
from dash import html

dash.register_page(__name__, path="/segments")

layout = html.Div([
    html.H1("Segments"),
])
