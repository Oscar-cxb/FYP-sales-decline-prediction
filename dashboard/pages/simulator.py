"""
simulator.py — dashboard page stub.
"""

import dash
from dash import html

dash.register_page(__name__, path="/simulator")

layout = html.Div([
    html.H1("Simulator"),
])
