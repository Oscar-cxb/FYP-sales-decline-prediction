"""
app.py

Plotly Dash entrypoint. Dark multi-page dashboard.
Pages: Overview, Segments, Risk Analysis, Interventions, Simulator.

run with:  python dashboard/app.py
then open  http://127.0.0.1:8050/

note: the css is injected through index_string from theme.py rather
than sitting in assets/style.css - the assets folder wasnt getting
picked up reliably and this removes that whole failure mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dash
from dash import Dash, html, dcc

from theme import CSS

app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    suppress_callback_exceptions=True,
)
app.title = "Sales Decline Risk"

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>__CSS__</style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
""".replace("__CSS__", CSS)

NAV_LINKS = [
    ("Overview", "/"),
    ("Seller Risk", "/sellers"),
    ("Segments", "/segments"),
    ("Risk Analysis", "/risk-analysis"),
    ("Interventions", "/interventions"),
    ("Simulator", "/simulator"),
]

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div("SD", className="brand-mark"),
            html.Div([
                html.Div("Sales Decline", className="brand-line-1"),
                html.Div("Risk Prediction", className="brand-line-2"),
            ]),
        ], className="brand"),

        html.Nav([
            dcc.Link(label, href=href, className="nav-link")
            for label, href in NAV_LINKS
        ], className="nav"),

        html.Div("Olist · 2016–2018", className="sidebar-footer"),
    ], className="sidebar"),

    html.Div(dash.page_container, className="content"),
], className="app-shell")


if __name__ == "__main__":
    app.run(debug=True)