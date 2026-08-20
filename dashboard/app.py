"""
app.py

Plotly Dash entrypoint. Dark multi-page dashboard.

run with:  python dashboard/app.py
then open  http://127.0.0.1:8050/

the css is injected through index_string from theme.py rather than
sitting in assets/style.css - the assets folder wasnt getting picked up
reliably and this removes that whole failure mode
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# joblib still uses an old numpy call to restore array shapes when loading
# the model pickles. harmless, but it prints a DeprecationWarning on every
# start and looks like something is broken when youre screen sharing
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# grouped, because there are two levels of analysis here and a flat list of
# seven made them all look like the same kind of thing.
# customers first because "/" lands on the customer Overview, so the nav
# order matches where the app actually opens. the seller pages are the
# primary model though - worth saying so when walking someone through it.
NAV_GROUPS = [
    ("Customers", [
        ("Overview", "/"),
        ("Segments", "/segments"),
        ("Customer list", "/customers"),
    ]),
    ("Sellers", [
        ("Seller risk", "/sellers"),
        ("Interventions", "/interventions"),
        ("Risk analysis", "/risk-analysis"),
        ("Simulator", "/simulator"),
    ]),
    ("Evidence", [
        ("Model evidence", "/model-evidence"),
    ]),
]


def nav():
    out = []
    for group, links in NAV_GROUPS:
        out.append(html.Div(group, className="nav-group"))
        out.extend(dcc.Link(label, href=href, className="nav-link")
                   for label, href in links)
    return out


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div("SD", className="brand-mark"),
            html.Div([
                html.Div("Sales Decline", className="brand-line-1"),
                html.Div("Risk Prediction", className="brand-line-2"),
            ]),
        ], className="brand"),

        html.Nav(nav(), className="nav"),

        html.Div("Olist · 2016–2018", className="sidebar-footer"),
    ], className="sidebar"),

    html.Div(dash.page_container, className="content"),
], className="app-shell")


if __name__ == "__main__":
    app.run(debug=True)