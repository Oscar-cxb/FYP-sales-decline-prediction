"""
customers.py - drill-down list behind the KPI cards on the overview.

Reached by clicking a KPI card. Dash passes query string params
straight into layout() as keyword args, so /customers?band=high
arrives here as band="high".

The band (low/watch/high) comes from the url, then the three
dropdowns filter further on top of it - segment, product category,
payment type. Those go through a callback since they change after
the page has already loaded.
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, dash_table, callback, Input, Output
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import load_risk_table
from theme import (
    LOW, MID, HIGH, TEXT, TEXT_DIM, TEXT_FAINT, LINE, LINE_SOFT, MONO, SANS,
)

dash.register_page(__name__, path="/customers", name="Customers")

WATCH_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.50

BANDS = {
    "all":   ("All customers", TEXT_DIM),
    "low":   ("Low risk", LOW),
    "watch": ("Watch", MID),
    "high":  ("High risk", HIGH),
}


def band_of(score):
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= WATCH_THRESHOLD:
        return "mid"
    return "low"


def filter_by_band(df, band):
    if band == "low":
        return df[df["risk_score"] < WATCH_THRESHOLD]
    if band == "watch":
        return df[(df["risk_score"] >= WATCH_THRESHOLD) & (df["risk_score"] < HIGH_THRESHOLD)]
    if band == "high":
        return df[df["risk_score"] >= HIGH_THRESHOLD]
    return df


def options_from(df, col):
    return [{"label": v, "value": v} for v in sorted(df[col].dropna().unique())]


# ----------------------------------------------------------- the table

def build_table(df, sort_by):
    view = df.sort_values(sort_by, ascending=False).copy()

    view["risk_band"] = view["risk_score"].map(band_of)
    view["monetary"] = view["monetary"].map(lambda v: f"{v:,.0f}")
    view["avg_delivery_delay"] = view["avg_delivery_delay"].map(lambda v: f"{v:+.0f}")
    view["risk_score"] = view["risk_score"].map(lambda v: f"{v:.3f}")
    view["customer_unique_id"] = view["customer_unique_id"].str[:12]

    cols = [
        ("customer_unique_id", "Customer"),
        ("segment_name", "Segment"),
        ("monetary", "Spend (R$)"),
        ("frequency", "Orders"),
        ("recency", "Recency (d)"),
        ("avg_delivery_delay", "Delivery (d)"),
        ("dominant_category", "Top category"),
        ("dominant_payment_type", "Payment"),
        ("risk_score", "Risk"),
    ]
    numeric = ["monetary", "frequency", "recency", "avg_delivery_delay", "risk_score"]

    return dash_table.DataTable(
        data=view[[c for c, _ in cols] + ["risk_band"]].to_dict("records"),
        columns=[{"name": label, "id": cid} for cid, label in cols],
        sort_action="native",
        page_size=20,
        style_as_list_view=True,
        style_header={
            "backgroundColor": "rgba(0,0,0,0)",
            "color": TEXT_FAINT,
            "fontWeight": "500",
            "fontSize": "11px",
            "border": "none",
            "borderBottom": f"1px solid {LINE}",
            "padding": "0 14px 11px 0",
            "textAlign": "left",
        },
        style_cell={
            "backgroundColor": "rgba(0,0,0,0)",
            "color": TEXT,
            "border": "none",
            "padding": "11px 14px 11px 0",
            "fontSize": "12.5px",
            "fontFamily": SANS,
            "textAlign": "left",
            "maxWidth": 0,
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "right"} for c in numeric
        ],
        style_data_conditional=[
            {"if": {"column_id": "customer_unique_id"},
             "fontFamily": MONO, "color": TEXT_DIM, "fontSize": "11.5px"},
            {"if": {"column_id": ["segment_name", "dominant_category", "dominant_payment_type"]},
             "color": TEXT_DIM},
            {"if": {"column_id": numeric}, "fontFamily": MONO, "fontSize": "12px"},
            {"if": {"filter_query": "{risk_band} = high", "column_id": "risk_score"},
             "color": HIGH, "fontWeight": "600"},
            {"if": {"filter_query": "{risk_band} = mid", "column_id": "risk_score"},
             "color": MID},
            {"if": {"filter_query": "{risk_band} = low", "column_id": "risk_score"},
             "color": LOW},
        ],
    )


# ---------------------------------------------------------- the layout

def layout(band="all", sort="risk_score", **kwargs):
    # guarding both since anyone can type whatever they like in the url bar
    if band not in BANDS:
        band = "all"
    if sort not in ("risk_score", "monetary"):
        sort = "risk_score"

    df = load_risk_table()
    counts = {key: len(filter_by_band(df, key)) for key in BANDS}
    view = filter_by_band(df, band)
    title, _ = BANDS[band]

    return html.Div([
        dcc.Link("\u2190 Back to overview", href="/", className="back-link"),

        html.Div([
            html.H1(title),
            html.P(
                "Narrow the list with the filters below, or click any column "
                "header to re-sort. Risk band comes from the card you clicked.",
                className="page-sub",
            ),
        ], className="page-head"),

        # band switcher - stays as links so the url keeps reflecting the band
        html.Div([
            dcc.Link([
                label,
                html.Span(f"{counts[key]:,}", className="pill-count"),
            ], href=f"/customers?band={key}&sort={sort}", className="pill",
                style={"borderColor": colour, "color": colour} if key == band else {})
            for key, (label, colour) in BANDS.items()
        ], className="pill-row"),

        # the band + sort get parked in stores so the callback can read them
        # without having to re-parse the url
        dcc.Store(id="cust-band", data=band),
        dcc.Store(id="cust-sort", data=sort),

        html.Div([
            html.Div([
                html.Label("Segment", className="filter-label"),
                dcc.Dropdown(
                    id="cust-segment",
                    options=options_from(view, "segment_name"),
                    multi=True, placeholder="All segments",
                    className="dash-dropdown",
                ),
            ], className="filter-field"),

            html.Div([
                html.Label("Top category", className="filter-label"),
                dcc.Dropdown(
                    id="cust-category",
                    options=options_from(view, "dominant_category"),
                    multi=True, placeholder="All categories",
                    className="dash-dropdown",
                ),
            ], className="filter-field"),

            html.Div([
                html.Label("Payment type", className="filter-label"),
                dcc.Dropdown(
                    id="cust-payment",
                    options=options_from(view, "dominant_payment_type"),
                    multi=True, placeholder="All payment types",
                    className="dash-dropdown",
                ),
            ], className="filter-field"),

            html.Div(id="cust-count", className="filter-count"),
        ], className="filter-row"),

        html.Div(html.Div(id="cust-table"), className="card"),
    ])


# -------------------------------------------------------- the callback

@callback(
    Output("cust-table", "children"),
    Output("cust-count", "children"),
    Input("cust-segment", "value"),
    Input("cust-category", "value"),
    Input("cust-payment", "value"),
    Input("cust-band", "data"),
    Input("cust-sort", "data"),
)
def update_table(segments, categories, payments, band, sort):
    df = filter_by_band(load_risk_table(), band or "all")

    # dropdowns come back as None (untouched) or [] (everything cleared) -
    # both mean "dont filter on this one"
    if segments:
        df = df[df["segment_name"].isin(segments)]
    if categories:
        df = df[df["dominant_category"].isin(categories)]
    if payments:
        df = df[df["dominant_payment_type"].isin(payments)]

    if df.empty:
        return (
            html.P("No customers match these filters.",
                   style={"color": TEXT_FAINT, "fontSize": "13px", "padding": "18px 0"}),
            "",
        )

    spend = df["monetary"].sum()
    summary = html.Span([
        html.B(f"{len(df):,}"), " customers · ",
        html.B(f"R$ {spend:,.0f}"), " spend",
    ])

    return build_table(df, sort or "risk_score"), summary