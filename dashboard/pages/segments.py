"""
segments.py - the customer segments from MiniBatchKMeans.

This is the customer half of the project. The Seller Risk page answers
"whose sales will fall next month"; this one answers "what kinds of
customer are behind those sales".

Segments come from clustering.py - k=6, chosen by silhouette score, on
scaled recency / frequency / log monetary / delivery delay. The names
are assigned by ranking the clusters on those same scaled features, so
"Champions" is genuinely the best cluster rather than a label i picked.

WORTH KNOWING WHEN READING THIS PAGE
Frequency is nearly flat across the whole dataset - about 90% of olist
customers order exactly once - so only one cluster has any real repeat
purchasing. the segments are therefore driven mostly by recency,
monetary and delivery. thats a property of a one-shot marketplace, not
a bug in the clustering, and its called out on the page rather than
hidden.
"""

import sys
from functools import lru_cache
from pathlib import Path

import dash
from dash import html, dcc, dash_table, callback, Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import load_risk_table
from theme import (
    VIOLET, CYAN, AMBER, ROSE, EMERALD, TEXT, TEXT_DIM, TEXT_FAINT,
    LINE, LINE_SOFT, CARD, CARD_2, MONO, SANS, rgba,
)

dash.register_page(__name__, path="/segments", name="Segments")

# ordered best -> worst, matching how clustering.py ranks them
SEGMENT_ORDER = ["Champions", "Promising", "Needs Attention",
                 "At Risk", "Delayed Delivery", "Hibernating"]

SEGMENT_COLOUR = {
    "Champions": EMERALD,
    "Promising": CYAN,
    "Needs Attention": VIOLET,
    "At Risk": AMBER,
    "Delayed Delivery": "#f97316",
    "Hibernating": ROSE,
}

# one line each, so the names arent left to be guessed at
SEGMENT_NOTE = {
    "Champions": "the only segment with real repeat purchasing",
    "Promising": "recent and high spending, but still one-time buyers",
    "Needs Attention": "spent well, then went quiet",
    "At Risk": "low spend and drifting away",
    "Delayed Delivery": "defined by late deliveries, not by low value",
    "Hibernating": "dormant — longest gap since their last order",
}

SCATTER_SAMPLE = 6000   # 77k points is more than a browser wants to draw


@lru_cache(maxsize=1)
def _customers():
    """cached read - the csv is 77k rows and both the layout and the
    checkbox callback need it. reading it fresh every time made the page
    feel sluggish for no reason."""
    return load_risk_table()

LABEL_STYLE = {
    "display": "inline-flex", "alignItems": "center", "gap": "7px",
    "marginRight": "8px", "padding": "6px 12px",
    "border": f"1px solid {LINE_SOFT}", "borderRadius": "20px",
    "background": CARD_2, "color": TEXT_DIM,
    "fontSize": "12.5px", "cursor": "pointer",
}
INPUT_STYLE = {"accentColor": VIOLET, "width": "13px", "height": "13px",
               "margin": "0", "cursor": "pointer"}
CONTROL_BAR = {"display": "flex", "flexWrap": "wrap", "alignItems": "center",
               "gap": "18px", "marginBottom": "16px"}
CONTROL_LABEL = {"fontSize": "11px", "color": TEXT_FAINT,
                 "textTransform": "uppercase", "letterSpacing": "0.07em"}
SWATCH = {"width": "8px", "height": "8px", "borderRadius": "50%",
          "display": "inline-block"}
EMPTY_NOTE = {"color": TEXT_FAINT, "fontSize": "12.5px",
              "padding": "30px 0", "textAlign": "center"}


def present_segments(df):
    """segments actually in the data, in rank order, ignoring any that arent."""
    have = set(df["segment_name"].unique())
    ordered = [s for s in SEGMENT_ORDER if s in have]
    return ordered + sorted(have - set(ordered))


def base_fig(fig, height):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, size=11, family=SANS),
        margin=dict(l=6, r=16, t=6, b=34), height=height, showlegend=False,
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE,
                        font=dict(color=TEXT, size=11.5, family=SANS)),
    )
    fig.update_xaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT)
    fig.update_yaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT)
    return fig


def kpi_card(icon, colour, label, value, sub=None):
    children = [
        html.Div([
            html.Div(icon, className="kpi-chip",
                     style={"background": rgba(colour, 0.14), "color": colour}),
            html.Div(label, className="kpi-label"),
        ], className="kpi-top"),
        html.Div(value, className="kpi-value"),
    ]
    if sub:
        children.append(html.Div(sub, className="kpi-sub"))
    return html.Div(children, className="kpi-card")


# ------------------------------------------------------------- charts

def build_scatter(df, segments):
    """recency against spend, one trace per segment so the legend filters.

    monetary is plotted on a log axis - the raw spread runs from a few
    reais to over 13,000, so on a linear axis everyone piles into the
    bottom inch of the chart
    """
    d = df[df["segment_name"].isin(segments)]
    if d.empty:
        return None
    if len(d) > SCATTER_SAMPLE:
        d = d.sample(SCATTER_SAMPLE, random_state=42)

    fig = go.Figure()
    for seg in present_segments(d):
        s = d[d["segment_name"] == seg]
        fig.add_trace(go.Scattergl(
            x=s["recency"], y=s["monetary"], mode="markers", name=seg,
            marker=dict(size=4, opacity=0.55,
                        color=SEGMENT_COLOUR.get(seg, TEXT_DIM),
                        line=dict(width=0)),
            hovertemplate=(f"<b>{seg}</b><br>recency %{{x:.0f}} days"
                           "<br>spend R$ %{y:,.0f}<extra></extra>"),
        ))
    fig = base_fig(fig, height=330)
    fig.update_xaxes(title="Days since last order")
    fig.update_yaxes(title="Total spend (R$)", type="log")
    fig.update_layout(margin=dict(l=6, r=16, t=6, b=40))
    return fig


def build_size_chart(stats):
    d = stats.sort_values("customers")
    fig = go.Figure(go.Bar(
        x=d["customers"], y=d["segment_name"], orientation="h",
        marker=dict(color=[SEGMENT_COLOUR.get(s, TEXT_DIM) for s in d["segment_name"]],
                    line=dict(width=0)),
        width=0.6, customdata=d["share"],
        text=[f"{v:,}" for v in d["customers"]], textposition="outside",
        textfont=dict(size=10.5, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>%{x:,} customers (%{customdata:.1%})<extra></extra>",
    ))
    fig = base_fig(fig, height=280)
    fig.update_xaxes(title=None, range=[0, d["customers"].max() * 1.22])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=52, t=6, b=26))
    return fig


def build_profile_chart(stats, metric, title):
    d = stats.sort_values(metric)
    fig = go.Figure(go.Bar(
        x=d[metric], y=d["segment_name"], orientation="h",
        marker=dict(color=[SEGMENT_COLOUR.get(s, TEXT_DIM) for s in d["segment_name"]],
                    line=dict(width=0)),
        width=0.6,
        text=[f"{v:,.1f}" for v in d[metric]], textposition="outside",
        textfont=dict(size=10.5, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>" + title + " %{x:,.1f}<extra></extra>",
    ))
    fig = base_fig(fig, height=250)
    lo = min(0, d[metric].min() * 1.25)
    fig.update_xaxes(title=None, range=[lo, d[metric].max() * 1.25])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=52, t=6, b=26))
    return fig


# -------------------------------------------------------------- table

def segment_stats(df):
    g = df.groupby("segment_name").agg(
        customers=("customer_unique_id", "count"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        delay=("avg_delivery_delay", "mean"),
        risk=("risk_score", "mean"),
        spend=("monetary", "sum"),
    ).reset_index()

    g["share"] = g["customers"] / g["customers"].sum()
    g["spend_share"] = g["spend"] / g["spend"].sum()

    order = present_segments(df)
    g["__rank"] = g["segment_name"].map({s: i for i, s in enumerate(order)})
    return g.sort_values("__rank").drop(columns="__rank").reset_index(drop=True)


def build_table(stats):
    t = stats.copy()
    t["customers_f"] = t["customers"].map(lambda v: f"{v:,}")
    t["share_f"] = t["share"].map(lambda v: f"{v:.1%}")
    t["recency_f"] = t["recency"].map(lambda v: f"{v:,.0f}")
    t["frequency_f"] = t["frequency"].map(lambda v: f"{v:.2f}")
    t["monetary_f"] = t["monetary"].map(lambda v: f"{v:,.0f}")
    t["delay_f"] = t["delay"].map(lambda v: f"{v:+.1f}")
    t["spend_share_f"] = t["spend_share"].map(lambda v: f"{v:.1%}")
    t["note"] = t["segment_name"].map(SEGMENT_NOTE).fillna("")

    cols = [
        ("segment_name", "Segment"), ("customers_f", "Customers"),
        ("share_f", "Share"), ("recency_f", "Recency (d)"),
        ("frequency_f", "Orders"), ("monetary_f", "Spend (R$)"),
        ("delay_f", "Delivery (d)"), ("spend_share_f", "of revenue"),
        ("note", "What defines them"),
    ]
    numeric = ["customers_f", "share_f", "recency_f", "frequency_f",
               "monetary_f", "delay_f", "spend_share_f"]

    return dash_table.DataTable(
        data=t[[c for c, _ in cols]].to_dict("records"),
        columns=[{"name": lbl, "id": cid} for cid, lbl in cols],
        style_as_list_view=True,
        style_header={
            "backgroundColor": "rgba(0,0,0,0)", "color": TEXT_FAINT,
            "fontWeight": "500", "fontSize": "11px", "border": "none",
            "borderBottom": f"1px solid {LINE}",
            "padding": "0 14px 11px 0", "textAlign": "left",
        },
        style_cell={
            "backgroundColor": "rgba(0,0,0,0)", "color": TEXT, "border": "none",
            "padding": "11px 14px 11px 0", "fontSize": "12.5px",
            "fontFamily": SANS, "textAlign": "left",
        },
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "right"} for c in numeric
        ] + [{"if": {"column_id": "note"}, "textAlign": "left",
              "paddingLeft": "18px"}],
        style_data_conditional=[
            {"if": {"column_id": numeric}, "fontFamily": MONO, "fontSize": "12px"},
            {"if": {"column_id": "note"}, "color": TEXT_FAINT, "fontSize": "11.5px"},
        ] + [
            {"if": {"filter_query": f"{{segment_name}} = '{seg}'",
                    "column_id": "segment_name"},
             "color": colour, "fontWeight": "600"}
            for seg, colour in SEGMENT_COLOUR.items()
        ],
    )


# ------------------------------------------------------------- layout

def layout():
    try:
        df = _customers()
    except FileNotFoundError:
        return html.Div([
            html.H1("Segments"),
            html.P("Run the customer pipeline first — data_pipeline.py, "
                   "rfm_features.py, sentiment.py, extra_features.py, "
                   "clustering.py, then classifier.py.", className="page-sub"),
        ])

    if "segment_name" not in df.columns:
        return html.Div([
            html.H1("Segments"),
            html.P("No segment_name column — rerun src/clustering.py then "
                   "src/classifier.py.", className="page-sub"),
        ])

    stats = segment_stats(df)
    order = present_segments(df)

    biggest = stats.loc[stats["customers"].idxmax()]
    riskiest = stats.loc[stats["risk"].idxmax()]
    repeat = stats.loc[stats["frequency"].idxmax()]

    return html.Div([
        html.Div([
            html.Div([
                html.H1("Customer segments"),
                html.P(
                    f"{len(df):,} customers grouped by MiniBatchKMeans into "
                    f"{len(order)} segments, using recency, frequency, spend and "
                    f"delivery performance. Names come from ranking the clusters "
                    f"on those same features, not from picking labels by hand.",
                    className="page-sub"),
            ], className="page-head-text"),

            html.Div([
                html.Div([
                    html.Span("Method", className="model-stat-label"),
                    html.Span("MiniBatchKMeans", className="model-stat-value"),
                ], className="model-stat"),
                html.Div([
                    html.Span("Clusters", className="model-stat-label"),
                    html.Span(f"k = {len(order)}", className="model-stat-value accent"),
                ], className="model-stat"),
                html.Div([
                    html.Span("Chosen by", className="model-stat-label"),
                    html.Span("silhouette", className="model-stat-value"),
                ], className="model-stat"),
            ], className="model-strip"),
        ], className="page-head"),

        html.Div([
            kpi_card("\u25c8", VIOLET, "Segments", f"{len(order)}",
                     "k chosen by silhouette score"),
            kpi_card("\u25c9", SEGMENT_COLOUR.get(biggest["segment_name"], CYAN),
                     "Largest segment", biggest["segment_name"],
                     f"{biggest['customers']:,.0f} customers · {biggest['share']:.0%}"),
            kpi_card("\u25b2", ROSE, "Highest mean risk", riskiest["segment_name"],
                     f"risk {riskiest['risk']:.3f}"),
            kpi_card("\u21bb", EMERALD, "Most repeat buying", repeat["segment_name"],
                     f"{repeat['frequency']:.2f} orders per customer"),
        ], className="kpi-row"),

        html.Div([
            html.P("Where the segments sit", className="card-title"),
            html.P("Each dot is a customer — spend on a log scale, since it runs "
                   "from a few reais to over 13,000", className="card-sub"),

            html.Div([
                html.Div([
                    html.Span("Show", style=CONTROL_LABEL),
                    dcc.Checklist(
                        id="seg-filter",
                        options=[{"label": html.Span([
                            html.Span(style={**SWATCH,
                                             "background": SEGMENT_COLOUR.get(s, TEXT_DIM)}),
                            html.Span(s, style={"marginLeft": "6px"}),
                        ]), "value": s} for s in order],
                        value=order, inline=True,
                        labelStyle=LABEL_STYLE, inputStyle=INPUT_STYLE,
                        style={"display": "inline-flex", "flexWrap": "wrap"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "12px",
                          "flexWrap": "wrap"}),
            ], style=CONTROL_BAR),

            html.Div(id="seg-scatter"),
        ], className="card"),

        html.Div([
            html.Div([
                html.P("Segment sizes", className="card-title"),
                html.P("How the customer base splits", className="card-sub"),
                dcc.Graph(figure=build_size_chart(stats),
                          config={"displayModeBar": False}),
            ], className="card"),

            html.Div([
                html.P("Average spend per customer", className="card-title"),
                html.P("Where the money actually sits", className="card-sub"),
                dcc.Graph(figure=build_profile_chart(stats, "monetary", "spend"),
                          config={"displayModeBar": False}),
            ], className="card"),
        ], className="duo-row"),

        html.Div([
            html.Div([
                html.P("Average days since last order", className="card-title"),
                html.P("Lower is more recently active", className="card-sub"),
                dcc.Graph(figure=build_profile_chart(stats, "recency", "recency"),
                          config={"displayModeBar": False}),
            ], className="card"),

            html.Div([
                html.P("Average delivery delay", className="card-title"),
                html.P("Negative means early — most Olist orders beat the estimate",
                       className="card-sub"),
                dcc.Graph(figure=build_profile_chart(stats, "delay", "delay"),
                          config={"displayModeBar": False}),
                html.P("One segment stands out here with genuinely late deliveries, "
                       "which is what its name refers to.", className="chart-note"),
            ], className="card"),
        ], className="duo-row"),

        html.Div([
            html.P("Segment profiles", className="card-title"),
            html.P("The numbers behind each cluster", className="card-sub"),
            build_table(stats),
            html.P("Orders per customer is close to 1.00 in every segment except "
                   "one — roughly 90% of Olist customers buy exactly once. That "
                   "limits how much the Frequency part of RFM can separate anyone, "
                   "so these clusters are driven mainly by recency, spend and "
                   "delivery. It's a property of a one-shot marketplace rather "
                   "than a fault in the clustering.",
                   className="chart-note"),
        ], className="card"),
    ])


@callback(Output("seg-scatter", "children"), Input("seg-filter", "value"))
def update_scatter(segments):
    if not segments:
        return html.P("Tick at least one segment.", style=EMPTY_NOTE)

    df = _customers()
    fig = build_scatter(df, segments)
    if fig is None:
        return html.P("Nothing to plot for that selection.", style=EMPTY_NOTE)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})