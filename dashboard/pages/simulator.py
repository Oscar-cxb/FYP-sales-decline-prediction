"""
simulator.py - what-if on a real seller.

Pick a seller off the worklist, move the sliders, watch the predicted
risk update live.

WHY IT STARTS FROM A REAL SELLER
The model needs all 49 features. If the sliders only cover 4 or 5 and
the rest get filled with medians, youre not simulating a seller at all,
youre simulating an average one with a few numbers bolted on. Starting
from an actual row means everything else stays true to that seller and
the sliders show a real marginal effect.

PROPAGATING THE CHANGE (this is the bit that matters)
Every level feature has derived momentum features built from it in
seller_features.py, and they all have to move together or the row
becomes internally inconsistent - e.g. review score 3.0 -> 4.5 while
d_avg_review_score still says the seller dropped half a star. the model
was trained on consistent rows, so feeding it a contradictory one gives
a meaningless answer.

given a level change of D on feature x:
    x       += D
    d_x     += D        (d_x = x - x_prev, and x_prev doesnt move)
    roll3_x += D/3      (mean of 3 months, only this one changed)
    trend_x += D/2      (least squares slope over 3 points: (c-a)/2,
                         so a change of D in c moves it by D/2)
    pct_x   recomputed from x_prev, which we can recover as x - d_x

run after seller_model.py
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, callback, Input, Output
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import load_seller_risk_table
from theme import (
    VIOLET, CYAN, AMBER, ROSE, EMERALD, TEXT, TEXT_DIM, TEXT_FAINT,
    LINE, LINE_SOFT, CARD, CARD_2, MONO, SANS,
)

dash.register_page(__name__, path="/simulator", name="Simulator")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# the levers. deliberately the ones a marketplace could actually do
# something about - you cant "decide" to have more revenue, but you can
# chase late deliveries and unhappy customers
LEVERS = [
    # revenue is here even though a marketplace cant simply "decide" to have
    # more of it. leaving it out made the simulator look broken: the other
    # levers are all customer side, and those are only about a third of the
    # model's SHAP importance, so nothing visibly moved. including it shows
    # honestly what the model actually reacts to - revenue momentum.
    # "relative" means the slider range is built from this seller's own
    # revenue rather than a fixed scale, since sellers differ by orders of
    # magnitude
    {"id": "revenue", "label": "This month's revenue (R$)",
     "relative": True, "lo_mult": 0.3, "hi_mult": 2.0, "steps": 60,
     "fmt": "{:,.0f}",
     "help": "the model leans on revenue momentum more than anything else"},
    {"id": "avg_review_score", "label": "Average review score",
     "min": 1.0, "max": 5.0, "step": 0.1, "fmt": "{:.1f}",
     "help": "1 = everyone unhappy, 5 = everyone happy"},
    {"id": "pct_negative_reviews", "label": "Share of negative reviews",
     "min": 0.0, "max": 1.0, "step": 0.02, "fmt": "{:.0%}",
     "help": "reviews scoring 1 or 2"},
    {"id": "pct_late_deliveries", "label": "Share of late deliveries",
     "min": 0.0, "max": 1.0, "step": 0.02, "fmt": "{:.0%}",
     "help": "arrived after the estimated date"},
    {"id": "avg_delivery_delay", "label": "Average delivery delay (days)",
     "min": -40.0, "max": 40.0, "step": 1.0, "fmt": "{:+.0f}",
     "help": "negative = early, positive = late"},
    {"id": "rfm_recency", "label": "Customer recency (days)",
     "min": 1.0, "max": 120.0, "step": 1.0, "fmt": "{:.0f}",
     "help": "average days since a customer last bought"},
]

CONTROL_LABEL = {"fontSize": "11px", "color": TEXT_FAINT,
                 "textTransform": "uppercase", "letterSpacing": "0.07em"}
EMPTY_NOTE = {"color": TEXT_FAINT, "fontSize": "12.5px",
              "padding": "30px 0", "textAlign": "center"}


def load_model():
    return (joblib.load(MODELS_DIR / "seller_best_model.pkl"),
            joblib.load(MODELS_DIR / "seller_feature_columns.pkl"))


def apply_change(row, feature, new_value):
    """set a level feature and move its derived features with it.

    see the module docstring for where the /3 and /2 come from
    """
    if feature not in row:
        return row
    delta = new_value - row[feature]
    if delta == 0:
        return row

    row[feature] = new_value

    d_col = f"d_{feature}"
    prev = None
    if d_col in row:
        prev = row[feature] - delta - row[d_col]   # value last month
        row[d_col] = row[d_col] + delta
    if f"roll3_{feature}" in row:
        row[f"roll3_{feature}"] += delta / 3.0
    if f"trend_{feature}" in row:
        row[f"trend_{feature}"] += delta / 2.0
    if f"pct_{feature}" in row and prev not in (None, 0):
        row[f"pct_{feature}"] = (row[feature] - prev) / prev

    return row


def predict_row(model, feature_cols, row):
    X = pd.DataFrame([[row.get(c, 0.0) for c in feature_cols]], columns=feature_cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return float(model.predict_proba(X)[0, 1])


def gauge(original, simulated):
    """two markers on one track - where the seller was, where they'd be."""
    better = simulated < original
    move_colour = EMERALD if better else ROSE

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], mode="lines",
        line=dict(color=LINE, width=10), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[original], y=[0], mode="markers+text",
        marker=dict(size=15, color=TEXT_DIM, symbol="line-ns",
                    line=dict(width=3, color=TEXT_DIM)),
        text=[f"now {original:.3f}"], textposition="top center",
        textfont=dict(size=11, color=TEXT_DIM, family=MONO), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[simulated], y=[0], mode="markers+text",
        marker=dict(size=17, color=move_colour),
        text=[f"{simulated:.3f}"], textposition="bottom center",
        textfont=dict(size=13, color=move_colour, family=MONO), hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=30), height=110, showlegend=False,
        font=dict(family=SANS),
    )
    fig.update_xaxes(range=[-0.03, 1.03], showgrid=False, zeroline=False,
                     tickformat=".1f", tickfont=dict(color=TEXT_FAINT, size=10),
                     linecolor=LINE_SOFT)
    fig.update_yaxes(range=[-1, 1], showgrid=False, zeroline=False,
                     showticklabels=False)
    return fig


def layout():
    try:
        df = load_seller_risk_table()
        load_model()
    except Exception:
        return html.Div([
            html.H1("Simulator"),
            html.P("Run src/seller_features.py then src/seller_model.py first.",
                   className="page-sub"),
        ])

    if "risk_rank" not in df.columns:
        df = df.copy()
        df["risk_rank"] = df["risk_score"].rank(ascending=False, method="first").astype(int)

    ranked = df.nsmallest(25, "risk_rank")
    options = [
        {"label": f"#{int(r.risk_rank)}  ·  {r.seller_id[:12]}  ·  risk {r.risk_score:.3f}",
         "value": r.seller_id}
        for r in ranked.itertuples()
    ]

    return html.Div([
        html.Div([
            html.Div([
                html.H1("What-if simulator"),
                html.P("Take a seller off the worklist and change what a "
                       "marketplace could actually influence. The model re-scores "
                       "them live.", className="page-sub"),
            ], className="page-head-text"),
        ], className="page-head"),

        html.Div([
            html.Div([
                html.Span("Seller", style=CONTROL_LABEL),
                html.Div(
                    dcc.Dropdown(id="sim-seller", options=options,
                                 value=options[0]["value"] if options else None,
                                 clearable=False, className="dash-dropdown"),
                    style={"minWidth": "360px", "position": "relative", "zIndex": 100},
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "12px"}),
        ], style={"marginBottom": "16px"}),

        dcc.Store(id="sim-baseline"),

        html.Div([
            html.Div([
                html.P("Levers", className="card-title"),
                html.P("Drag to change this seller's numbers", className="card-sub"),
                html.Div(id="sim-sliders"),
                html.Button("Reset to actual", id="sim-reset", n_clicks=0,
                            style={
                                "marginTop": "18px", "padding": "8px 16px",
                                "background": CARD_2, "color": TEXT_DIM,
                                "border": f"1px solid {LINE_SOFT}",
                                "borderRadius": "20px", "fontSize": "12.5px",
                                "fontFamily": SANS, "cursor": "pointer",
                            }),
            ], className="card"),

            html.Div([
                html.P("Predicted risk", className="card-title"),
                html.P("Grey marker is the seller as they actually are",
                       className="card-sub"),
                html.Div(id="sim-result"),
            ], className="card"),
        ], className="duo-row"),
    ])


@callback(
    Output("sim-sliders", "children"),
    Output("sim-baseline", "data"),
    Input("sim-seller", "value"),
    Input("sim-reset", "n_clicks"),
)
def build_sliders(seller_id, _reset):
    if not seller_id:
        return html.P("Pick a seller.", style=EMPTY_NOTE), None

    df = load_seller_risk_table()
    match = df[df["seller_id"] == seller_id]
    if match.empty:
        return html.P("Seller not found.", style=EMPTY_NOTE), None

    row = match.iloc[0].to_dict()
    row = {k: v for k, v in row.items() if isinstance(v, (int, float, np.number))}

    blocks = []
    for lev in LEVERS:
        if lev["id"] not in row:
            continue
        actual = float(row[lev["id"]])
        if lev.get("relative"):
            lo = actual * lev["lo_mult"]
            hi = max(actual * lev["hi_mult"], lo + 1)
            step = (hi - lo) / lev["steps"]
        else:
            lo, hi, step = lev["min"], lev["max"], lev["step"]
        actual = min(max(actual, lo), hi)

        blocks.append(html.Div([
            html.Div([
                html.Span(lev["label"], style={"fontSize": "12.5px", "color": TEXT}),
                html.Span(lev["fmt"].format(actual),
                          id=f"sim-val-{lev['id']}",
                          style={"marginLeft": "auto", "fontFamily": MONO,
                                 "fontSize": "12.5px", "color": VIOLET}),
            ], style={"display": "flex", "alignItems": "baseline",
                      "marginBottom": "6px"}),
            dcc.Slider(
                id={"type": "sim-slider", "feature": lev["id"]},
                min=lo, max=hi, step=step, value=actual,
                marks=None, tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            ),
            html.Div(lev["help"], style={"fontSize": "11px", "color": TEXT_FAINT,
                                         "marginTop": "2px"}),
        ], style={"marginBottom": "22px"}))

    return blocks, {"seller_id": seller_id, "row": row}


@callback(
    Output("sim-result", "children"),
    Input({"type": "sim-slider", "feature": dash.ALL}, "value"),
    Input({"type": "sim-slider", "feature": dash.ALL}, "id"),
    Input("sim-baseline", "data"),
)
def simulate(values, ids, baseline):
    if not baseline or not values:
        return html.P("Pick a seller to begin.", style=EMPTY_NOTE)

    model, feature_cols = load_model()
    row = dict(baseline["row"])
    original = predict_row(model, feature_cols, row)

    changed = []
    for val, ident in zip(values, ids):
        feat = ident["feature"]
        if val is None or feat not in row:
            continue
        if abs(val - row[feat]) > 1e-9:
            lev = next(l for l in LEVERS if l["id"] == feat)
            changed.append((lev["label"], row[feat], val, lev["fmt"]))
            row = apply_change(row, feat, float(val))

    simulated = predict_row(model, feature_cols, row)
    delta = simulated - original

    if delta < -0.001:
        verdict, colour = "lower risk", EMERALD
    elif delta > 0.001:
        verdict, colour = "higher risk", ROSE
    else:
        verdict, colour = "no change", TEXT_DIM

    change_rows = [
        html.Div([
            html.Span(label, style={"color": TEXT_DIM, "fontSize": "12px"}),
            html.Span([
                html.Span(fmt.format(old), style={"color": TEXT_FAINT}),
                html.Span("  →  ", style={"color": TEXT_FAINT}),
                html.Span(fmt.format(new), style={"color": TEXT}),
            ], style={"marginLeft": "auto", "fontFamily": MONO, "fontSize": "12px"}),
        ], style={"display": "flex", "marginBottom": "7px"})
        for label, old, new, fmt in changed
    ]

    return html.Div([
        dcc.Graph(figure=gauge(original, simulated), config={"displayModeBar": False}),

        html.Div([
            html.Span(f"{delta:+.3f}", style={
                "fontFamily": MONO, "fontSize": "26px", "color": colour,
                "fontWeight": "600"}),
            html.Span(verdict, style={"marginLeft": "10px", "color": TEXT_DIM,
                                      "fontSize": "13px"}),
        ], style={"marginBottom": "18px"}),

        html.Div(change_rows if change_rows else
                 html.P("Move a slider to see the effect.",
                        style={"color": TEXT_FAINT, "fontSize": "12px"}),
                 style={"borderTop": f"1px solid {LINE_SOFT}", "paddingTop": "14px"}),

        html.P("Changing a level also moves its month-on-month change, rolling "
               "average and trend, so the row stays internally consistent — the "
               "model was trained on consistent rows. This shows the model's "
               "sensitivity, not a guarantee of what would happen.",
               className="chart-note"),
    ])