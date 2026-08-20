"""
simulator.py - what-if on a real seller.

Pick a seller off the worklist, change what a marketplace could plausibly
influence, and the model re-scores them.

PERFORMANCE
The first version reloaded the model pickle from disk on every callback,
and the sliders were set to updatemode="drag" so that fired on every
pixel of movement. A 400 tree forest is not small - that was the lag.
Now the model is cached with lru_cache (loaded once per process) and the
sliders update on release instead, with the tooltip giving feedback
while you drag.

WHY IT STARTS FROM A REAL SELLER
The model needs all 49 features. If the sliders cover 6 and the rest are
filled with medians, youre not simulating a seller, youre simulating an
average one with a few numbers bolted on. Starting from a real row keeps
everything else true to that seller.

PROPAGATING A CHANGE (the bit that matters)
Every level feature has momentum features derived from it in
seller_features.py, and they have to move together or the row
contradicts itself - review score 4.5 while d_avg_review_score still
says the seller dropped half a star. the model was trained on consistent
rows, so an inconsistent one gives a meaningless answer.

for a level change of D on feature x:
    x       += D
    d_x     += D      (d_x = x - x_prev, and x_prev doesnt move)
    roll3_x += D/3    (3 month mean, only this month changed)
    trend_x += D/2    (slope over 3 points is (c-a)/2, so c+D moves it D/2)
    pct_x   recomputed from x_prev, recovered as x - d_x

run after seller_model.py
"""

import sys
from functools import lru_cache
from pathlib import Path

import dash
from dash import html, dcc, callback, Input, Output, State, ctx, ALL
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

LEVERS = [
    # revenue is in here even though a marketplace cant simply decide to
    # have more of it. without it the page looked broken - the other levers
    # are all customer side and those are only about a third of the model's
    # SHAP importance, so barely anything moved. including it shows honestly
    # what the model actually reacts to.
    # "relative" = build the slider range from this seller's own revenue,
    # since sellers differ by orders of magnitude
    {"id": "revenue", "label": "Revenue this month",
     "relative": True, "lo_mult": 0.3, "hi_mult": 2.0, "steps": 60,
     "fmt": "R$ {:,.0f}", "unit": "",
     "help": "the model leans on revenue momentum more than anything else"},
    {"id": "avg_review_score", "label": "Average review score",
     "min": 1.0, "max": 5.0, "step": 0.1, "fmt": "{:.1f}", "unit": " / 5",
     "help": "1 = everyone unhappy, 5 = everyone happy"},
    {"id": "pct_negative_reviews", "label": "Negative reviews",
     "min": 0.0, "max": 1.0, "step": 0.02, "fmt": "{:.0%}", "unit": "",
     "help": "share of reviews scoring 1 or 2"},
    {"id": "pct_late_deliveries", "label": "Late deliveries",
     "min": 0.0, "max": 1.0, "step": 0.02, "fmt": "{:.0%}", "unit": "",
     "help": "share arriving after the estimated date"},
    {"id": "avg_delivery_delay", "label": "Average delivery delay",
     "min": -40.0, "max": 40.0, "step": 1.0, "fmt": "{:+.0f}", "unit": " days",
     "help": "negative = early, positive = late"},
    {"id": "rfm_recency", "label": "Customer recency",
     "min": 1.0, "max": 120.0, "step": 1.0, "fmt": "{:.0f}", "unit": " days",
     "help": "average days since a customer last bought"},
]

# one-click stories, so you dont have to work out what a sensible
# combination of sliders even looks like
PRESETS = {
    "sim-preset-service": {
        "label": "Fix the service",
        "desc": "late deliveries down to 5%, reviews recover to 4.5",
        "set": {"pct_late_deliveries": 0.05, "avg_delivery_delay": -12.0,
                "avg_review_score": 4.5, "pct_negative_reviews": 0.05},
    },
    "sim-preset-slump": {
        "label": "Service collapses",
        "desc": "half the deliveries late, reviews fall to 2.0",
        "set": {"pct_late_deliveries": 0.50, "avg_delivery_delay": 15.0,
                "avg_review_score": 2.0, "pct_negative_reviews": 0.45},
    },
    "sim-preset-quiet": {
        "label": "Revenue stalls",
        "desc": "this month's revenue drops 40%",
        "set": {"__revenue_mult__": 0.6},
    },
}

BTN = {"padding": "8px 14px", "background": CARD_2, "color": TEXT_DIM,
       "border": f"1px solid {LINE_SOFT}", "borderRadius": "20px",
       "fontSize": "12.5px", "fontFamily": SANS, "cursor": "pointer",
       "marginRight": "8px", "marginBottom": "8px"}
CONTROL_LABEL = {"fontSize": "11px", "color": TEXT_FAINT,
                 "textTransform": "uppercase", "letterSpacing": "0.07em"}
EMPTY_NOTE = {"color": TEXT_FAINT, "fontSize": "12.5px",
              "padding": "30px 0", "textAlign": "center"}
STEP_STYLE = {"display": "flex", "gap": "9px", "alignItems": "baseline",
              "fontSize": "12.5px", "color": TEXT_DIM, "marginBottom": "5px"}


@lru_cache(maxsize=1)
def load_model():
    # cached - reloading a 400 tree forest on every slider move was the lag
    return (joblib.load(MODELS_DIR / "seller_best_model.pkl"),
            joblib.load(MODELS_DIR / "seller_feature_columns.pkl"))


def slider_range(lev, actual):
    """range for one slider, always wide enough to contain the real value.

    the fixed min/max below are sensible defaults, not hard limits - real
    sellers go outside them (avg_delivery_delay runs from about -147 to
    +188 in the olist data, well past the +/-40 default). if the range
    clamped the actual value, the slider would open at a different number
    than the seller really has, and the page would show a change nobody
    made. so widen to fit instead.
    """
    if lev.get("relative"):
        lo = actual * lev["lo_mult"]
        hi = max(actual * lev["hi_mult"], lo + 1)
        # round the step to something human - (hi-lo)/60 on a revenue of
        # 3137 gives a step of 3430.224 and the tooltip reads like a
        # rounding error. snap to 1/5/10/50/100/... instead
        raw = (hi - lo) / lev["steps"]
        mag = 10 ** int(np.floor(np.log10(raw))) if raw > 0 else 1
        step = next((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), raw)
        return lo, hi, step

    lo, hi, step = lev["min"], lev["max"], lev["step"]
    if actual < lo:
        lo = actual - abs(actual) * 0.1 - step
    if actual > hi:
        hi = actual + abs(actual) * 0.1 + step
    return lo, hi, step


def fmt_value(lev, v):
    return lev["fmt"].format(v) + lev.get("unit", "")


def apply_change(row, feature, new_value):
    """set a level feature and move its derived features with it."""
    if feature not in row:
        return row
    delta = new_value - row[feature]
    if delta == 0:
        return row

    row[feature] = new_value
    d_col = f"d_{feature}"
    prev = None
    if d_col in row:
        prev = row[feature] - delta - row[d_col]
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
    better = simulated < original
    colour = EMERALD if better else (ROSE if simulated > original else TEXT_DIM)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode="lines",
                             line=dict(color=LINE, width=10), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[original], y=[0], mode="markers+text",
        marker=dict(size=13, color=TEXT_FAINT, symbol="line-ns",
                    line=dict(width=3, color=TEXT_FAINT)),
        text=[f"actual {original:.3f}"], textposition="top center",
        textfont=dict(size=10.5, color=TEXT_FAINT, family=MONO), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[simulated], y=[0], mode="markers+text",
        marker=dict(size=17, color=colour),
        text=[f"{simulated:.3f}"], textposition="bottom center",
        textfont=dict(size=14, color=colour, family=MONO), hoverinfo="skip"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=20, r=20, t=28, b=28), height=115,
                      showlegend=False, font=dict(family=SANS))
    fig.update_xaxes(range=[-0.03, 1.03], showgrid=False, zeroline=False,
                     tickformat=".1f", tickfont=dict(color=TEXT_FAINT, size=10),
                     linecolor=LINE_SOFT, title=None)
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
    options = [{"label": f"#{int(r.risk_rank)}  ·  {r.seller_id[:12]}  ·  risk {r.risk_score:.3f}",
                "value": r.seller_id} for r in ranked.itertuples()]

    return html.Div([
        html.Div([
            html.Div([
                html.H1("What-if simulator"),
                html.P("Take a seller off the worklist, change something, and the "
                       "model re-scores them.", className="page-sub"),
            ], className="page-head-text"),
        ], className="page-head"),

        html.Div([
            html.Div([html.Span("1", style={"color": VIOLET, "fontFamily": MONO}),
                      "Pick a seller below — they're ordered by risk."], style=STEP_STYLE),
            html.Div([html.Span("2", style={"color": VIOLET, "fontFamily": MONO}),
                      "Click a scenario, or drag any slider."], style=STEP_STYLE),
            html.Div([html.Span("3", style={"color": VIOLET, "fontFamily": MONO}),
                      "The risk on the right updates when you let go."],
                     style={**STEP_STYLE, "marginBottom": "0"}),
        ], className="card"),

        html.Div([
            html.Span("Select High-Risk Seller (Top 25 Worklist)", style=CONTROL_LABEL),
            html.Div(dcc.Dropdown(id="sim-seller", options=options,
                                  value=options[0]["value"] if options else None,
                                  clearable=False, className="dash-dropdown"),
                     style={"minWidth": "380px", "position": "relative", "zIndex": 100}),
        ], style={"display": "flex", "alignItems": "center", "gap": "12px",
                  "marginBottom": "16px"}),

        dcc.Store(id="sim-baseline"),

        html.Div([
            html.Div([
                html.P("Scenarios", className="card-title"),
                html.P("One click sets several sliders at once", className="card-sub"),
                html.Div([
                    html.Button([
                        html.Div(p["label"], style={"color": TEXT, "fontSize": "12.5px"}),
                        html.Div(p["desc"], style={"color": TEXT_FAINT, "fontSize": "11px",
                                                   "marginTop": "3px"}),
                    ], id=bid, n_clicks=0,
                        style={**BTN, "textAlign": "left", "padding": "10px 14px"})
                    for bid, p in PRESETS.items()
                ] + [
                    html.Button("Reset to actual", id="sim-reset", n_clicks=0,
                                style={**BTN, "color": VIOLET}),
                ], style={"display": "flex", "flexWrap": "wrap"}),

                html.Div(style={"height": "1px", "background": LINE_SOFT,
                                "margin": "18px 0"}),

                html.P("Levers", className="card-title"),
                html.P("Or set them yourself", className="card-sub"),
                html.Div(id="sim-sliders"),
            ], className="card"),

            html.Div([
                html.P("Predicted risk", className="card-title"),
                html.P("Grey tick is the seller as they actually are",
                       className="card-sub"),
                html.Div(id="sim-result"),
            ], className="card"),
        ], className="duo-row"),
    ])


@callback(Output("sim-baseline", "data"), Input("sim-seller", "value"))
def set_baseline(seller_id):
    if not seller_id:
        return None
    df = load_seller_risk_table()
    match = df[df["seller_id"] == seller_id]
    if match.empty:
        return None
    row = {k: float(v) for k, v in match.iloc[0].to_dict().items()
           if isinstance(v, (int, float, np.number))}
    return {"seller_id": seller_id, "row": row}


@callback(Output("sim-sliders", "children"), Input("sim-baseline", "data"))
def build_sliders(baseline):
    if not baseline:
        return html.P("Pick a seller.", style=EMPTY_NOTE)

    row = baseline["row"]
    blocks = []
    for lev in LEVERS:
        if lev["id"] not in row:
            continue
        actual = float(row[lev["id"]])
        lo, hi, step = slider_range(lev, actual)

        blocks.append(html.Div([
            html.Div([
                html.Span(lev["label"], style={"fontSize": "12.5px", "color": TEXT}),
                html.Span(fmt_value(lev, actual),
                          id={"type": "sim-val", "feature": lev["id"]},
                          style={"marginLeft": "auto", "fontFamily": MONO,
                                 "fontSize": "12.5px", "color": VIOLET}),
            ], style={"display": "flex", "alignItems": "baseline",
                      "marginBottom": "4px"}),
             # no tooltip. dash 4.x renders it with
            # background-color: var(--Dash-Fill-Inverse-Strong) = white, and
            # sets no text colour so the text inherits our light body colour -
            # an unreadable white box. tried styling .dash-slider-tooltip and
            # it didnt take, so dropping it: the violet value beside each
            # slider label already shows the number, so the tooltip was
            # duplicate information anyway
            dcc.Slider(id={"type": "sim-slider", "feature": lev["id"]},
                       min=lo, max=hi, step=step, value=actual, marks=None,
                       allow_direct_input=False,
                       tooltip={"placement": "bottom", "always_visible": False}),
            html.Div(lev["help"], style={"fontSize": "11px", "color": TEXT_FAINT,
                                         "marginTop": "0px"}),
        ], style={"marginBottom": "20px"}))
    return blocks


@callback(
    Output({"type": "sim-slider", "feature": ALL}, "value"),
    [Input(bid, "n_clicks") for bid in PRESETS] + [Input("sim-reset", "n_clicks")],
    State({"type": "sim-slider", "feature": ALL}, "id"),
    State("sim-baseline", "data"),
    prevent_initial_call=True,
)
def apply_preset(*args):
    ids, baseline = args[-2], args[-1]
    if not baseline or not ids:
        raise dash.exceptions.PreventUpdate

    row = baseline["row"]
    which = ctx.triggered_id

    if which == "sim-reset" or which not in PRESETS:
        return [float(row.get(i["feature"], 0.0)) for i in ids]

    preset = PRESETS[which]["set"]
    out = []
    for i in ids:
        f = i["feature"]
        actual = float(row.get(f, 0.0))
        if "__revenue_mult__" in preset and f == "revenue":
            out.append(actual * preset["__revenue_mult__"])
        elif f in preset:
            out.append(float(preset[f]))
        else:
            out.append(actual)
    return out


@callback(
    Output({"type": "sim-val", "feature": ALL}, "children"),
    Input({"type": "sim-slider", "feature": ALL}, "value"),
    State({"type": "sim-slider", "feature": ALL}, "id"),
)
def update_value_labels(values, ids):
    # the first version never updated these, so the number beside each
    # slider stayed on the original value no matter what you did
    out = []
    for v, i in zip(values, ids):
        lev = next((l for l in LEVERS if l["id"] == i["feature"]), None)
        out.append(fmt_value(lev, float(v)) if lev and v is not None else "")
    return out


@callback(
    Output("sim-result", "children"),
    Input({"type": "sim-slider", "feature": ALL}, "value"),
    Input("sim-baseline", "data"),
    State({"type": "sim-slider", "feature": ALL}, "id"),
)
def simulate(values, baseline, ids):
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
            changed.append((lev, row[feat], float(val)))
            row = apply_change(row, feat, float(val))

    simulated = predict_row(model, feature_cols, row)
    delta = simulated - original

    if delta < -0.001:
        verdict, colour = "less likely to decline", EMERALD
    elif delta > 0.001:
        verdict, colour = "more likely to decline", ROSE
    else:
        verdict, colour = "no change", TEXT_DIM

    rows = [
        html.Div([
            html.Span(lev["label"], style={"color": TEXT_DIM, "fontSize": "12px"}),
            html.Span([
                html.Span(fmt_value(lev, old), style={"color": TEXT_FAINT}),
                html.Span("  →  ", style={"color": TEXT_FAINT}),
                html.Span(fmt_value(lev, new), style={"color": TEXT}),
            ], style={"marginLeft": "auto", "fontFamily": MONO, "fontSize": "12px"}),
        ], style={"display": "flex", "marginBottom": "7px"})
        for lev, old, new in changed
    ]

    return html.Div([
        dcc.Graph(figure=gauge(original, simulated), config={"displayModeBar": False}),
        html.Div([
            html.Span(f"{delta:+.3f}", style={"fontFamily": MONO, "fontSize": "26px",
                                              "color": colour, "fontWeight": "600"}),
            html.Span(verdict, style={"marginLeft": "10px", "color": TEXT_DIM,
                                      "fontSize": "13px"}),
        ], style={"marginBottom": "16px"}),
        html.Div(rows if rows else html.P("Nothing changed yet — click a scenario "
                                          "or drag a slider.",
                                          style={"color": TEXT_FAINT, "fontSize": "12px"}),
                 style={"borderTop": f"1px solid {LINE_SOFT}", "paddingTop": "14px"}),
        html.P("Changing a level also moves its month-on-month change, rolling "
               "average and trend, so the row stays internally consistent — the "
               "model was trained on consistent rows. This shows the model's "
               "sensitivity, not a promise about what would happen.",
               className="chart-note"),
    ])