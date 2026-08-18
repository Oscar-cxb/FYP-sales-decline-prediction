"""
interventions.py - what to actually do about the flagged sellers.

The Seller Risk page says who is likely to decline. This page says why,
and what to do about it, because "seller #7 has a 0.72 chance of
declining" is not something anyone can act on.

RULES, NOT A SECOND MODEL
Deliberately a rule engine rather than another classifier. The rules are
readable, a marketplace could argue with them, and each one points at a
specific department - logistics, support, marketing. A second black box
predicting "recommended action" would be worse in every way: no training
labels for it, and nothing anyone could challenge.

THRESHOLDS ARE RELATIVE, NOT HARDCODED
"late deliveries above 20%" means nothing without knowing the norm. Every
threshold here is a percentile of the flagged cohort, so the rules adapt
to the data instead of to numbers i picked. the cutoffs are stated on the
page so theyre not hidden.

WHERE SENTIMENT SHOWS UP
Two of the five rules key on review data - the level and, more usefully,
the month-on-month movement. a seller whose score fell 0.6 stars is a
different problem from one that has always been mediocre. this is also
the only page where the sentiment features are visible as themselves
rather than as a bar inside a SHAP chart.

run after seller_model.py
"""

import sys
from functools import lru_cache
from pathlib import Path

import dash
from dash import html, dcc, dash_table, callback, Input, Output
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import load_seller_risk_table
from theme import (
    VIOLET, CYAN, AMBER, ROSE, EMERALD, TEXT, TEXT_DIM, TEXT_FAINT,
    LINE, LINE_SOFT, CARD, CARD_2, MONO, SANS, rgba,
)

dash.register_page(__name__, path="/interventions", name="Interventions")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DEFAULT_CAPACITY = 50

# each action carries who owns it, so the page isnt just adjectives
ACTIONS = {
    "Logistics review": {
        "colour": AMBER, "owner": "Operations",
        "what": "Late deliveries are high or climbing. Chase the courier, "
                "and widen the delivery estimate so buyers stop being "
                "disappointed by a date that was never realistic.",
    },
    "Service recovery": {
        "colour": ROSE, "owner": "Customer support",
        "what": "Review scores are falling. Contact the recent 1-2 star "
                "buyers, find out what went wrong, and check whether it is "
                "the product or the packaging.",
    },
    "Re-engagement campaign": {
        "colour": CYAN, "owner": "Marketing",
        "what": "This seller's buyers are drifting - the gap since their "
                "last order is widening. Worth a targeted offer to that "
                "seller's customer list.",
    },
    "Account call": {
        "colour": VIOLET, "owner": "Account management",
        "what": "High revenue and high risk. Whatever the cause, this one "
                "is worth a person picking up the phone this week.",
    },
    "Monitor only": {
        "colour": TEXT_DIM, "owner": "No action",
        "what": "Flagged mainly because revenue jumped and tends to fall "
                "back afterwards. Service looks fine, so intervening here "
                "would be spending effort on a statistical artefact.",
    },
}

ACTION_ORDER = ["Account call", "Logistics review", "Service recovery",
                "Re-engagement campaign", "Monitor only"]

CONTROL_LABEL = {"fontSize": "11px", "color": TEXT_FAINT,
                 "textTransform": "uppercase", "letterSpacing": "0.07em"}
LABEL_STYLE = {"display": "inline-flex", "alignItems": "center", "gap": "7px",
               "marginRight": "8px", "padding": "6px 12px",
               "border": f"1px solid {LINE_SOFT}", "borderRadius": "20px",
               "background": CARD_2, "color": TEXT_DIM,
               "fontSize": "12.5px", "cursor": "pointer"}
INPUT_STYLE = {"accentColor": VIOLET, "width": "13px", "height": "13px",
               "margin": "0", "cursor": "pointer"}
SWATCH = {"width": "8px", "height": "8px", "borderRadius": "50%",
          "display": "inline-block"}
EMPTY_NOTE = {"color": TEXT_FAINT, "fontSize": "12.5px",
              "padding": "30px 0", "textAlign": "center"}


@lru_cache(maxsize=1)
def _sellers():
    df = load_seller_risk_table()
    if "risk_rank" not in df.columns:
        df = df.copy()
        df["risk_rank"] = df["risk_score"].rank(ascending=False,
                                                method="first").astype(int)
    return df


def get_capacity():
    try:
        return int(joblib.load(MODELS_DIR / "seller_model_meta.pkl")
                   .get("capacity", DEFAULT_CAPACITY))
    except Exception:
        return DEFAULT_CAPACITY


def col(df, name, default=0.0):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def build_rules(work, cohort):
    """assign one action per seller, plus the evidence behind it.

    first matching rule wins, in ACTION_ORDER priority. one action each on
    purpose - a list where every row says "do four things" is a list
    nobody works through.
    """
    late = col(work, "pct_late_deliveries")
    late_up = col(work, "d_pct_late_deliveries")
    review = col(work, "avg_review_score", 5.0)
    review_dn = col(work, "d_avg_review_score")
    neg = col(work, "pct_negative_reviews")
    recency_up = col(work, "d_rfm_recency")
    rev = col(work, "revenue")
    rev_up = col(work, "d_revenue")

    # percentiles of the cohort, so the thresholds mean something
    t = {
        "late_hi": float(np.nanpercentile(col(cohort, "pct_late_deliveries"), 75)),
        "late_rising": float(np.nanpercentile(col(cohort, "d_pct_late_deliveries"), 75)),
        "review_lo": float(np.nanpercentile(col(cohort, "avg_review_score", 5.0), 25)),
        "review_falling": float(np.nanpercentile(col(cohort, "d_avg_review_score"), 20)),
        "neg_hi": float(np.nanpercentile(col(cohort, "pct_negative_reviews"), 75)),
        "recency_rising": float(np.nanpercentile(col(cohort, "d_rfm_recency"), 75)),
        "rev_hi": float(np.nanpercentile(col(cohort, "revenue"), 80)),
        "rev_jump": float(np.nanpercentile(col(cohort, "d_revenue"), 75)),
    }

    delivery_bad = (late >= t["late_hi"]) | (late_up >= t["late_rising"])
    reviews_bad = ((review_dn <= t["review_falling"]) | (review <= t["review_lo"])
                   | (neg >= t["neg_hi"]))
    drifting = recency_up >= t["recency_rising"]
    big = rev >= t["rev_hi"]
    just_spiked = rev_up >= t["rev_jump"]

    action = pd.Series("Monitor only", index=work.index)
    reason = pd.Series("no service problem stands out", index=work.index)

    # lowest priority first, so higher ones overwrite
    m = just_spiked & ~delivery_bad & ~reviews_bad
    action[m], reason[m] = "Monitor only", "revenue jumped, service looks fine"

    m = drifting & ~delivery_bad & ~reviews_bad
    action[m], reason[m] = "Re-engagement campaign", "buyers taking longer to return"

    m = reviews_bad
    action[m], reason[m] = "Service recovery", "review scores falling or already low"

    m = delivery_bad
    action[m], reason[m] = "Logistics review", "late deliveries high or climbing"

    m = big & (delivery_bad | reviews_bad | drifting)
    action[m], reason[m] = "Account call", "large seller with a service problem"

    out = work.copy()
    out["action"] = action
    out["reason"] = reason
    return out, t


def build_action_chart(work):
    counts = (work.groupby("action")
              .agg(sellers=("seller_id", "count"), revenue=("revenue", "sum"))
              .reindex([a for a in ACTION_ORDER if a in set(work["action"])])
              .reset_index())
    counts = counts[counts["sellers"] > 0].sort_values("sellers")

    fig = go.Figure(go.Bar(
        x=counts["sellers"], y=counts["action"], orientation="h",
        marker=dict(color=[ACTIONS[a]["colour"] for a in counts["action"]],
                    line=dict(width=0)),
        width=0.6, customdata=counts["revenue"],
        text=[f"{v}" for v in counts["sellers"]], textposition="outside",
        textfont=dict(size=11, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>%{x} sellers<br>R$ %{customdata:,.0f} revenue<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, size=11, family=SANS),
        margin=dict(l=6, r=44, t=6, b=26), height=250, showlegend=False,
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE,
                        font=dict(color=TEXT, size=11.5, family=SANS)))
    fig.update_xaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT,
                     range=[0, counts["sellers"].max() * 1.25], title=None)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", zeroline=False, linecolor=LINE_SOFT)
    return fig


def build_sentiment_chart(work):
    """month-on-month change in review score across the worklist.

    the sentiment features are otherwise invisible on this dashboard - a
    6.7% bar inside a SHAP chart. this shows the thing itself: who is
    getting worse, who is recovering.
    """
    if "d_avg_review_score" not in work.columns:
        return None

    d = work["d_avg_review_score"].dropna()
    if d.empty:
        return None

    fig = go.Figure()
    worse = d[d < 0]
    better = d[d >= 0]
    for series, colour, name in [(worse, ROSE, "fell"), (better, EMERALD, "held or rose")]:
        if len(series):
            fig.add_trace(go.Histogram(
                x=series, nbinsx=26,
                marker=dict(color=rgba(colour, 0.55),
                            line=dict(color=colour, width=1)),
                name=name,
                hovertemplate=name + ": %{y} sellers<br>change %{x:.2f}<extra></extra>"))

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=TEXT_FAINT)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, size=11, family=SANS), barmode="overlay",
        margin=dict(l=6, r=16, t=6, b=34), height=250, showlegend=False,
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE,
                        font=dict(color=TEXT, size=11.5, family=SANS)))
    fig.update_xaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT,
                     title="Change in average review score vs last month")
    fig.update_yaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT,
                     title=None)
    return fig


def build_table(work):
    # paginated rather than head(15) - the whole point of a worklist is
    # being able to work down all of it
    t = work.sort_values("risk_rank").copy()
    t["rev_f"] = t["revenue"].map(lambda v: f"{v:,.0f}")
    t["review_f"] = col(t, "avg_review_score", np.nan).map(
        lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    t["review_d_f"] = col(t, "d_avg_review_score", np.nan).map(
        lambda v: "—" if pd.isna(v) else f"{v:+.2f}")
    t["late_f"] = col(t, "pct_late_deliveries", np.nan).map(
        lambda v: "—" if pd.isna(v) else f"{v*100:.0f}%")
    t["risk_f"] = t["risk_score"].map(lambda v: f"{v:.3f}")
    t["seller_f"] = t["seller_id"].str[:12]
    t["owner"] = t["action"].map(lambda a: ACTIONS[a]["owner"])

    cols = [("risk_rank", "#"), ("seller_f", "Seller"), ("rev_f", "Revenue (R$)"),
            ("review_f", "Reviews"), ("review_d_f", "Δ reviews"),
            ("late_f", "Late"), ("risk_f", "Risk"),
            ("action", "Action"), ("owner", "Owner"), ("reason", "Why")]
    numeric = ["risk_rank", "rev_f", "review_f", "review_d_f", "late_f", "risk_f"]

    return dash_table.DataTable(
        data=t[[c for c, _ in cols]].to_dict("records"),
        columns=[{"name": lbl, "id": cid} for cid, lbl in cols],
        sort_action="native", page_size=15, style_as_list_view=True,
        style_header={"backgroundColor": "rgba(0,0,0,0)", "color": TEXT_FAINT,
                      "fontWeight": "500", "fontSize": "11px", "border": "none",
                      "borderBottom": f"1px solid {LINE}",
                      "padding": "0 12px 11px 0", "textAlign": "left"},
        style_cell={"backgroundColor": "rgba(0,0,0,0)", "color": TEXT,
                    "border": "none", "padding": "11px 12px 11px 0",
                    "fontSize": "12.5px", "fontFamily": SANS, "textAlign": "left"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "right"}
                                for c in numeric],
        style_data_conditional=[
            {"if": {"column_id": "seller_f"}, "fontFamily": MONO,
             "color": TEXT_DIM, "fontSize": "11.5px"},
            {"if": {"column_id": numeric}, "fontFamily": MONO, "fontSize": "12px"},
            {"if": {"column_id": ["owner", "reason"]}, "color": TEXT_FAINT,
             "fontSize": "11.5px"},
        ] + [
            {"if": {"filter_query": f"{{action}} = '{a}'", "column_id": "action"},
             "color": cfg["colour"], "fontWeight": "600"}
            for a, cfg in ACTIONS.items()
        ] + [
            {"if": {"filter_query": "{review_d_f} contains '-'",
                    "column_id": "review_d_f"}, "color": ROSE},
        ],
    )


def action_cards(work):
    present = [a for a in ACTION_ORDER if a in set(work["action"])]
    out = []
    for a in present:
        n = int((work["action"] == a).sum())
        rev = float(work.loc[work["action"] == a, "revenue"].sum())
        cfg = ACTIONS[a]
        out.append(html.Div([
            html.Div([
                html.Span(style={**SWATCH, "background": cfg["colour"]}),
                html.Span(a, style={"marginLeft": "8px", "color": TEXT,
                                    "fontSize": "13px", "fontWeight": "600"}),
                html.Span(cfg["owner"], style={"marginLeft": "auto",
                                               "color": TEXT_FAINT,
                                               "fontSize": "11px"}),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "8px"}),
            html.Div(cfg["what"], style={"color": TEXT_DIM, "fontSize": "12px",
                                         "lineHeight": "1.55"}),
            html.Div([
                html.Span(f"{n} sellers", style={"fontFamily": MONO,
                                                 "color": cfg["colour"],
                                                 "fontSize": "12px"}),
                html.Span(f"R$ {rev:,.0f}", style={"marginLeft": "auto",
                                                   "fontFamily": MONO,
                                                   "color": TEXT_DIM,
                                                   "fontSize": "12px"}),
            ], style={"display": "flex", "marginTop": "10px",
                      "paddingTop": "10px",
                      "borderTop": f"1px solid {LINE_SOFT}"}),
        ], style={"background": CARD_2, "border": f"1px solid {LINE_SOFT}",
                  "borderRadius": "10px", "padding": "14px 16px"}))
    return out


def layout():
    try:
        df = _sellers()
    except FileNotFoundError:
        return html.Div([
            html.H1("Interventions"),
            html.P("Run src/seller_features.py then src/seller_model.py first.",
                   className="page-sub"),
        ])

    default_capacity = min(get_capacity(), len(df))
    capacity = default_capacity
    work = df.nsmallest(capacity, "risk_rank")
    work, thresholds = build_rules(work, df)

    month = str(df["month"].iloc[0]) if len(df) else "n/a"

    # worklist sizes worth offering, plus everyone scored this month
    size_options = [n for n in (10, 25, 50, 100) if n <= len(df)]
    if len(df) not in size_options:
        size_options.append(len(df))
    actionable = int((work["action"] != "Monitor only").sum())
    rev_actionable = float(work.loc[work["action"] != "Monitor only", "revenue"].sum())
    top_action = work.loc[work["action"] != "Monitor only", "action"]
    top_action = top_action.mode().iloc[0] if len(top_action) else "—"

    if "d_avg_review_score" in work.columns:
        falling = int((work["d_avg_review_score"] < 0).sum())
        falling_sub = f"of {len(work)} on the worklist"
    else:
        falling, falling_sub = 0, "no review data"

    # built once - it was being constructed twice, once for the None check
    # and again for the figure itself
    sentiment_fig = build_sentiment_chart(work)

    return html.Div([
        html.Div([
            html.Div([
                html.H1("Interventions"),
                html.P(f"The top {capacity} sellers by risk in {month}, each with "
                       f"one recommended action. Rules, not a second model — every "
                       f"threshold is a percentile of the cohort, so they adapt to "
                       f"the data rather than to numbers picked by hand.",
                       className="page-sub"),
            ], className="page-head-text"),

            html.Div([
                html.Div([html.Span("Method", className="model-stat-label"),
                                          html.Span("Rule Based", className="model-stat-value")],
                                         className="model-stat"),
                html.Div([html.Span("Actionable", className="model-stat-label"),
                          html.Span(f"{actionable}",
                                    className="model-stat-value accent")],
                         className="model-stat"),
                html.Div([html.Span("Worklist", className="model-stat-label"),
                                          html.Span(f"Top {capacity}", className="model-stat-value")],
                                         className="model-stat"),
            ], className="model-strip"),
        ], className="page-head"),

        html.Div(id="int-kpis"),

        html.Div(id="int-charts"),

        html.Div(id="int-cards"),

        html.Div([
            html.P("The worklist", className="card-title"),
            html.P("In rank order, HIGHEST RISK first", className="card-sub"),

            html.Div([
                html.Div([
                    html.Span("Worklist size", style=CONTROL_LABEL),
                    dcc.RadioItems(
                        id="int-capacity",
                        options=[{"label": (f"All {n}" if n == len(df) else f"Top {n}"),
                                  "value": n} for n in size_options],
                        value=default_capacity, inline=True,
                        labelStyle=LABEL_STYLE, inputStyle=INPUT_STYLE,
                        style={"display": "inline-flex", "flexWrap": "wrap"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "12px",
                          "flexWrap": "wrap"}),

                html.Div([
                    html.Span("Action", style=CONTROL_LABEL),
                    dcc.Checklist(
                        id="int-filter",
                        options=[{"label": html.Span([
                            html.Span(style={**SWATCH,
                                             "background": ACTIONS[a]["colour"]}),
                            html.Span(a, style={"marginLeft": "6px"}),
                        ]), "value": a} for a in ACTION_ORDER],
                        value=list(ACTION_ORDER), inline=True,
                        labelStyle=LABEL_STYLE, inputStyle=INPUT_STYLE,
                        style={"display": "inline-flex", "flexWrap": "wrap"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "12px",
                          "flexWrap": "wrap"}),
            ], style={"display": "flex", "flexDirection": "column", "gap": "12px",
                      "marginBottom": "16px"}),

            html.Div(id="int-table"),
            html.Div(id="int-thresholds"),
        ], className="card"),
    ])


# ------------------------------------------------------------ callbacks

def kpi_card(icon, colour, label, value, sub):
    return html.Div([
        html.Div([
            html.Div(icon, className="kpi-chip",
                     style={"background": rgba(colour, 0.14), "color": colour}),
            html.Div(label, className="kpi-label"),
        ], className="kpi-top"),
        html.Div(value, className="kpi-value"),
        html.Div(sub, className="kpi-sub"),
    ], className="kpi-card")


def worklist(capacity):
    """top N by risk, with the rules applied against the FULL cohort.

    thresholds are always percentiles of every scored seller, never of
    the slice being shown - otherwise changing the worklist size would
    move the goalposts, and a seller could switch action just because
    you chose to look at fewer of them
    """
    df = _sellers()
    capacity = min(int(capacity or DEFAULT_CAPACITY), len(df))
    work, th = build_rules(df.nsmallest(capacity, "risk_rank"), df)
    return work, th, df


@callback(
    Output("int-kpis", "children"),
    Output("int-charts", "children"),
    Output("int-cards", "children"),
    Output("int-thresholds", "children"),
    Input("int-capacity", "value"),
)
def update_summary(capacity):
    work, th, df = worklist(capacity)

    month = str(df["month"].iloc[0]) if len(df) else "n/a"
    actionable = int((work["action"] != "Monitor only").sum())
    rev_actionable = float(work.loc[work["action"] != "Monitor only", "revenue"].sum())

    if "d_avg_review_score" in work.columns:
        falling = int((work["d_avg_review_score"] < 0).sum())
        falling_sub = f"of {len(work)} On The Worklist"
    else:
        falling, falling_sub = 0, "no review data"

    kpis = html.Div([
        kpi_card("\u25c8", VIOLET, "On The Worklist", f"{len(work)}",
                 f"Highest Risk Sellers in {month}"),
        kpi_card("\u2713", EMERALD, "Worth Acting On", f"{actionable}",
                 "The Rest Look Like Mean Reversion"),
        kpi_card("\u2193", ROSE, "Reviews Falling", f"{falling}", falling_sub),
        kpi_card("\u25ae", AMBER, "Revenue At Stake", f"R$ {rev_actionable:,.0f}",
                 "Held by Sellers Worth Acting On"),
    ], className="kpi-row")

    sentiment_fig = build_sentiment_chart(work)
    charts = html.Div([
        html.Div([
            html.P("Where the worklist goes", className="card-title"),
            html.P("One action per seller - the highest priority rule wins",
                   className="card-sub"),
            dcc.Graph(figure=build_action_chart(work),
                      config={"displayModeBar": False}),
        ], className="card"),

        html.Div([
            html.P("Review movement across the worklist", className="card-title"),
            html.P("Month-on-month change in average review score",
                   className="card-sub"),
            (dcc.Graph(figure=sentiment_fig, config={"displayModeBar": False})
             if sentiment_fig is not None
             else html.P("No review data in the scored table.", style=EMPTY_NOTE)),
            html.P("Red is a seller whose reviews got worse this month. Two of the "
                   "five rules key on this rather than on the review level, because "
                   "a score that fell half a star is a different problem from one "
                   "that was always mediocre.", className="chart-note"),
        ], className="card"),
    ], className="duo-row")

    cards = html.Div([
        html.P("What each action means", className="card-title"),
        html.P("And who would own it", className="card-sub"),
        html.Div(action_cards(work),
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(270px, 1fr))",
                        "gap": "12px"}),
    ], className="card")

    note = html.P(
        f"Thresholds in use: late deliveries above {th['late_hi']*100:.0f}% or "
        f"rising more than {th['late_rising']*100:.0f} points; review score below "
        f"{th['review_lo']:.2f} or falling more than {abs(th['review_falling']):.2f}; "
        f"recency rising more than {th['recency_rising']:.0f} days. All are "
        f"percentiles of every seller scored this month, so they do not shift when "
        f"you change the worklist size.",
        className="chart-note")

    return kpis, charts, cards, note


@callback(
    Output("int-table", "children"),
    Input("int-capacity", "value"),
    Input("int-filter", "value"),
)
def update_table(capacity, actions):
    if not actions:
        return html.P("Tick at least one action.", style=EMPTY_NOTE)

    work, _, _ = worklist(capacity)
    work = work[work["action"].isin(actions)]
    if work.empty:
        return html.P("No sellers on this worklist have that action.",
                      style=EMPTY_NOTE)
    return build_table(work)