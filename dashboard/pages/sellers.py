"""
sellers.py - seller sales decline risk. this is the primary model.

The customer pages answer "which customers look unhappy". This one
answers the actual project question: which sellers are heading into a
revenue decline next month.

Everything here is scored on the held-out test period (2018-03
onwards) - the model never saw these months during training.
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, dash_table
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import (
    load_seller_risk_table, load_seller_model_comparison,
    load_seller_shap_groups, load_ablation_results,
)
from theme import (
    LOW, MID, HIGH, VIOLET, CYAN, AMBER, ROSE, EMERALD,
    TEXT, TEXT_DIM, TEXT_FAINT, LINE, LINE_SOFT, CARD, MONO, SANS, rgba,
)

dash.register_page(__name__, path="/sellers", name="Sellers")

WATCH_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.60

GROUP_COLOUR = {
    "Trading": VIOLET,
    "RFM": CYAN,
    "Delivery": AMBER,
    "Sentiment": ROSE,
}


def band_of(score):
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= WATCH_THRESHOLD:
        return "mid"
    return "low"


BAND_COLOUR = {"low": LOW, "mid": MID, "high": HIGH}


def base_layout(fig, height=260):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, size=11, family=SANS),
        margin=dict(l=6, r=12, t=6, b=24),
        height=height, showlegend=False,
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE,
                        font=dict(color=TEXT, size=11.5, family=SANS)),
    )
    fig.update_xaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT)
    fig.update_yaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT)
    return fig


def kpi_card(icon, colour, label, value, sub=None, fill_pct=None):
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
    if fill_pct is not None:
        children.append(html.Div(
            html.Div(className="kpi-bar-fill",
                     style={"width": f"{min(fill_pct, 100):.1f}%", "background": colour}),
            className="kpi-bar"))
    return html.Div(children, className="kpi-card")


def build_model_strip():
    try:
        comp = load_seller_model_comparison()
        best = comp.loc[comp["f1"].idxmax()]
        stats = [
            ("Model", str(best["model"]), False),
            ("AUC-ROC", f"{best['auc']:.3f}", True),
            ("F1 decline", f"{best['f1']:.3f}", False),
            ("Validation", "temporal", False),
        ]
    except Exception:
        return html.Div()

    return html.Div([
        html.Div([
            html.Span(label, className="model-stat-label"),
            html.Span(value, className="model-stat-value accent" if accent
                      else "model-stat-value"),
        ], className="model-stat")
        for label, value, accent in stats
    ], className="model-strip")


# ------------------------------------------------- risk over the months

def build_monthly_risk(df):
    monthly = df.groupby("month").agg(
        mean_risk=("risk_score", "mean"),
        sellers=("seller_id", "nunique"),
    ).reset_index()
    monthly["month_str"] = monthly["month"].astype(str)

    fig = go.Figure(go.Scatter(
        x=monthly["month_str"], y=monthly["mean_risk"],
        mode="lines+markers",
        line=dict(color=VIOLET, width=2, shape="spline", smoothing=0.5),
        marker=dict(size=6, color=VIOLET),
        fill="tozeroy", fillcolor=rgba(VIOLET, 0.12),
        customdata=monthly["sellers"],
        hovertemplate="%{x}<br>mean risk %{y:.3f}<br>%{customdata} sellers<extra></extra>",
    ))
    fig = base_layout(fig, height=250)
    fig.update_yaxes(range=[0, max(monthly["mean_risk"].max() * 1.3, 0.1)])
    return fig


# ------------------------------------------ what drives it (SHAP groups)

def build_group_chart():
    try:
        g = load_seller_shap_groups().sort_values("share_pct")
    except Exception:
        return None

    fig = go.Figure(go.Bar(
        x=g["share_pct"], y=g["group"], orientation="h",
        marker=dict(color=[GROUP_COLOUR.get(n, VIOLET) for n in g["group"]],
                    line=dict(width=0)),
        width=0.55,
        text=[f"{v:.1f}%" for v in g["share_pct"]],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>%{x:.1f}% of total SHAP<extra></extra>",
    ))
    fig = base_layout(fig, height=230)
    fig.update_xaxes(range=[0, g["share_pct"].max() * 1.25], title=None)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=40, t=6, b=24))
    return fig


# ------------------------------------------------ ablation (objective 3)

def build_ablation_chart():
    try:
        a = load_ablation_results().sort_values("auc")
    except Exception:
        return None

    # 0.5 is random guessing, so the bars are drawn from there rather than
    # from zero - otherwise everything looks impressively tall for no reason
    fig = go.Figure(go.Bar(
        x=a["auc"] - 0.5, y=a["feature_set"], orientation="h",
        base=0.5,
        marker=dict(
            color=[VIOLET if "All" in s else
                   CYAN if "RFM + Sentiment" in s else
                   rgba(TEXT_DIM, 0.55) for s in a["feature_set"]],
            line=dict(width=0)),
        width=0.6,
        text=[f"{v:.3f}" for v in a["auc"]],
        textposition="outside",
        textfont=dict(size=10.5, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>AUC %{x:.3f}<extra></extra>",
    ))
    fig = base_layout(fig, height=280)
    fig.update_xaxes(range=[0.5, a["auc"].max() * 1.08], title="AUC-ROC")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=46, t=6, b=32))
    return fig


# ----------------------------------------------------------- the table

def build_table(df):
    # most recent scored month per seller, biggest sellers first - a seller
    # doing R$20k that the model flags matters more than one doing R$200
    latest = df.sort_values("month").groupby("seller_id").tail(1)
    flagged = latest[latest["risk_score"] >= WATCH_THRESHOLD]
    top = flagged.nlargest(10, "revenue").copy()

    top["risk_band"] = top["risk_score"].map(band_of)
    top["month"] = top["month"].astype(str)
    top["revenue"] = top["revenue"].map(lambda v: f"{v:,.0f}")
    top["avg_review_score"] = top["avg_review_score"].map(lambda v: f"{v:.2f}")
    top["pct_late_deliveries"] = top["pct_late_deliveries"].map(lambda v: f"{v*100:.0f}%")
    top["revenue_change_prev"] = top["revenue_change_prev"].map(lambda v: f"{v*100:+.0f}%")
    top["risk_score"] = top["risk_score"].map(lambda v: f"{v:.3f}")
    top["seller_id"] = top["seller_id"].str[:12]

    cols = [
        ("seller_id", "Seller"), ("month", "Month"),
        ("revenue", "Revenue (R$)"), ("orders", "Orders"),
        ("revenue_change_prev", "MoM"), ("avg_review_score", "Reviews"),
        ("pct_late_deliveries", "Late"), ("risk_score", "Risk"),
    ]
    numeric = ["revenue", "orders", "revenue_change_prev",
               "avg_review_score", "pct_late_deliveries", "risk_score"]

    return dash_table.DataTable(
        data=top[[c for c, _ in cols] + ["risk_band"]].to_dict("records"),
        columns=[{"name": lbl, "id": cid} for cid, lbl in cols],
        sort_action="native",
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
        ],
        style_data_conditional=[
            {"if": {"column_id": "seller_id"},
             "fontFamily": MONO, "color": TEXT_DIM, "fontSize": "11.5px"},
            {"if": {"column_id": "month"}, "color": TEXT_DIM},
            {"if": {"column_id": numeric}, "fontFamily": MONO, "fontSize": "12px"},
            {"if": {"filter_query": "{risk_band} = high", "column_id": "risk_score"},
             "color": HIGH, "fontWeight": "600"},
            {"if": {"filter_query": "{risk_band} = mid", "column_id": "risk_score"},
             "color": MID},
            {"if": {"filter_query": "{risk_band} = low", "column_id": "risk_score"},
             "color": LOW},
        ],
    )


# ------------------------------------------------------------- layout

def layout():
    try:
        df = load_seller_risk_table()
    except FileNotFoundError:
        return html.Div([
            html.H1("Sellers"),
            html.P("Run src/seller_features.py then src/seller_classifier.py "
                   "to generate the seller risk table.", className="page-sub"),
        ])

    latest = df.sort_values("month").groupby("seller_id").tail(1)
    n_sellers = latest["seller_id"].nunique()
    high = int((latest["risk_score"] >= HIGH_THRESHOLD).sum())
    watch = int(((latest["risk_score"] >= WATCH_THRESHOLD)
                 & (latest["risk_score"] < HIGH_THRESHOLD)).sum())
    mean_risk = latest["risk_score"].mean()
    rev_at_risk = latest.loc[latest["risk_score"] >= WATCH_THRESHOLD, "revenue"].sum()
    total_rev = latest["revenue"].sum()

    months = f"{df['month'].min()} – {df['month'].max()}"

    blocks = [
        html.Div([
            html.Div([
                html.H1("Seller decline risk"),
                html.P(
                    f"Probability that a seller's revenue falls more than 20% next "
                    f"month. Scored on held-out months ({months}) the model never "
                    f"trained on.",
                    className="page-sub"),
            ], className="page-head-text"),
            build_model_strip(),
        ], className="page-head"),

        html.Div([
            kpi_card("\u25c8", VIOLET, "Sellers scored", f"{n_sellers:,}",
                     "in the latest scored month"),
            kpi_card("\u25b2", ROSE, "High risk", f"{high:,}",
                     f"{high / n_sellers * 100:.1f}% of sellers",
                     high / n_sellers * 100),
            kpi_card("\u25d0", AMBER, "Watch", f"{watch:,}",
                     "risk between 0.40 and 0.60",
                     watch / n_sellers * 100),
            kpi_card("\u25ae", EMERALD, "Revenue at risk", f"R$ {rev_at_risk:,.0f}",
                     f"{rev_at_risk / total_rev * 100:.1f}% of monthly revenue",
                     rev_at_risk / total_rev * 100),
        ], className="kpi-row"),

        html.Div([
            html.P("Mean risk by month", className="card-title"),
            html.P("Across all sellers scored in each held-out month",
                   className="card-sub"),
            dcc.Graph(figure=build_monthly_risk(df), config={"displayModeBar": False}),
        ], className="card"),
    ]

    group_fig = build_group_chart()
    abl_fig = build_ablation_chart()

    if group_fig is not None and abl_fig is not None:
        blocks.append(html.Div([
            html.Div([
                html.P("What drives the prediction", className="card-title"),
                html.P("Share of total SHAP importance by feature group",
                       className="card-sub"),
                dcc.Graph(figure=group_fig, config={"displayModeBar": False}),
                html.P(
                    "Trading momentum dominates — a seller's own revenue trend "
                    "carries most of the signal.",
                    className="chart-note"),
            ], className="card"),

            html.Div([
                html.P("Feature groups tested separately", className="card-title"),
                html.P("Same model and split, trained on each subset",
                       className="card-sub"),
                dcc.Graph(figure=abl_fig, config={"displayModeBar": False}),
                html.P(
                    "RFM and sentiment together beat either on its own, so they "
                    "carry complementary signal — but they add little once revenue "
                    "momentum is already known.",
                    className="chart-note"),
            ], className="card"),
        ], className="duo-row"))

    blocks.append(html.Div([
        html.P("Largest sellers currently flagged", className="card-title"),
        html.P("Most recent scored month per seller, ranked by revenue",
               className="card-sub"),
        build_table(df),
    ], className="card"))

    return html.Div(blocks)