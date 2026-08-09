"""
sellers.py - seller sales decline risk. this is the primary model.

Answers the actual project question: which sellers are heading into a
revenue decline next month.

Everything on this page comes from rolling origin cross validation -
the model is trained on months 1..k and tested on k+1, sliding forward
11 times. so every number here is out-of-sample, and the +/- figures
are the spread across folds rather than a single lucky split.
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, dash_table
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import (
    load_seller_risk_table, load_seller_cv_results, load_seller_model_meta,
    load_seller_shap_groups, load_ablation_results,
)
from theme import (
    LOW, MID, HIGH, VIOLET, CYAN, AMBER, ROSE, EMERALD,
    TEXT, TEXT_DIM, TEXT_FAINT, LINE, LINE_SOFT, CARD, MONO, SANS, rgba,
)

dash.register_page(__name__, path="/sellers", name="Seller risk")

# fallback only - the real threshold comes from models/seller_model_meta.pkl
# because it was tuned on out-of-fold predictions, not picked by hand
DEFAULT_THRESHOLD = 0.33

GROUP_COLOUR = {"Trading": VIOLET, "RFM": CYAN, "Delivery": AMBER, "Sentiment": ROSE}


def get_meta():
    try:
        m = load_seller_model_meta()
        return m.get("model", "Random Forest"), float(m.get("threshold", DEFAULT_THRESHOLD))
    except Exception:
        return "Random Forest", DEFAULT_THRESHOLD


def base_layout(fig, height=260):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, size=11, family=SANS),
        margin=dict(l=6, r=12, t=6, b=26), height=height, showlegend=False,
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


def build_model_strip(best_name, threshold):
    try:
        cv = load_seller_cv_results()
        row = cv.loc[cv["mean_auc"].idxmax()]
        stats = [
            ("Model", str(row["model"]), False),
            ("AUC-ROC", f"{row['mean_auc']:.3f} ± {row['std_auc']:.3f}", True),
            ("CV folds", f"{int(row['n_folds'])}", False),
            ("Threshold", f"{threshold:.2f}", False),
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


# ------------------------------------------------- model comparison

def build_model_chart():
    """three models with their fold-to-fold range as error bars.

    the baseline matters here - if Random Forest couldnt beat Logistic
    Regression there'd be no case for using it, so its worth showing
    rather than quietly leaving out
    """
    try:
        cv = load_seller_cv_results().sort_values("mean_auc")
    except Exception:
        return None

    colours = [VIOLET if m == cv.iloc[-1]["model"] else rgba(TEXT_DIM, 0.5)
               for m in cv["model"]]

    fig = go.Figure(go.Bar(
        x=cv["mean_auc"] - 0.5, y=cv["model"], orientation="h", base=0.5,
        marker=dict(color=colours, line=dict(width=0)), width=0.5,
        error_x=dict(type="data", array=cv["std_auc"], color=TEXT_FAINT,
                     thickness=1, width=4),
        text=[f"{v:.3f}" for v in cv["mean_auc"]], textposition="outside",
        textfont=dict(size=11, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>mean AUC %{x:.3f}<extra></extra>",
    ))
    fig = base_layout(fig, height=210)
    fig.update_xaxes(range=[0.5, cv["mean_auc"].max() * 1.12], title="AUC-ROC")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=48, t=6, b=32))
    return fig


# --------------------------------------------- risk distribution

def build_risk_hist(df, threshold):
    """where the scored sellers actually sit on the risk scale.

    a mean risk score on its own hides the shape - this shows whether
    the model is confident or hedging everything near the middle
    """
    fig = go.Figure(go.Histogram(
        x=df["risk_score"], nbinsx=26,
        marker=dict(color=rgba(VIOLET, 0.55), line=dict(color=VIOLET, width=1)),
        hovertemplate="risk %{x:.2f}<br>%{y} sellers<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_width=1.5, line_dash="dash", line_color=ROSE,
                  annotation_text=f"threshold {threshold:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(color=ROSE, size=10.5, family=SANS))
    fig = base_layout(fig, height=210)
    fig.update_xaxes(title="Predicted risk", range=[0, 1])
    fig.update_yaxes(title=None)
    return fig


# ----------------------------------------------------- SHAP groups

def build_group_chart():
    try:
        g = load_seller_shap_groups().sort_values("share_pct")
    except Exception:
        return None

    fig = go.Figure(go.Bar(
        x=g["share_pct"], y=g["group"], orientation="h",
        marker=dict(color=[GROUP_COLOUR.get(n, VIOLET) for n in g["group"]],
                    line=dict(width=0)), width=0.55,
        customdata=g["n_features"],
        text=[f"{v:.1f}%" for v in g["share_pct"]], textposition="outside",
        textfont=dict(size=11, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>%{x:.1f}% of total SHAP<br>%{customdata} features<extra></extra>",
    ))
    fig = base_layout(fig, height=230)
    fig.update_xaxes(range=[0, g["share_pct"].max() * 1.25], title=None)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=44, t=6, b=24))
    return fig


# ------------------------------------------ ablation (objective 3)

def build_ablation_chart():
    try:
        a = load_ablation_results().sort_values("mean_auc")
    except Exception:
        return None

    # bars start at 0.5 because thats random guessing - drawing from zero
    # would make everything look impressively tall for no reason
    colours = []
    for s in a["feature_set"]:
        if "All" in s:
            colours.append(VIOLET)
        elif s == "RFM + Sentiment":
            colours.append(CYAN)
        else:
            colours.append(rgba(TEXT_DIM, 0.5))

    fig = go.Figure(go.Bar(
        x=a["mean_auc"] - 0.5, y=a["feature_set"], orientation="h", base=0.5,
        marker=dict(color=colours, line=dict(width=0)), width=0.6,
        error_x=dict(type="data", array=a["std_auc"], color=TEXT_FAINT,
                     thickness=1, width=3),
        customdata=a["n_features"],
        text=[f"{v:.3f}" for v in a["mean_auc"]], textposition="outside",
        textfont=dict(size=10.5, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>AUC %{x:.3f}<br>%{customdata} features<extra></extra>",
    ))
    fig = base_layout(fig, height=290)
    fig.update_xaxes(range=[0.5, a["mean_auc"].max() * 1.10], title="AUC-ROC")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=50, t=6, b=32))
    return fig


# ----------------------------------------------------------- table

def build_table(df, threshold):
    flagged = df[df["risk_score"] >= threshold]
    top = flagged.nlargest(10, "revenue").copy()

    top["month"] = top["month"].astype(str)
    top["revenue"] = top["revenue"].map(lambda v: f"{v:,.0f}")
    top["avg_review_score"] = top["avg_review_score"].map(lambda v: f"{v:.2f}")
    top["pct_late_deliveries"] = top["pct_late_deliveries"].map(lambda v: f"{v*100:.0f}%")
    top["d_revenue"] = top["d_revenue"].map(lambda v: f"{v:+,.0f}")
    top["risk_score"] = top["risk_score"].map(lambda v: f"{v:.3f}")
    top["seller_id"] = top["seller_id"].str[:12]

    cols = [
        ("seller_id", "Seller"), ("revenue", "Revenue (R$)"),
        ("d_revenue", "Δ vs last month"), ("orders", "Orders"),
        ("avg_review_score", "Reviews"), ("pct_late_deliveries", "Late"),
        ("risk_score", "Risk"),
    ]
    numeric = ["revenue", "d_revenue", "orders", "avg_review_score",
               "pct_late_deliveries", "risk_score"]

    return dash_table.DataTable(
        data=top[[c for c, _ in cols]].to_dict("records"),
        columns=[{"name": lbl, "id": cid} for cid, lbl in cols],
        sort_action="native", style_as_list_view=True,
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
            {"if": {"column_id": numeric}, "fontFamily": MONO, "fontSize": "12px"},
            {"if": {"column_id": "risk_score"}, "color": HIGH, "fontWeight": "600"},
        ],
    )


# ---------------------------------------------------------- layout

def layout():
    try:
        df = load_seller_risk_table()
    except FileNotFoundError:
        return html.Div([
            html.H1("Seller risk"),
            html.P("Run src/seller_features.py then src/seller_model.py to "
                   "generate the seller risk table.", className="page-sub"),
        ])

    best_name, threshold = get_meta()

    n = len(df)
    flagged = int((df["risk_score"] >= threshold).sum())
    mean_risk = df["risk_score"].mean()
    rev_flagged = df.loc[df["risk_score"] >= threshold, "revenue"].sum()
    total_rev = df["revenue"].sum()
    month = str(df["month"].iloc[0]) if n else "n/a"

    blocks = [
        html.Div([
            html.Div([
                html.H1("Seller decline risk"),
                html.P(
                    f"Probability that a seller's revenue falls more than 20% the "
                    f"following month. Validated by rolling-origin cross-validation, "
                    f"so every score is out-of-sample. Table below shows {month}.",
                    className="page-sub"),
            ], className="page-head-text"),
            build_model_strip(best_name, threshold),
        ], className="page-head"),

        html.Div([
            kpi_card("\u25c8", VIOLET, "Sellers scored", f"{n:,}", f"in {month}"),
            kpi_card("\u25b2", ROSE, "Flagged", f"{flagged:,}",
                     f"{flagged / n * 100:.0f}% at threshold {threshold:.2f}",
                     flagged / n * 100),
            kpi_card("\u25d0", AMBER, "Mean risk", f"{mean_risk:.3f}",
                     "0 = stable, 1 = certain decline", mean_risk * 100),
            kpi_card("\u25ae", EMERALD, "Revenue flagged", f"R$ {rev_flagged:,.0f}",
                     f"{rev_flagged / total_rev * 100:.0f}% of the month's revenue",
                     rev_flagged / total_rev * 100),
        ], className="kpi-row"),
    ]

    model_fig = build_model_chart()
    if model_fig is not None:
        blocks.append(html.Div([
            html.Div([
                html.P("Model comparison", className="card-title"),
                html.P("Mean AUC across CV folds, bars show standard deviation",
                       className="card-sub"),
                dcc.Graph(figure=model_fig, config={"displayModeBar": False}),
                html.P("Logistic Regression is included as a baseline — the tree "
                       "models have to beat it to be worth using.",
                       className="chart-note"),
            ], className="card"),

            html.Div([
                html.P("Risk score distribution", className="card-title"),
                html.P(f"All sellers scored in {month}", className="card-sub"),
                dcc.Graph(figure=build_risk_hist(df, threshold),
                          config={"displayModeBar": False}),
                html.P(f"Threshold {threshold:.2f} was tuned on out-of-fold "
                       f"predictions, not left at the 0.50 default — missing a "
                       f"declining seller costs more than a false alarm.",
                       className="chart-note"),
            ], className="card"),
        ], className="duo-row"))

    group_fig = build_group_chart()
    abl_fig = build_ablation_chart()
    if group_fig is not None and abl_fig is not None:
        blocks.append(html.Div([
            html.Div([
                html.P("What drives the prediction", className="card-title"),
                html.P("Share of total SHAP importance by feature group",
                       className="card-sub"),
                dcc.Graph(figure=group_fig, config={"displayModeBar": False}),
                html.P("Trading momentum leads, but every group here gets the same "
                       "treatment — level, month-on-month change, rolling average "
                       "and trend — so the comparison is like-for-like.",
                       className="chart-note"),
            ], className="card"),

            html.Div([
                html.P("Feature groups tested separately", className="card-title"),
                html.P("Same model and folds, retrained on each subset",
                       className="card-sub"),
                dcc.Graph(figure=abl_fig, config={"displayModeBar": False}),
                html.P("RFM and sentiment together beat either alone, so they carry "
                       "complementary signal — though they add little once revenue "
                       "momentum is already known.",
                       className="chart-note"),
            ], className="card"),
        ], className="duo-row"))

    blocks.append(html.Div([
        html.P("Largest flagged sellers", className="card-title"),
        html.P("Highest revenue among sellers above the threshold — intervening "
               "here protects the most revenue", className="card-sub"),
        build_table(df, threshold),
    ], className="card"))

    return html.Div(blocks)