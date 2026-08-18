"""
overview.py - landing page.

KPI strip, risk distribution donut, monthly order volume, mean risk
by segment, and the highest-spend customers currently flagged.

colour encodes risk everywhere on this page: cyan = low, amber =
watch, rose = high. same three in the donut, the segment bars and
the table pills, so the scale only has to be learned once.
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, dash_table
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import load_risk_table, load_merged_dataset, load_model_comparison
from theme import (
    LOW, MID, HIGH, VIOLET, CYAN, AMBER, ROSE, EMERALD,
    TEXT, TEXT_DIM, TEXT_FAINT, LINE, LINE_SOFT, CARD, MONO, SANS, rgba,
)

dash.register_page(__name__, path="/", name="Overview")

WATCH_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.50


def band_of(score):
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= WATCH_THRESHOLD:
        return "mid"
    return "low"


BAND_COLOUR = {"low": LOW, "mid": MID, "high": HIGH}


def base_layout(fig, height=260):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, size=11, family=SANS),
        margin=dict(l=6, r=10, t=6, b=24),
        height=height,
        showlegend=False,
        hoverlabel=dict(
            bgcolor=CARD, bordercolor=LINE,
            font=dict(color=TEXT, size=11.5, family=SANS),
        ),
    )
    fig.update_xaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT)
    fig.update_yaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT)
    return fig


# ---------------------------------------------------------------- KPI

def kpi_card(icon, colour, label, value, sub=None, fill_pct=None, href=None, cta=None):
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
            className="kpi-bar",
        ))
    if cta:
        children.append(html.Div(cta + "  \u2192", className="kpi-cta"))

    card = html.Div(children, className="kpi-card")

    # wrapping in a Link makes the whole card clickable through to the
    # filtered customer list - dcc.Link instead of html.A so dash routes
    # it client side rather than reloading the page
    if href:
        return dcc.Link(card, href=href, className="kpi-link")
    return card


# ------------------------------------------------------------- donut

def build_donut(counts):
    low, mid, high = counts
    total = low + mid + high

    fig = go.Figure(go.Pie(
        values=[low, mid, high],
        labels=["Low", "Watch", "High"],
        hole=0.74,
        marker=dict(colors=[LOW, MID, HIGH], line=dict(color=CARD, width=3)),
        textinfo="none",
        hovertemplate="%{label}<br>%{value:,} customers (%{percent})<extra></extra>",
        sort=False,
        direction="clockwise",
    ))
    fig.add_annotation(
        text=f"<b>{total:,}</b>", showarrow=False,
        font=dict(size=21, color=TEXT, family=SANS), y=0.56,
    )
    fig.add_annotation(
        text="scored", showarrow=False,
        font=dict(size=11, color=TEXT_FAINT, family=SANS), y=0.40,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=4, b=4), height=210, showlegend=False,
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE,
                        font=dict(color=TEXT, size=11.5, family=SANS)),
    )
    return fig


def donut_legend(counts):
    low, mid, high = counts
    total = low + mid + high
    rows = [
        ("Low", low, LOW, "under 0.25"),
        ("Watch", mid, MID, "0.25 – 0.50"),
        ("High", high, HIGH, "0.50 and above"),
    ]
    return html.Div([
        html.Div([
            html.Span(className="legend-dot", style={"background": colour}),
            html.Span(name, className="legend-name"),
            html.Span(f"{count:,}", className="legend-val", style={"color": colour}),
            html.Span(f"{count / total * 100:.1f}%", className="legend-pct"),
        ], className="legend-row")
        for name, count, colour, _ in rows
    ], className="legend")


# -------------------------------------------------------- area chart

def build_monthly_trend():
    df = load_merged_dataset()
    monthly = (
        df.set_index("order_purchase_timestamp")
        .resample("ME")["order_id"].nunique().reset_index()
    )
    monthly.columns = ["month", "orders"]

    fig = go.Figure(go.Scatter(
        x=monthly["month"], y=monthly["orders"],
        mode="lines",
        line=dict(color=VIOLET, width=2, shape="spline", smoothing=0.6),
        fill="tozeroy", fillcolor=rgba(VIOLET, 0.13),
        hovertemplate="%{x|%b %Y}<br>%{y:,} orders<extra></extra>",
    ))
    return base_layout(fig, height=250)


# ------------------------------------------------------- segment bars

def build_segment_chart(df):
    seg = (
        df.groupby("segment_name")
        .agg(avg_risk=("risk_score", "mean"), customers=("customer_unique_id", "count"))
        .reset_index().sort_values("avg_risk")
    )

    fig = go.Figure(go.Bar(
        x=seg["avg_risk"], y=seg["segment_name"], orientation="h",
        marker=dict(
            color=[BAND_COLOUR[band_of(v)] for v in seg["avg_risk"]],
            line=dict(width=0),
        ),
        width=0.55,
        customdata=seg["customers"],
        text=[f"{v:.3f}" for v in seg["avg_risk"]],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_DIM, family=MONO),
        hovertemplate="%{y}<br>mean risk %{x:.3f}<br>%{customdata:,} customers<extra></extra>",
    ))
    fig = base_layout(fig, height=250)
    fig.update_xaxes(range=[0, seg["avg_risk"].max() * 1.3], title=None)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=34, t=6, b=24))
    return fig


# ------------------------------------------------- delivery vs risk

def build_delivery_chart(df):
    # delivery delay is the single biggest driver in the SHAP ranking but
    # nothing on this page actually showed it, so binning it here. negative
    # delay = arrived before the estimate, which is most of the dataset
    bins = [-float("inf"), -0.001, 3, 10, float("inf")]
    labels = ["Early", "On time (0-3d)", "Late (4-10d)", "Late (10d+)"]

    d = df.copy()
    d["delay_band"] = pd.cut(d["avg_delivery_delay"], bins=bins, labels=labels)

    grouped = (
        d.groupby("delay_band", observed=False)
        .agg(mean_risk=("risk_score", "mean"), customers=("customer_unique_id", "count"))
        .reindex(labels).reset_index()
    )
    grouped = grouped[grouped["customers"] > 0]

    fig = go.Figure(go.Bar(
        x=grouped["delay_band"].astype(str), y=grouped["mean_risk"],
        marker=dict(
            color=[BAND_COLOUR[band_of(v)] for v in grouped["mean_risk"]],
            line=dict(width=0),
        ),
        width=0.55,
        customdata=grouped["customers"],
        text=[f"{v:.3f}" for v in grouped["mean_risk"]],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_DIM, family=MONO),
        hovertemplate="%{x}<br>mean risk %{y:.3f}<br>%{customdata:,} customers<extra></extra>",
    ))
    fig = base_layout(fig, height=250)
    fig.update_yaxes(range=[0, max(grouped["mean_risk"].max() * 1.25, 0.05)], title=None)
    fig.update_xaxes(title=None)
    return fig


# --------------------------------------------------- model stat strip

def build_model_strip(n_customers):
    # putting the real metrics on the front page rather than hiding them.
    # reads better than pretending the model is stronger than it is
    try:
        comp = load_model_comparison()
        best = comp.loc[comp["f1"].idxmax()]
        name, f1, auc = best["model"], best["f1"], best["auc"]
    except Exception:
        # reports/model_comparison.csv only exists after classifier.py has run
        return html.Div()

    # dropped "customers scored" from here - its already the first KPI card
    # right underneath, no point saying it twice
    stats = [
        ("Model", str(name), False),
        ("AUC-ROC", f"{auc:.3f}", True),
        ("F1 at-risk", f"{f1:.3f}", False),
        ("Test split", "20%", False),
    ]

    return html.Div(
        [
            html.Div([
                html.Span(label, className="model-stat-label"),
                html.Span(value, className="model-stat-value accent" if accent
                          else "model-stat-value"),
            ], className="model-stat")
            for label, value, accent in stats
        ],
        className="model-strip",
    )


# ------------------------------------------------------------- table

def build_table(df):
    # high spend AND high risk - ranked by spend among the flagged, since
    # thats where an intervention protects the most revenue
    flagged = df[df["risk_score"] >= HIGH_THRESHOLD]
    top = flagged.nlargest(9, "monetary")[
        ["customer_unique_id", "segment_name", "monetary",
         "recency", "avg_delivery_delay", "risk_score"]
    ].copy()

    top["monetary"] = top["monetary"].map(lambda v: f"{v:,.0f}")
    top["avg_delivery_delay"] = top["avg_delivery_delay"].map(lambda v: f"{v:+.0f}")
    top["risk_score"] = top["risk_score"].map(lambda v: f"{v:.3f}")
    top["customer_unique_id"] = top["customer_unique_id"].str[:10]

    numeric = ["monetary", "recency", "avg_delivery_delay", "risk_score"]
    cell_conditional = [{"if": {"column_id": c}, "textAlign": "right"} for c in numeric]

    return dash_table.DataTable(
        data=top.to_dict("records"),
        columns=[
            {"name": "Customer", "id": "customer_unique_id"},
            {"name": "Segment", "id": "segment_name"},
            {"name": "Spend (R$)", "id": "monetary"},
            {"name": "Recency (d)", "id": "recency"},
            {"name": "Delivery (d)", "id": "avg_delivery_delay"},
            {"name": "Risk", "id": "risk_score"},
        ],
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
            "padding": "12px 14px 12px 0",
            "fontSize": "12.5px",
            "fontFamily": SANS,
            "textAlign": "left",
        },
        style_cell_conditional=cell_conditional,
        style_data_conditional=[
            {"if": {"column_id": "customer_unique_id"},
             "fontFamily": MONO, "color": TEXT_DIM, "fontSize": "11.5px"},
            {"if": {"column_id": "segment_name"}, "color": TEXT_DIM},
            {"if": {"column_id": "risk_score"}, "color": HIGH, "fontWeight": "600"},
            {"if": {"column_id": ["monetary", "recency", "avg_delivery_delay"]},
             "fontFamily": MONO, "fontSize": "12px"},
        ],
    )


# ------------------------------------------------------------ layout

def layout():
    # guarded like the sellers page. without this a missing customer csv
    # gives a red callback error box instead of telling you what to run,
    # which is not what you want happening in front of anyone
    try:
        df = load_risk_table()
        load_merged_dataset()
    except FileNotFoundError:
        return html.Div([
            html.H1("Overview"),
            html.P("Run the customer pipeline first: data_pipeline.py, "
                   "rfm_features.py, sentiment.py, extra_features.py, "
                   "clustering.py, then classifier.py.",
                   className="page-sub"),
        ])

    total = len(df)
    low = int((df["risk_score"] < WATCH_THRESHOLD).sum())
    mid = int(((df["risk_score"] >= WATCH_THRESHOLD) & (df["risk_score"] < HIGH_THRESHOLD)).sum())
    high = int((df["risk_score"] >= HIGH_THRESHOLD).sum())
    counts = (low, mid, high)

    avg_risk = df["risk_score"].mean()
    exposed = df.loc[df["risk_score"] >= HIGH_THRESHOLD, "monetary"].sum()
    total_spend = df["monetary"].sum()

    return html.Div([
        html.Div([
            html.Div([
                html.H1("Overview"),
                html.P(
                    "Dissatisfaction risk across the customer base, scored from "
                    "purchase behaviour, delivery performance and payment profile. "
                    "Observation window closes 1 June 2018.",
                    className="page-sub",
                ),
            ], className="page-head-text"),

            build_model_strip(total),
        ], className="page-head"),

        html.Div([
            kpi_card("◈", VIOLET, "Customers scored", f"{total:,}",
                     "All Active Before The June Cutoff", 100,
                     href="/customers?band=all&sort=risk_score",
                     cta="See every customer"),
            kpi_card("▲", ROSE, "High risk", f"{high:,}",
                     f"{high / total * 100:.1f}% Of The Base", high / total * 100,
                     href="/customers?band=high&sort=risk_score",
                     cta="See the flagged list"),
            kpi_card("◐", AMBER, "Mean risk score", f"{avg_risk:.3f}",
                     "0 = safe, 1 = certain", avg_risk * 100,
                     href="/customers?band=watch&sort=risk_score",
                     cta="See the watch band"),
            kpi_card("▮", EMERALD, "Spend exposed", f"R$ {exposed:,.0f}",
                     f"{exposed / total_spend * 100:.1f}% Of All Spend",
                     exposed / total_spend * 100,
                     href="/customers?band=high&sort=monetary",
                     cta="See who holds it"),
        ], className="kpi-row"),

        html.Div([
            html.Div([
                html.P("Risk distribution", className="card-title"),
                html.P("Where the base sits on the scale", className="card-sub"),
                dcc.Graph(figure=build_donut(counts), config={"displayModeBar": False}),
                donut_legend(counts),
            ], className="card"),

            html.Div([
                html.P("Monthly order volume", className="card-title"),
                html.P("Unique orders per month across the dataset", className="card-sub"),
                dcc.Graph(figure=build_monthly_trend(), config={"displayModeBar": False}),
                html.P(
                    "The fall after mid-2018 is where the dataset ends, not a real "
                    "decline — which is why scoring stops at the June cutoff.",
                    className="chart-note",
                ),
            ], className="card"),
        ], className="split-row"),

        html.Div([
            html.Div([
                html.P("Mean risk by segment", className="card-title"),
                html.P("Bars take the colour of their risk band", className="card-sub"),
                dcc.Graph(figure=build_segment_chart(df), config={"displayModeBar": False}),
            ], className="card"),

            html.Div([
                html.P("Risk by delivery performance", className="card-title"),
                html.P("The strongest single driver in the model", className="card-sub"),
                dcc.Graph(figure=build_delivery_chart(df), config={"displayModeBar": False}),
                html.P(
                    "Most Olist deliveries beat their estimate, so 'early' is the "
                    "norm rather than the exception — the signal is in the late tail.",
                    className="chart-note",
                ),
            ], className="card"),
        ], className="duo-row"),

        html.Div([
            html.P("Highest spend among high-risk customers", className="card-title"),
            html.P("Ranked by lifetime spend — the top rows are where intervention "
                   "protects the most revenue", className="card-sub"),
            build_table(df),
        ], className="card"),
    ])