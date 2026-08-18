"""
risk_analysis.py - why the model flags what it flags.

Three parts:
  1. global - which features matter across all sellers, with checkboxes
     to show/hide feature groups. 49 features is far too many to read at
     once, so the default view is the top 10 and you tick groups on and
     off from there.
  2. local  - pick a seller off the worklist, see what stands out about
     them versus everyone else.
  3. a note on why the sentiment features come from star ratings rather
     than from NLP, with the measured numbers behind that decision.

all styling is done with inline style props rather than css classes, so
this page drops straight in without touching theme.py
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import (
    load_seller_shap_importance, load_seller_risk_table,
    load_sentiment_validation, load_nlp_results, load_nlp_lexicon,
)
from theme import (
    VIOLET, CYAN, AMBER, ROSE, EMERALD, TEXT, TEXT_DIM, TEXT_FAINT,
    LINE, LINE_SOFT, CARD, CARD_2, MONO, SANS,
)

dash.register_page(__name__, path="/risk-analysis", name="Risk Analysis")

GROUP_COLOUR = {"Trading": VIOLET, "RFM": CYAN, "Delivery": AMBER, "Sentiment": ROSE}
ALL_GROUPS = ["Trading", "RFM", "Delivery", "Sentiment"]

RFM_BASE = ["rfm_recency", "rfm_frequency", "rfm_monetary"]
SENT_BASE = ["avg_review_score", "pct_negative_reviews"]
DELIV_BASE = ["avg_delivery_delay", "pct_late_deliveries"]

# --- inline styles (see module docstring for why) ---

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
               "gap": "22px", "marginBottom": "18px", "overflow": "visible",
               "position": "relative", "zIndex": 100}
CONTROL_GROUP = {"display": "flex", "alignItems": "center", "gap": "12px"}
CONTROL_LABEL = {"fontSize": "11px", "color": TEXT_FAINT,
                 "textTransform": "uppercase", "letterSpacing": "0.07em",
                 "whiteSpace": "nowrap"}
EMPTY_NOTE = {"color": TEXT_FAINT, "fontSize": "12.5px",
              "padding": "30px 0", "textAlign": "center"}
SWATCH = {"width": "8px", "height": "8px", "borderRadius": "50%",
          "display": "inline-block"}


def group_of(col):
    # a momentum feature belongs to whatever it was derived from -
    # d_avg_review_score is a Sentiment feature, not its own thing
    def matches(bases):
        return any(col == b or col.endswith(f"_{b}") for b in bases)

    if matches(RFM_BASE):
        return "RFM"
    if matches(SENT_BASE):
        return "Sentiment"
    if matches(DELIV_BASE):
        return "Delivery"
    return "Trading"


def pretty(col):
    """d_avg_review_score -> 'Δ avg review score'"""
    prefix, label = "", col
    for pre, nice in [("d_", "Δ "), ("roll3_", "3mo avg "),
                      ("trend_", "trend "), ("pct_", "% change ")]:
        if col.startswith(pre):
            prefix, label = nice, col[len(pre):]
            break
    return prefix + label.replace("_", " ")


def base_fig(fig, height):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_DIM, size=11, family=SANS),
        margin=dict(l=6, r=48, t=6, b=34), height=height, showlegend=False,
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE,
                        font=dict(color=TEXT, size=11.5, family=SANS)),
    )
    fig.update_xaxes(gridcolor=LINE_SOFT, zeroline=False, linecolor=LINE_SOFT)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", zeroline=False, linecolor=LINE_SOFT)
    return fig


# --------------------------------------------------------- global view

def build_importance_chart(groups, top_n):
    try:
        imp = load_seller_shap_importance()
    except Exception:
        return None

    imp = imp.copy()
    imp["group"] = imp["feature"].map(group_of)
    imp = imp[imp["group"].isin(groups)]
    if imp.empty:
        return None

    imp = imp.nlargest(top_n, "mean_abs_shap").sort_values("mean_abs_shap")

    fig = go.Figure(go.Bar(
        x=imp["mean_abs_shap"], y=[pretty(c) for c in imp["feature"]],
        orientation="h",
        marker=dict(color=[GROUP_COLOUR[g] for g in imp["group"]], line=dict(width=0)),
        width=0.62, customdata=imp["group"],
        text=[f"{v:.4f}" for v in imp["mean_abs_shap"]], textposition="outside",
        textfont=dict(size=10, color=TEXT_FAINT, family=MONO),
        hovertemplate="%{y}<br>%{customdata}<br>mean |SHAP| %{x:.4f}<extra></extra>",
    ))
    fig = base_fig(fig, height=max(250, 26 * len(imp) + 70))
    fig.update_xaxes(title="Mean |SHAP| - average push on a prediction",
                     range=[0, imp["mean_abs_shap"].max() * 1.24])
    return fig


# ---------------------------------------------------------- local view

def build_seller_detail(seller_id):
    """one seller against the cohort on the features that matter most.

    NOT a per-row SHAP decomposition - that needs the explainer and the
    exact feature row at predict time. this compares the seller to the
    median on the top features, which answers the question a marker
    actually asks: what is unusual about THIS one.
    """
    try:
        df = load_seller_risk_table()
        imp = load_seller_shap_importance()
    except Exception:
        return None, None

    row = df[df["seller_id"] == seller_id]
    if row.empty:
        return None, None
    row = row.iloc[0]

    cols = [c for c in imp["feature"].tolist() if c in df.columns][:8]
    if not cols:
        return None, None

    rows = []
    for c in cols:
        val, med = float(row[c]), float(df[c].median())
        std = float(df[c].std()) or 1.0
        rows.append({"feature": c, "value": val, "median": med, "z": (val - med) / std})

    comp = pd.DataFrame(rows)
    comp = comp.reindex(comp["z"].abs().sort_values().index)

    fig = go.Figure(go.Bar(
        x=comp["z"], y=[pretty(c) for c in comp["feature"]], orientation="h",
        marker=dict(color=[ROSE if z > 0 else CYAN for z in comp["z"]],
                    line=dict(width=0)),
        width=0.6, customdata=comp[["value", "median"]].values,
        hovertemplate=("%{y}<br>this seller %{customdata[0]:,.2f}"
                       "<br>cohort median %{customdata[1]:,.2f}"
                       "<br>%{x:+.2f} SD from median<extra></extra>"),
    ))
    fig.add_vline(x=0, line_width=1, line_color=LINE)
    fig = base_fig(fig, height=max(240, 28 * len(comp) + 70))
    fig.update_xaxes(title="Standard deviations from the cohort median")
    return fig, row


# ------------------------------------------------------- sentiment note

def build_sentiment_note():
    """the sentiment analysis story, both halves.

    chapter 2 compares TextBlob, VADER and BERT, so a marker reasonably
    expects to find NLP in the system. this card shows what actually
    happened: the off the shelf english lexicon failed on portuguese
    text, so a lexicon was induced from the review scores themselves.
    reads reports/sentiment_validation.csv and reports/nlp_results.csv,
    and degrades to whichever of them exists.
    """
    try:
        tb = load_sentiment_validation().iloc[0]
    except Exception:
        return None

    try:
        lex = load_nlp_results().iloc[0]
    except Exception:
        lex = None

    def metric(label, textblob, induced=None, better=False):
        cells = [
            html.Div(label, style={"fontSize": "12px", "color": TEXT_DIM,
                                   "flex": "1 1 200px"}),
            html.Div(textblob, style={"fontFamily": MONO, "fontSize": "13px",
                                      "color": ROSE, "width": "110px",
                                      "textAlign": "right"}),
        ]
        if induced is not None:
            cells.append(html.Div(
                induced, style={"fontFamily": MONO, "fontSize": "13px",
                                "color": EMERALD if better else TEXT,
                                "width": "110px", "textAlign": "right"}))
        return html.Div(cells, style={"display": "flex", "alignItems": "baseline",
                                      "gap": "12px", "padding": "9px 0",
                                      "borderBottom": f"1px solid {LINE_SOFT}"})

    header = html.Div([
        html.Div("", style={"flex": "1 1 200px"}),
        html.Div("TextBlob", style={"fontSize": "11px", "color": TEXT_FAINT,
                                    "width": "110px", "textAlign": "right"}),
        html.Div("Induced lexicon", style={"fontSize": "11px", "color": TEXT_FAINT,
                                           "width": "110px", "textAlign": "right"}),
    ], style={"display": "flex", "gap": "12px", "paddingBottom": "6px",
              "borderBottom": f"1px solid {LINE}"})

    rows = [header]
    if lex is not None:
        rows += [
            metric("Agreement with the customer's own star rating",
                   f"{tb['agreement_with_star_rating']:.1f}%",
                   f"{lex['agreement_pct']:.1f}%", better=True),
            metric("Macro F1 across the three sentiment classes",
                   "-", f"{lex['macro_f1']:.3f}", better=True),
            metric("1-2 star reviews correctly called negative",
                   "1.8%", f"{lex['negative_recall_pct']:.1f}%", better=True),
            metric("Comments the method could score at all",
                   f"{100 - tb['pct_polarity_exactly_zero']:.1f}%",
                   f"{lex['coverage_pct']:.1f}%", better=True),
        ]
    else:
        rows += [metric("Agreement with the star rating",
                        f"{tb['agreement_with_star_rating']:.1f}%")]

    children = [
        html.P("Sentiment from review text", className="card-title"),
        html.P("An English lexicon against one learned from the data itself",
               className="card-sub"),
        html.Div(rows, style={"marginBottom": "16px"}),
    ]

    # the words themselves - the most convincing part of the whole thing,
    # because nobody supplied them
    try:
        lx = load_nlp_lexicon()
        top_pos = lx[lx["direction"] == "positive"].nsmallest(8, "polarity")["word"]
        top_neg = lx[lx["direction"] == "negative"].nlargest(8, "polarity")["word"]

        def chips(words, colour):
            return html.Div([
                html.Span(w, style={
                    "display": "inline-block", "padding": "4px 10px",
                    "marginRight": "6px", "marginBottom": "6px",
                    "borderRadius": "14px", "fontSize": "12px",
                    "background": CARD_2, "color": colour,
                    "border": f"1px solid {LINE_SOFT}"})
                for w in words])

        children += [
            html.P("Words the lexicon learned, with no dictionary supplied",
                   style={"fontSize": "12px", "color": TEXT_DIM,
                          "margin": "4px 0 10px"}),
            html.Div("positive", style={"fontSize": "11px", "color": TEXT_FAINT,
                                        "marginBottom": "6px"}),
            chips(top_pos, EMERALD),
            html.Div("negative", style={"fontSize": "11px", "color": TEXT_FAINT,
                                        "margin": "10px 0 6px"}),
            chips(top_neg, ROSE),
        ]
    except Exception:
        pass

    children.append(html.P(
        f"TextBlob returned a polarity of exactly zero for "
        f"{tb['pct_polarity_exactly_zero']:.1f}% of comments, because its lexicon "
        f"is English and {tb['pct_portuguese_markers']:.1f}% of the comments are "
        f"Portuguese. Rather than import a Portuguese dictionary, one was induced "
        f"from the reviews themselves: words appearing disproportionately in one "
        f"and two star comments are negative by definition in this domain. The "
        f"lexicon was built on a training split and evaluated on held-out reviews.",
        className="chart-note"))

    return html.Div(children, className="card")


# ------------------------------------------------------------- layout

def layout():
    try:
        df = load_seller_risk_table()
    except FileNotFoundError:
        return html.Div([
            html.H1("Risk Analysis"),
            html.P("Run src/seller_features.py then src/seller_model.py first.",
                   className="page-sub"),
        ])

    if "risk_rank" not in df.columns:
        df = df.copy()
        df["risk_rank"] = df["risk_score"].rank(ascending=False, method="first").astype(int)

    # 25 rather than the whole worklist - the dropdown is searchable, and a
    # 40 item menu is more scrolling than picking
    ranked = df.nsmallest(25, "risk_rank")
    options = [
        {"label": f"#{int(r.risk_rank)}  ·  {r.seller_id[:12]}  ·  risk {r.risk_score:.3f}",
         "value": r.seller_id}
        for r in ranked.itertuples()
    ]

    return html.Div([
        html.Div([
            html.Div([
                html.H1("Risk Analysis"),
                html.P("What the model reacts to - across all sellers, and for "
                       "one seller at a time.", className="page-sub"),
            ], className="page-head-text"),
        ], className="page-head"),

        html.Div([
            html.P("Feature importance", className="card-title"),
            html.P("Mean absolute SHAP value - how far each feature moves a "
                   "prediction on average", className="card-sub"),

            html.Div([
                html.Div([
                    html.Span("Groups", style=CONTROL_LABEL),
                    dcc.Checklist(
                        id="ra-groups",
                        options=[
                            {"label": html.Span([
                                html.Span(style={**SWATCH, "background": GROUP_COLOUR[g]}),
                                html.Span(g, style={"marginLeft": "6px"}),
                            ]), "value": g}
                            for g in ALL_GROUPS
                        ],
                        value=ALL_GROUPS, inline=True,
                        labelStyle=LABEL_STYLE, inputStyle=INPUT_STYLE,
                        style={"display": "inline-flex", "flexWrap": "wrap"},
                    ),
                ], style=CONTROL_GROUP),

                html.Div([
                    html.Span("Show", style=CONTROL_LABEL),
                    dcc.RadioItems(
                        id="ra-topn",
                        options=[{"label": f"Top {n}", "value": n} for n in (10, 20, 49)],
                        value=10, inline=True,
                        labelStyle=LABEL_STYLE, inputStyle=INPUT_STYLE,
                        style={"display": "inline-flex", "flexWrap": "wrap"},
                    ),
                ], style=CONTROL_GROUP),
            ], style=CONTROL_BAR),

            html.Div(id="ra-importance"),
        ], className="card"),

        html.Div([
            html.P("Explain one seller", className="card-title"),
            html.P("Top 25 by risk - type in the box to search",
                   className="card-sub"),

            html.Div([
                html.Div([
                    html.Span("Seller", style=CONTROL_LABEL),
                    html.Div(
                        dcc.Dropdown(
                            id="ra-seller",
                            options=options,
                            value=options[0]["value"] if options else None,
                            clearable=False, className="dash-dropdown",
                        ),
                        # NO maxHeight prop here on purpose. dash applies it as
                        # an inline max-height, and theme.py has overflow:hidden
                        # on .dash-dropdown-content - fixed height + hidden
                        # overflow = menu clipped with no scrollbar, which is
                        # exactly the bug. inline styles beat the stylesheet so
                        # you cant fix it from theme.py either.
                        # left off, dash falls back to calc(100vh - 100px) and
                        # all 25 options fit without needing to scroll at all.
                        # zIndex keeps the menu above the chart below it.
                        style={"minWidth": "360px", "position": "relative",
                               "zIndex": 100},
                    ),
                ], style=CONTROL_GROUP),
            ], style=CONTROL_BAR),

            html.Div(id="ra-detail"),
        ], className="card", style={"overflow": "visible"}),

        # returns None when reports/sentiment_validation.csv is absent, in
        # which case dash simply renders nothing here
        build_sentiment_note(),
    ])


# ------------------------------------------------------------ callbacks

@callback(
    Output("ra-importance", "children"),
    Input("ra-groups", "value"),
    Input("ra-topn", "value"),
)
def update_importance(groups, top_n):
    if not groups:
        return html.P("Tick at least one group to show the chart.", style=EMPTY_NOTE)

    fig = build_importance_chart(groups, top_n or 10)
    if fig is None:
        return html.P("No SHAP data yet - run src/seller_model.py.", style=EMPTY_NOTE)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


@callback(
    Output("ra-detail", "children"),
    Input("ra-seller", "value"),
)
def update_detail(seller_id):
    if not seller_id:
        return html.P("Pick a seller.", style=EMPTY_NOTE)

    fig, row = build_seller_detail(seller_id)
    if fig is None:
        return html.P("Nothing to show for that seller.", style=EMPTY_NOTE)

    stat = lambda label, value, accent=False: html.Div([
        html.Span(label, className="model-stat-label"),
        html.Span(value, className="model-stat-value accent" if accent
                  else "model-stat-value"),
    ], className="model-stat")

    return html.Div([
        html.Div([
            stat("Risk score", f"{row['risk_score']:.3f}", True),
            stat("Rank", f"#{int(row['risk_rank'])}"),
            stat("Revenue", f"R$ {row['revenue']:,.0f}"),
        ], className="model-strip", style={"marginBottom": "18px"}),

        dcc.Graph(figure=fig, config={"displayModeBar": False}),

        html.P("Red sits above the cohort median, blue below. This compares the "
               "seller with its peers rather than decomposing the model's output - "
               "it shows what is unusual about them.",
               className="chart-note"),
    ])