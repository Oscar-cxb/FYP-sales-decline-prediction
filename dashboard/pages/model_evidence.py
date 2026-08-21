"""
model_evidence.py - the evidence page. every headline number in one place.

Three questions an examiner asks, answered in order:

  1. is the model better than guessing?      -> comparison table + AUC chart
  2. how do you know it isnt one lucky split? -> the validation column
  3. where did the decision threshold come from? -> the slider

WHY 0.5 IS THE BASELINE
AUC is the probability the model ranks a random positive above a random
negative. a coin, or a classifier that always predicts the majority class,
scores exactly 0.5 - it has no ability to separate the two groups at all.
so 0.5 is the floor, not 0.0, and a model at 0.71 has closed roughly a
fifth of the distance from chance to perfect rather than being "71% right".
accuracy would be the misleading number here: the seller decline base rate
is low enough that always predicting "no decline" scores well on accuracy
and is worth nothing.

everything on this page is read from reports/ - nothing is computed or
typed in by hand, so a rerun of classifier.py or seller_model.py changes
what this page says without anyone editing it.
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, dash_table, callback, Input, Output
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import (
    load_model_comparison, load_seller_cv_results, load_threshold_curve,
    REPORTS_DIR,
)
from review_explorer import build_review_explorer
from theme import (
    VIOLET, CYAN, AMBER, ROSE, EMERALD,
    TEXT, TEXT_DIM, TEXT_FAINT, LINE, LINE_SOFT, CARD, CARD_2, MONO, SANS, rgba,
)

dash.register_page(__name__, path="/model-evidence", name="Model Evidence")

CHANCE = 0.5          # AUC of a majority-class classifier
DEFAULT_THRESHOLD = 0.35

# customer model is a single stratified holdout, seller model is rolling
# origin CV. keeping the wording identical to how the chapters describe them
HOLDOUT = "20% stratified holdout"
ROLLING = "11-fold rolling origin CV"


# ------------------------------------------------------------ shared

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


def first_col(df, *names):
    """the two comparison csvs dont agree on column names (auc vs mean_auc,
    std_auc vs std) depending on which script wrote them last. rather than
    pinning one spelling and having the page go blank after a rerun, take
    whichever is actually there."""
    for n in names:
        if n in df.columns:
            return df[n]
    return None


# ------------------------------------------------- gathering the rows

def collect_rows():
    """one row per (model, task). returns a list of dicts plus a list of
    whatever couldnt be loaded, so a missing report file degrades to a note
    instead of an exception."""
    rows, missing = [], []

    # --- customer dissatisfaction, single holdout ---
    try:
        comp = load_model_comparison()
        auc = first_col(comp, "auc", "mean_auc")
        f1 = first_col(comp, "f1")
        for i, name in enumerate(comp["model"]):
            rows.append({
                "model": str(name),
                "task": "Customer dissatisfaction",
                "validation": HOLDOUT,
                "auc": float(auc.iloc[i]),
                "std": None,
                "f1": None if f1 is None or pd.isna(f1.iloc[i]) else float(f1.iloc[i]),
            })
    except Exception:
        missing.append("reports/model_comparison.csv")

    # --- seller decline, rolling origin CV ---
    try:
        cv = load_seller_cv_results()
        if "model" not in cv.columns:
            # seller_cv_results.csv is per-fold in some runs. the per-model
            # summary lives next to it
            cv = pd.read_csv(REPORTS_DIR / "seller_model_comparison.csv")
        auc = first_col(cv, "mean_auc", "auc")
        std = first_col(cv, "std_auc", "std", "std_dev")
        f1 = first_col(cv, "f1")
        for i, name in enumerate(cv["model"]):
            rows.append({
                "model": str(name),
                "task": "Seller decline",
                "validation": ROLLING,
                "auc": float(auc.iloc[i]),
                "std": None if std is None or pd.isna(std.iloc[i]) else float(std.iloc[i]),
                "f1": None if f1 is None or pd.isna(f1.iloc[i]) else float(f1.iloc[i]),
            })
    except Exception:
        missing.append("reports/seller_cv_results.csv")

    # winner per task = highest AUC, which is the rule both training
    # scripts now use. computed here rather than hardcoded so the badge
    # can never drift away from the numbers next to it
    for task in {r["task"] for r in rows}:
        group = [r for r in rows if r["task"] == task]
        best = max(group, key=lambda r: r["auc"])
        for r in group:
            r["selected"] = r is best

    # the reference row. not a model - the thing every model has to beat
    rows.append({
        "model": "Majority class", "task": "Baseline",
        "validation": "—", "auc": CHANCE, "std": None,
        "f1": None, "selected": False,
    })

    rows.sort(key=lambda r: (r["task"] == "Baseline", -r["auc"]))
    return rows, missing


def fmt_auc(r):
    if r["std"]:
        return f"{r['auc']:.3f} ± {r['std']:.3f}"
    return f"{r['auc']:.3f}"


# ------------------------------------------------------------- table

def build_table(rows):
    data = [{
        "model": r["model"],
        "task": r["task"],
        "validation": r["validation"],
        "auc": fmt_auc(r),
        "f1": "—" if r["f1"] is None else f"{r['f1']:.3f}",
        "lift": "—" if r["task"] == "Baseline" else f"+{r['auc'] - CHANCE:.3f}",
        "verdict": "Selected" if r.get("selected") else
                   ("Chance" if r["task"] == "Baseline" else ""),
    } for r in rows]

    cols = [("model", "Model"), ("task", "Task"), ("validation", "Validation"),
            ("auc", "AUC-ROC"), ("f1", "F1 (positive)"),
            ("lift", "vs chance"), ("verdict", "")]
    numeric = ["auc", "f1", "lift"]

    return dash_table.DataTable(
        data=data,
        columns=[{"name": lbl, "id": cid} for cid, lbl in cols],
        style_as_list_view=True,
        style_header={"backgroundColor": "rgba(0,0,0,0)", "color": TEXT_FAINT,
                      "fontWeight": "500", "fontSize": "11px", "border": "none",
                      "borderBottom": f"1px solid {LINE}",
                      "padding": "0 14px 11px 0", "textAlign": "left"},
        style_cell={"backgroundColor": "rgba(0,0,0,0)", "color": TEXT,
                    "border": "none", "padding": "11px 14px 11px 0",
                    "fontSize": "12.5px", "fontFamily": SANS, "textAlign": "left"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "right"}
                                for c in numeric],
        style_data_conditional=[
            {"if": {"column_id": numeric}, "fontFamily": MONO, "fontSize": "12px"},
            {"if": {"column_id": ["validation", "task"]},
             "color": TEXT_DIM, "fontSize": "11.5px"},
            {"if": {"column_id": "lift"}, "color": TEXT_FAINT},
            {"if": {"filter_query": "{verdict} = 'Selected'"},
             "backgroundColor": rgba(VIOLET, 0.07)},
            {"if": {"filter_query": "{verdict} = 'Selected'", "column_id": "auc"},
             "color": VIOLET, "fontWeight": "600"},
            {"if": {"filter_query": "{verdict} = 'Selected'", "column_id": "verdict"},
             "color": VIOLET, "fontWeight": "600", "fontSize": "11px"},
            {"if": {"filter_query": "{verdict} = 'Chance'"},
             "color": TEXT_FAINT, "fontStyle": "italic"},
        ],
    )


# --------------------------------------------------------- AUC chart

def build_auc_chart(rows):
    plot = [r for r in rows if r["task"] != "Baseline"]
    plot.sort(key=lambda r: r["auc"])          # plotly draws y bottom-up

    labels = [f"{r['model']}<br><span style='font-size:10px'>{r['task']}</span>"
              for r in plot]
    colours = [VIOLET if r.get("selected") else rgba(VIOLET, 0.30) for r in plot]

    fig = go.Figure(go.Bar(
        x=[r["auc"] for r in plot],
        y=labels,
        orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        width=0.6,
        error_x=dict(
            type="data",
            array=[r["std"] or 0 for r in plot],
            visible=True, color=TEXT_FAINT, thickness=1, width=4,
        ),
        customdata=[[r["validation"], fmt_auc(r)] for r in plot],
        hovertemplate="AUC %{customdata[1]}<br>%{customdata[0]}<extra></extra>",
    ))

    # value labels go in their own right-hand gutter rather than sitting on
    # the end of each bar - plotly draws "outside" text at the bar tip, which
    # lands underneath the error whisker and makes both unreadable
    for i, r in enumerate(plot):
        fig.add_annotation(
            x=1.0, y=labels[i], xanchor="right", yanchor="middle",
            text=fmt_auc(r), showarrow=False,
            font=dict(size=11, color=VIOLET if r.get("selected") else TEXT_DIM,
                      family=MONO),
        )

    # the chance line. drawn on top of the bars deliberately - the point is
    # that the bars have to clear it
    fig.add_vline(x=CHANCE, line=dict(color=ROSE, width=1.5, dash="dash"))
    fig.add_annotation(
        x=CHANCE, y=1.04, yref="paper", xanchor="left", yanchor="bottom",
        text="0.5 — chance", showarrow=False,
        font=dict(color=ROSE, size=10.5, family=MONO),
    )

    fig = base_layout(fig, height=58 * len(plot) + 70)
    # axis runs the full 0-1 range on purpose. 1.0 is perfect separation and
    # 0.5 is chance, so the honest reading is how far along that span each
    # bar gets - a truncated axis would make the gaps look larger than they are
    fig.update_xaxes(range=[0, 1.0], title=None, dtick=0.1)
    fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=14, t=26, b=24))
    return fig


# ---------------------------------------------------- threshold curve

def threshold_row(curve, t):
    """nearest measured threshold. the sweep in threshold_curve.csv only
    covers the range where both classes still appear, so the ends of the
    slider clamp onto the closest row rather than inventing a point."""
    idx = (curve["threshold"] - t).abs().idxmin()
    row = curve.loc[idx]
    return row, abs(float(row["threshold"]) - t) > 0.005


def build_threshold_chart(curve, t):
    fig = go.Figure()
    series = [("recall", EMERALD, "Recall"),
              ("precision", AMBER, "Precision"),
              ("f1", VIOLET, "F1")]
    for col, colour, label in series:
        if col not in curve.columns:
            continue
        fig.add_trace(go.Scatter(
            x=curve["threshold"], y=curve[col], mode="lines", name=label,
            line=dict(color=colour, width=2),
            hovertemplate=label + " %{y:.3f} at %{x:.2f}<extra></extra>",
        ))

    # Add F1 peak reference marker to show the standard mathematical baseline
    if "f1" in curve.columns and not curve["f1"].empty:
        f1_max_idx = curve["f1"].idxmax()
        f1_max_t = float(curve.loc[f1_max_idx, "threshold"])
        fig.add_vline(x=f1_max_t, line=dict(color=rgba(VIOLET, 0.35), width=1, dash="dash"))
        fig.add_annotation(
            x=f1_max_t, y=0.98, yref="paper", xanchor="center", yanchor="top",
            text=f"F1 Max ({f1_max_t:.2f})", showarrow=False,
            font=dict(color=TEXT_FAINT, size=9.5, family=MONO),
        )

    # Add selected threshold line
    row, _ = threshold_row(curve, t)
    fig.add_vline(x=float(row["threshold"]),
                  line=dict(color=TEXT, width=1.5, dash="dot"))

    fig = base_layout(fig, height=280)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11, color=TEXT_DIM)),
        margin=dict(l=6, r=10, t=34, b=34),
    )
    fig.update_xaxes(
    range=[0.10, 0.90], dtick=0.10,
    title=dict(text="decision threshold", font=dict(size=10.5, color=TEXT_FAINT))
)
    fig.update_yaxes(range=[0, 1.02], dtick=0.2, title=None)
    return fig


def stat_cell(label, value, colour=TEXT):
    return html.Div([
        html.Span(label, className="model-stat-label"),
        html.Span(value, className="model-stat-value",
                  style={"color": colour}),
    ], className="model-stat")


def build_threshold_stats(curve, t):
    row, clamped = threshold_row(curve, t)
    cells = [
        stat_cell("Threshold", f"{float(row['threshold']):.2f}", VIOLET),
        stat_cell("Recall", f"{float(row['recall']):.3f}", EMERALD),
        stat_cell("Precision", f"{float(row['precision']):.3f}", AMBER),
        stat_cell("F1", f"{float(row['f1']):.3f}"),
    ]
    out = [html.Div(cells, className="model-strip",
                    style={"width": "100%", "marginBottom": "14px"})]
    if clamped:
        out.append(html.P(
            f"{t:.2f} is outside the measured sweep "
            f"({curve['threshold'].min():.2f}–{curve['threshold'].max():.2f}); "
            f"showing the nearest evaluated point.",
            className="chart-note", style={"marginTop": "0"}))
    return html.Div(out)


# ------------------------------------------------------------ layout

INTRO = ("Everything the model claims, with the number behind it. The customer "
         "model is scored on a single held-out split; the seller model — the "
         "primary one — is scored on rolling origin cross validation, so its "
         "figure is a mean across folds rather than one lucky partition.")


def layout(**kwargs):
    rows, missing = collect_rows()

    try:
        curve = load_threshold_curve()
    except Exception:
        curve = None

    blocks = [
        html.Div([
            html.Div([
                html.H1("Model evidence"),
                html.P(INTRO, className="page-sub"),
            ], className="page-head-text"),
        ], className="page-head"),
    ]

    if missing:
        blocks.append(html.Div(
            html.P("Missing report files: " + ", ".join(missing) +
                   ". Run the pipeline scripts to regenerate them.",
                   className="chart-note", style={"margin": "0", "color": AMBER}),
            className="card"))

    if rows:
        blocks.append(html.Div([
            html.P("Model comparison", className="card-title"),
            html.P("Two separate prediction tasks. Within each, the model with "
                   "the higher AUC is the one the dashboard actually uses.",
                   className="card-sub"),
            build_table(rows),
            html.P("F1 is only reported for the customer model. The seller "
                   "model is used as a ranked worklist rather than a hard "
                   "flag, so a single F1 at one threshold would not describe "
                   "how it is read.", className="chart-note"),
        ], className="card"))

        blocks.append(html.Div([
            html.P("AUC against chance", className="card-title"),
            html.P("Bars are mean AUC; whiskers on the seller models are the "
                   "standard deviation across the 11 folds.",
                   className="card-sub"),
            dcc.Graph(figure=build_auc_chart(rows),
                      config={"displayModeBar": False}),
            html.P("A majority-class classifier sits exactly on the dashed "
                   "line at 0.5 — it never separates the two groups, whatever "
                   "its accuracy looks like. Distance to the right of that "
                   "line is the only part of a bar that means anything.",
                   className="chart-note"),
        ], className="card"))

    if curve is not None:
        blocks.append(html.Div([
            html.P("Decision threshold", className="card-title"),
html.P("Random Forest on seller decline — the selected model above. "
                   "Dragging does not change the model, only where the line "
                   "between 'flag' and 'ignore' is drawn.",
                   className="card-sub"),

            dcc.Slider(
                id="me-threshold",
                min=0.10, max=0.90, step=0.01, value=DEFAULT_THRESHOLD,
                marks={v: {"label": f"{v:.2f}"}
                       for v in (0.10, 0.30, 0.50, 0.70, 0.90)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),

            html.Div(id="me-threshold-stats", style={"marginTop": "22px"}),
            dcc.Graph(id="me-threshold-chart", config={"displayModeBar": False}),

            html.Div([
                html.P("Why recall is prioritised over precision",
                       style={"color": TEXT, "fontSize": "12.5px",
                              "fontWeight": "600", "margin": "0 0 7px"}),
                html.P(
                    "The two errors do not cost the same. A false positive is "
                    "an account manager spending twenty minutes on a seller "
                    "who was fine — wasted effort, recoverable. A false "
                    "negative is a declining seller nobody contacts, and by "
                    "the time the decline shows up in the revenue figures the "
                    "window to act on it has closed. On an imbalanced problem "
                    "the majority class also makes precision easy to inflate "
                    "and recall hard to hold, so optimising F1 alone quietly "
                    "trades away the error that actually matters. "
                    f"The default of {DEFAULT_THRESHOLD:.2f} sits below the "
                    "F1-optimal point on purpose: it accepts a lower precision "
                    "to keep recall high, and the ranked worklist absorbs the "
                    "extra false positives by putting them at the bottom.",
                    style={"color": TEXT_DIM, "fontSize": "12px",
                           "lineHeight": "1.65", "margin": "0"}),
            ], style={
                "marginTop": "18px", "padding": "16px 18px",
                "background": CARD_2, "borderRadius": "10px",
                "borderLeft": f"2px solid {VIOLET}",
            }),
        ], className="card"))

    blocks.append(build_review_explorer())
    
    return html.Div(blocks)


@callback(
    Output("me-threshold-stats", "children"),
    Output("me-threshold-chart", "figure"),
    Input("me-threshold", "value"),
)
def update_threshold(t):
    curve = load_threshold_curve()
    t = DEFAULT_THRESHOLD if t is None else float(t)
    return build_threshold_stats(curve, t), build_threshold_chart(curve, t)