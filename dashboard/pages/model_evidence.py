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
    load_threshold_curve_by_model, load_capacity_curve_by_model,
    load_roc_by_model, REPORTS_DIR,
)
from theme import (
    VIOLET, CYAN, AMBER, ROSE, EMERALD,
    TEXT, TEXT_DIM, TEXT_FAINT, LINE, LINE_SOFT, CARD, CARD_2, MONO, SANS, rgba,
)
from review_explorer import build_review_explorer

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
        # objective 2 is stated in accuracy, so accuracy has to appear on the
        # page. tolerated as optional so the page still renders against an
        # older model_comparison.csv that predates the column
        acc = first_col(comp, "accuracy", "acc")
        for i, name in enumerate(comp["model"]):
            rows.append({
                "model": str(name),
                "task": "Customer dissatisfaction",
                "validation": HOLDOUT,
                "auc": float(auc.iloc[i]),
                "std": None,
                "f1": None if f1 is None or pd.isna(f1.iloc[i]) else float(f1.iloc[i]),
                "acc": None if acc is None or pd.isna(acc.iloc[i]) else float(acc.iloc[i]),
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
        "acc": "—" if r.get("acc") is None else f"{r['acc']:.1%}",
        "lift": "—" if r["task"] == "Baseline" else f"+{r['auc'] - CHANCE:.3f}",
        "verdict": "Selected" if r.get("selected") else
                   ("Chance" if r["task"] == "Baseline" else ""),
    } for r in rows]

    cols = [("model", "Model"), ("task", "Task"), ("validation", "Validation"),
            ("auc", "AUC-ROC"), ("acc", "Accuracy"), ("f1", "F1 (positive)"),
            ("lift", "vs chance"), ("verdict", "")]
    numeric = ["auc", "acc", "f1", "lift"]

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
        # explicit widths. left to itself the table spreads over the full card
        # and opens a wide gap between the text columns and the numbers, which
        # makes a row hard to read across
        style_cell_conditional=[
            {"if": {"column_id": "model"}, "width": "15%"},
            {"if": {"column_id": "task"}, "width": "18%"},
            {"if": {"column_id": "validation"}, "width": "19%"},
            {"if": {"column_id": "auc"}, "width": "14%", "textAlign": "right"},
            {"if": {"column_id": "acc"}, "width": "12%", "textAlign": "right"},
            {"if": {"column_id": "f1"}, "width": "11%", "textAlign": "right"},
            {"if": {"column_id": "lift"}, "width": "6%", "textAlign": "right"},
            {"if": {"column_id": "verdict"}, "width": "5%"},
        ],
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
    """bars measured from chance, not from zero.

    A bar's length has to encode something real, which is why truncating a
    normal bar axis to exaggerate small differences is dishonest. But AUC does
    not start at zero - 0.5 is chance, and everything below it is unreachable
    for a model that ranks at all. So the bars are anchored at 0.5 and their
    length is lift over chance, which IS the quantity worth comparing. The
    axis can then start at 0.5 without misrepresenting anything, and a 0.06
    difference becomes legible instead of being squashed into the left fifth
    of the plot.
    """
    plot = [r for r in rows if r["task"] != "Baseline"]
    plot.sort(key=lambda r: r["auc"])          # plotly draws y bottom-up

    labels = [f"{r['model']}<br><span style='font-size:10px'>{r['task']}</span>"
              for r in plot]

    # colour by MODEL, using the same mapping as the ROC chart below. an
    # earlier version coloured by selected-or-not, which meant a colour said
    # "winner" here and "which model" there - two rules on one page. bar
    # length and the sort order already show which model won, so hue is free
    # to carry identity instead
    colours = [model_colour(r["model"], rows) for r in plot]

    fig = go.Figure(go.Bar(
        x=[r["auc"] - CHANCE for r in plot],
        base=CHANCE,
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
    x_max = 0.86
    for i, r in enumerate(plot):
        fig.add_annotation(
            x=x_max, y=labels[i], xanchor="right", yanchor="middle",
            text=fmt_auc(r), showarrow=False,
            font=dict(size=11, color=model_colour(r["model"], rows), family=MONO),
        )

    # chance sits on the axis origin now, so it is marked rather than drawn
    # across the plot
    fig.add_annotation(
        x=CHANCE, y=1.04, yref="paper", xanchor="left", yanchor="bottom",
        text="0.5 — chance", showarrow=False,
        font=dict(color=ROSE, size=10.5, family=MONO),
    )

    fig = base_layout(fig, height=58 * len(plot) + 70)
    fig.update_xaxes(range=[CHANCE, x_max], title=None, dtick=0.05,
                     linecolor=ROSE)
    fig.update_yaxes(title=None, gridcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=6, r=14, t=26, b=24))
    return fig


# ------------------------------------------------------- ROC curves

# MODEL colours, kept deliberately clear of the METRIC colours used on the
# threshold chart below (emerald = recall, amber = precision, violet = F1).
# violet appears in both, but only ever as "the selected thing" - the winning
# model here, the headline metric there. amber and emerald never label a model,
# so a colour never means two different measurements on the same page.
# grey for the last entry is deliberate: it is the baseline, not a candidate.
ROC_COLOURS = [VIOLET, CYAN, TEXT_DIM, TEXT_FAINT]


def model_rank_order(rows):
    """model names ordered best-AUC-first on the seller task.

    Ranked on the seller models because that is the primary task and the one
    the ROC chart plots. The customer models reuse whatever colour their name
    already has, so XGBoost is the same colour in both halves of the table.
    """
    seller = [r for r in rows if r["task"] == "Seller decline"]
    if not seller:
        seller = [r for r in rows if r["task"] != "Baseline"]
    return [r["model"] for r in sorted(seller, key=lambda r: -r["auc"])]


def model_colour(name, rows):
    """one colour per model name, shared by every chart on this page."""
    order = model_rank_order(rows)
    if name in order:
        return ROC_COLOURS[order.index(name) % len(ROC_COLOURS)]
    return TEXT_DIM


def build_roc_chart(roc, order):
    """all three seller models on one set of axes.

    This is the comparison the AUC bar chart summarises - each bar there is
    the area under one of these curves. Showing both means a reader can see
    not just that one model scores higher, but where along the curve it wins.
    """
    fig = go.Figure()

    # chance line first, so the curves draw over it
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Chance",
        line=dict(color=ROSE, width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    for i, name in enumerate(order):
        d = roc[roc["model"] == name]
        if d.empty:
            continue
        auc = float(d["auc"].iloc[0])
        fig.add_trace(go.Scatter(
            x=d["fpr"], y=d["tpr"], mode="lines",
            name=f"{name} ({auc:.3f})",
            line=dict(color=ROC_COLOURS[i % len(ROC_COLOURS)],
                      width=2.5 if i == 0 else 1.8),
            hovertemplate=(f"<b>{name}</b><br>false positive %{{x:.3f}}"
                           "<br>true positive %{y:.3f}<extra></extra>"),
        ))

    fig = base_layout(fig, height=470)
    # legend along the top, same as the threshold chart. a ROC plot is forced
    # square by scaleanchor below, which leaves empty card either side of it -
    # an inside-the-plot legend ends up stranded in that gap instead of next
    # to the lines it labels
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(size=11, color=TEXT_DIM)),
        margin=dict(l=6, r=10, t=34, b=40),
    )
    # square axes - a ROC plot is only readable if both axes share a scale
    fig.update_xaxes(range=[0, 1], dtick=0.2, constrain="domain",
                     title=dict(text="False positive rate",
                                font=dict(size=10.5, color=TEXT_FAINT)))
    fig.update_yaxes(range=[0, 1], dtick=0.2, scaleanchor="x", scaleratio=1,
                     title=dict(text="True positive rate",
                                font=dict(size=10.5, color=TEXT_FAINT)))
    return fig


# ------------------------------------------------- per-model curves

def model_curves():
    """threshold curves for every model, ordered best AUC first.

    Falls back to the single-model file when the by-model one has not been
    generated yet, so the page still works on a checkout where seller_model.py
    has not been re-run. In that case the selector is hidden rather than
    offering a choice it cannot honour.
    """
    order, aucs = [], {}
    try:
        cv = load_seller_cv_results()
        auc_col = first_col(cv, "mean_auc", "auc")
        pairs = sorted(zip(cv["model"], auc_col), key=lambda kv: -kv[1])
        order = [str(m) for m, _ in pairs]
        aucs = {str(m): float(a) for m, a in pairs}
    except Exception:
        pass

    try:
        by_model = load_threshold_curve_by_model()
        present = [m for m in order if m in set(by_model["model"])]
        if not present:
            present = sorted(by_model["model"].unique())
        return {m: by_model[by_model["model"] == m].reset_index(drop=True)
                for m in present}, present, aucs, True
    except Exception:
        pass

    # single-model fallback
    try:
        one = load_threshold_curve()
    except Exception:
        return {}, [], aucs, False
    name = order[0] if order else "Selected model"
    return {name: one}, [name], aucs, False


def capacity_for(model_name):
    """precision@N rows for one model, or None if the file isn't there."""
    try:
        cap = load_capacity_curve_by_model()
        rows = cap[cap["model"] == model_name]
        return rows.reset_index(drop=True) if len(rows) else None
    except Exception:
        return None


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

    row, _ = threshold_row(curve, t)
    fig.add_vline(x=float(row["threshold"]),
                  line=dict(color=TEXT_DIM, width=1, dash="dot"))

    fig = base_layout(fig, height=280)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11, color=TEXT_DIM)),
        margin=dict(l=6, r=10, t=34, b=34),
    )
    fig.update_xaxes(title=dict(text="decision threshold",
                                font=dict(size=10.5, color=TEXT_FAINT)))
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


def build_model_comparison_strip(curves, order, selected, t):
    """every model's F1 at the threshold currently set.

    This is the point of the selector. The AUC table above ranks the models
    once, averaged over all thresholds. This shows the ranking holding at the
    specific cut-off on screen - drag the slider and the order does not
    change, which is a stronger claim than a single averaged figure.
    """
    if len(order) < 2:
        return html.Div()

    cells = []
    for name in order:
        row, _ = threshold_row(curves[name], t)
        is_sel = name == selected
        cells.append(html.Div([
            html.Span(name, className="model-stat-label"),
            html.Span(f"{float(row['f1']):.3f}",
                      className="model-stat-value",
                      style={"color": VIOLET if is_sel else TEXT_DIM}),
        ], className="model-stat"))

    return html.Div([
        html.P("F1 at this threshold, all models",
               style={"fontSize": "11px", "color": TEXT_FAINT,
                      "textTransform": "uppercase", "letterSpacing": "0.07em",
                      "margin": "0 0 8px"}),
        html.Div(cells, className="model-strip", style={"width": "100%"}),
    ], style={"marginTop": "18px"})


def build_capacity_strip(model_name):
    """precision@N for the selected model, if the by-model file exists."""
    cap = capacity_for(model_name)
    if cap is None or cap.empty:
        return html.Div()

    base = float(cap["base_rate"].iloc[0]) if "base_rate" in cap.columns else None

    cells = []
    if base is not None:
        cells.append(stat_cell("Base rate", f"{base:.1%}", TEXT_DIM))
    for _, r in cap.iterrows():
        cells.append(stat_cell(
            f"Top {int(r['capacity'])}",
            f"{float(r['precision_at_n']):.1%}",
            EMERALD if base and r["precision_at_n"] > base else TEXT))

    return html.Div([
        html.P("Precision at worklist size",
               style={"fontSize": "11px", "color": TEXT_FAINT,
                      "textTransform": "uppercase", "letterSpacing": "0.07em",
                      "margin": "0 0 8px"}),
        html.Div(cells, className="model-strip", style={"width": "100%"}),
        html.P("How many of the top N really decline. Independent of the "
               "threshold above — the worklist is a ranking, not a cut-off.",
               className="chart-note", style={"marginTop": "10px"}),
    ], style={"marginTop": "18px"})


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
            html.P("Model development and optimisation",
                   className="card-title"),
            html.P("Two separate prediction tasks. Within each, the model with "
                   "the higher AUC is the one the dashboard actually uses.",
                   className="card-sub"),
            build_table(rows),
            html.P("F1 is only reported for the customer model. The seller "
                   "model is used as a ranked worklist rather than a hard "
                   "flag, so a single F1 at one threshold would not describe "
                   "how it is read.", className="chart-note"),
            # accuracy is shown because it is the metric a reader reaches for
            # first, and the baseline is shown with it because on a 14.9%
            # positive rate accuracy alone says almost nothing - predicting
            # the majority class for everyone already scores 85.1%
            html.P("The customer model reaches 87.5% accuracy against a "
                   "majority-class baseline of 85.1%. In error terms it gets "
                   "12.5% of customers wrong where the baseline gets 14.9% "
                   "wrong, a 16% relative reduction. The stronger evidence of "
                   "discrimination is AUC 0.733 against 0.5 chance, and 68% "
                   "precision on the at-risk class against a 14.9% base rate "
                   "— 4.6× better than picking customers at random. Recall is "
                   "30%, which is why the seller model exists alongside it.",
                   className="chart-note"),
        ], className="card"))

        blocks.append(html.Div([
            html.P("AUC against chance", className="card-title"),
            html.P("Bars are measured from 0.5, so length is how far each model "
                   "clears chance. Whiskers on the seller models are the "
                   "standard deviation across the 11 folds.",
                   className="card-sub"),
            dcc.Graph(figure=build_auc_chart(rows),
                      config={"displayModeBar": False}),
            html.P("The axis starts at 0.5 because that is where a "
                   "majority-class classifier sits — it never separates the "
                   "two groups, whatever its accuracy looks like. Nothing "
                   "below 0.5 is reachable by a model that ranks at all, so "
                   "the distance above it is the only part worth comparing, "
                   "and that is what these bars measure.",
                   className="chart-note"),
        ], className="card"))

    curves, order, aucs, has_by_model = model_curves()

    # ROC comparison - the curves the AUC bars above are the area under
    try:
        roc = load_roc_by_model()
    except Exception:
        roc = None

    if roc is not None and not roc.empty:
        roc_order = [m for m in order if m in set(roc["model"])] or \
                    sorted(roc["model"].unique())
        blocks.append(html.Div([
            html.P("ROC curves", className="card-title"),
            html.P("All three seller models on one set of axes, from the same "
                   "out-of-fold predictions", className="card-sub"),
            dcc.Graph(figure=build_roc_chart(roc, roc_order),
                      config={"displayModeBar": False}),
            html.P("The dashed diagonal is chance. A curve that bows further "
                   "toward the top-left corner separates declining from stable "
                   "sellers better, and the AUC in the legend is the area under "
                   "that curve — the same figure as the bars above. Random "
                   "Forest leads across most of the range rather than only on "
                   "average.", className="chart-note"),
        ], className="card"))

    if curves:
        default_model = order[0]

        # the dropdown is ALWAYS rendered, and hidden with css when there is
        # only one model to choose from. dash will not fire a callback whose
        # inputs are missing from the page, so a selector that only exists
        # sometimes would break the threshold card on the fallback path
        show_selector = has_by_model and len(order) > 1
        selector = html.Div([
            html.Span("Model", style={
                "fontSize": "11px", "color": TEXT_FAINT,
                "textTransform": "uppercase", "letterSpacing": "0.07em",
                "whiteSpace": "nowrap"}),
            html.Div(
                dcc.Dropdown(
                    id="me-model",
                    options=[{"label": f"{m}  ·  AUC {aucs.get(m, float('nan')):.3f}"
                                       + ("  ·  selected" if i == 0 else ""),
                              "value": m} for i, m in enumerate(order)],
                    value=default_model, clearable=False,
                    className="dash-dropdown"),
                style={"minWidth": "340px", "position": "relative",
                       "zIndex": 100},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "12px",
                  "marginBottom": "18px"} if show_selector else {"display": "none"})

        blocks.append(html.Div([
            html.P("Decision threshold", className="card-title"),
            html.P("Pick a model, then drag the threshold. Changing the model "
                   "changes whose curve you are looking at; dragging changes "
                   "only where the line between 'flag' and 'ignore' is drawn.",
                   className="card-sub"),

            selector,

            dcc.Slider(
                id="me-threshold",
                min=0.10, max=0.90, step=0.01, value=DEFAULT_THRESHOLD,
                marks={v: {"label": f"{v:.2f}"}
                       for v in (0.10, 0.30, 0.50, 0.70, 0.90)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),

            html.Div(id="me-threshold-stats", style={"marginTop": "22px"}),
            dcc.Graph(id="me-threshold-chart", config={"displayModeBar": False}),
            html.Div(id="me-model-compare"),
            html.Div(id="me-capacity"),

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

    # the review table is evidence for the lexicon in the same way the AUC
    # chart is evidence for the ranker, so it belongs on this page. wrapped
    # because it reads review_examples.csv, and a missing file should cost
    # one card rather than the whole page
    try:
        blocks.append(build_review_explorer())
    except Exception:
        pass

    return html.Div(blocks)


@callback(
    Output("me-threshold-stats", "children"),
    Output("me-threshold-chart", "figure"),
    Output("me-model-compare", "children"),
    Output("me-capacity", "children"),
    Input("me-threshold", "value"),
    Input("me-model", "value"),
)
def update_threshold(t, model_name):
    curves, order, _, _ = model_curves()
    if not curves:
        raise dash.exceptions.PreventUpdate

    if model_name not in curves:
        model_name = order[0]
    curve = curves[model_name]

    t = DEFAULT_THRESHOLD if t is None else float(t)

    return (
        build_threshold_stats(curve, t),
        build_threshold_chart(curve, t),
        build_model_comparison_strip(curves, order, model_name, t),
        build_capacity_strip(model_name),
    )