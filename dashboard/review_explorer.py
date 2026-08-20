"""
review_explorer.py - the raw reviews behind the sentiment claim.

A drop-in card, not a page. Import it into whichever page should carry it
and call build_review_explorer() in that page's layout - the callbacks
register themselves on import.

    from review_explorer import build_review_explorer
    ...
    build_review_explorer(),

WHY A MODAL
The comment text is the evidence, and it is long, Portuguese, and unreadable
squeezed into a table cell. So the table holds the scores - which is what you
scan - and clicking a row opens the full comment with both scores against it.
That is the demo moment: a one star review saying the product never arrived,
TextBlob calling it 0.0, and the induced lexicon calling it negative.

SIGN CONVENTION - read this before touching the numbers
nlp_lexicon.py builds the lexicon so that a POSITIVE weight marks a word that
appears in NEGATIVE reviews (see the direction column in nlp_lexicon.csv:
"negative" if s > 0). score_comment returns the mean weight of the words that
fired, so a lex_score above zero means the comment reads as negative. It is
the opposite of TextBlob's convention, where positive means positive. Both raw
numbers are shown, each with the direction spelled out in words, because
showing two oppositely signed scores side by side without saying so would be
worse than showing neither.

all styling is inline, so this drops in without touching theme.py
"""

import sys
from pathlib import Path

import pandas as pd
from dash import html, dcc, dash_table, callback, Input, Output, State, no_update

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_review_examples
from theme import (
    VIOLET, CYAN, AMBER, ROSE, EMERALD, TEXT, TEXT_DIM, TEXT_FAINT,
    LINE, LINE_SOFT, CARD, CARD_2, MONO, SANS,
)

# TextBlob's own convention, and the one sentiment_validation.py uses
POLARITY_CUTOFF = 0.1

# the lexicon's neutral band. nlp_lexicon.py tunes its cutoff on the training
# split and doesnt persist it, so this is presentation only - it decides the
# wording next to the number, never a reported metric
LEX_CUTOFF = 0.1

STAR_COLOUR = {1: ROSE, 2: ROSE, 3: AMBER, 4: EMERALD, 5: EMERALD}

MODAL_HIDDEN = {"display": "none"}
MODAL_SHOWN = {
    "display": "flex", "position": "fixed", "inset": "0", "zIndex": "1000",
    "background": "rgba(6,7,10,0.72)", "alignItems": "center",
    "justifyContent": "center", "padding": "24px",
}

# the panel lives in the layout permanently and is only hidden, rather than
# being built inside the callback. dash will not fire a callback when any of
# its inputs is absent from the page, so a close button that only exists once
# the modal is open can never be the thing that closes it
PANEL = {
    "position": "relative",
    "background": CARD, "border": f"1px solid {LINE}",
    "borderRadius": "14px", "padding": "22px 24px",
    "maxWidth": "620px", "width": "100%",
    "maxHeight": "86vh", "overflowY": "auto",
}

CLOSE_BTN = {
    "position": "absolute", "top": "16px", "right": "18px",
    "background": "transparent", "border": "none", "color": TEXT_FAINT,
    "fontSize": "16px", "cursor": "pointer", "padding": "2px 6px",
    "lineHeight": "1", "zIndex": "2",
}


def textblob_reading(p):
    if p > POLARITY_CUTOFF:
        return "reads as positive", EMERALD
    if p < -POLARITY_CUTOFF:
        return "reads as negative", ROSE
    if p == 0:
        return "no sentiment words recognised", TEXT_FAINT
    return "reads as neutral", TEXT_DIM


def lex_reading(s):
    # remember: positive score = negative sentiment. see the module docstring
    if s is None or pd.isna(s):
        return "no lexicon words in this comment", TEXT_FAINT
    if s > LEX_CUTOFF:
        return "reads as negative", ROSE
    if s < -LEX_CUTOFF:
        return "reads as positive", EMERALD
    return "reads as neutral", TEXT_DIM


def snippet(text, n=88):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


# -------------------------------------------------------------- table

def build_table(df):
    t = df.copy()
    # dash_table treats an "id" field in each record as the row id, and hands
    # it back on active_cell as row_id. that is the only stable handle once
    # the user sorts or pages - active_cell["row"] is a position in the
    # current view and points at a different review after every sort
    t["id"] = t["review_id"].astype(str)
    t["stars"] = t["review_score"].map(lambda v: "★" * int(v))
    t["snippet"] = t["review_comment_message"].map(snippet)
    t["tb_f"] = t["textblob_polarity"].map(lambda v: f"{v:+.3f}")
    t["lex_f"] = t["lex_score"].map(
        lambda v: "—" if pd.isna(v) else f"{v:+.3f}")

    cols = [("stars", "Rating"), ("snippet", "Comment"),
            ("tb_f", "TextBlob"), ("lex_f", "Induced lexicon")]

    return dash_table.DataTable(
        id="rev-table",
        data=t[[c for c, _ in cols] + ["review_score", "id"]].to_dict("records"),
        columns=[{"name": lbl, "id": cid} for cid, lbl in cols],
        sort_action="native",
        page_size=8,
        style_as_list_view=True,
        cell_selectable=True,
        style_header={"backgroundColor": "rgba(0,0,0,0)", "color": TEXT_FAINT,
                      "fontWeight": "500", "fontSize": "11px", "border": "none",
                      "borderBottom": f"1px solid {LINE}",
                      "padding": "0 14px 11px 0", "textAlign": "left"},
        style_cell={"backgroundColor": "rgba(0,0,0,0)", "color": TEXT,
                    "border": "none", "padding": "11px 14px 11px 0",
                    "fontSize": "12.5px", "fontFamily": SANS, "textAlign": "left",
                    "maxWidth": 0, "overflow": "hidden",
                    "textOverflow": "ellipsis", "cursor": "pointer"},
        # percentages, not pixels. style_cell sets maxWidth 0 so the comment
        # column ellipsises instead of stretching the table, and a fixed pixel
        # width on the score columns collapses to nothing under that rule
        style_cell_conditional=[
            {"if": {"column_id": "stars"}, "width": "9%", "minWidth": "62px"},
            {"if": {"column_id": "snippet"}, "width": "55%"},
            {"if": {"column_id": "tb_f"},
             "width": "18%", "minWidth": "92px", "textAlign": "right"},
            {"if": {"column_id": "lex_f"},
             "width": "18%", "minWidth": "104px", "textAlign": "right"},
        ],
        style_data_conditional=[
            {"if": {"column_id": ["tb_f", "lex_f"]},
             "fontFamily": MONO, "fontSize": "12px"},
            {"if": {"column_id": "snippet"}, "color": TEXT_DIM},
            {"if": {"filter_query": "{review_score} <= 2", "column_id": "stars"},
             "color": ROSE},
            {"if": {"filter_query": "{review_score} = 3", "column_id": "stars"},
             "color": AMBER},
            {"if": {"filter_query": "{review_score} >= 4", "column_id": "stars"},
             "color": EMERALD},
            {"if": {"filter_query": "{tb_f} = '+0.000'", "column_id": "tb_f"},
             "color": TEXT_FAINT},
        ],
    )


# -------------------------------------------------------------- modal

def score_block(label, value, reading, colour):
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": TEXT_FAINT,
                               "textTransform": "uppercase",
                               "letterSpacing": "0.08em", "marginBottom": "7px"}),
        html.Div(value, style={"fontFamily": MONO, "fontSize": "21px",
                               "color": colour, "fontWeight": "500"}),
        html.Div(reading, style={"fontSize": "11.5px", "color": TEXT_DIM,
                                 "marginTop": "5px"}),
    ], style={"flex": "1 1 0", "padding": "14px 16px", "background": CARD_2,
              "borderRadius": "10px", "minWidth": "0"})


def build_modal_body(row):
    stars = int(row["review_score"])
    tb = float(row["textblob_polarity"])
    lex = None if pd.isna(row["lex_score"]) else float(row["lex_score"])

    tb_read, tb_colour = textblob_reading(tb)
    lex_read, lex_colour = lex_reading(lex)

    words = str(row.get("lex_words") or "").strip()

    body = [
        html.Div([
            html.Span("★" * stars, style={"color": STAR_COLOUR.get(stars, TEXT),
                                          "fontSize": "16px"}),
            html.Span(f"{stars} of 5", style={"color": TEXT_FAINT,
                                              "fontSize": "12px",
                                              "marginLeft": "10px"}),
        ], style={"marginBottom": "16px", "paddingRight": "28px"}),

        html.Div("The comment, as written", style={
            "fontSize": "10px", "color": TEXT_FAINT, "textTransform": "uppercase",
            "letterSpacing": "0.08em", "marginBottom": "8px"}),
        html.P(str(row["review_comment_message"]), style={
            "color": TEXT, "fontSize": "14px", "lineHeight": "1.65",
            "margin": "0 0 20px", "padding": "16px 18px",
            "background": CARD_2, "borderRadius": "10px",
            "borderLeft": f"2px solid {LINE}", "whiteSpace": "pre-wrap"}),

        html.Div([
            score_block("TextBlob polarity", f"{tb:+.3f}", tb_read, tb_colour),
            score_block("Induced lexicon",
                        "—" if lex is None else f"{lex:+.3f}",
                        lex_read, lex_colour),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),
    ]

    if words:
        body.append(html.Div([
            html.Div("Lexicon words that fired", style={
                "fontSize": "10px", "color": TEXT_FAINT,
                "textTransform": "uppercase", "letterSpacing": "0.08em",
                "marginBottom": "8px"}),
            html.Div([
                html.Span(w.strip(), style={
                    "display": "inline-block", "padding": "4px 10px",
                    "marginRight": "6px", "marginBottom": "6px",
                    "borderRadius": "14px", "fontSize": "12px",
                    "background": CARD_2, "color": VIOLET,
                    "border": f"1px solid {LINE_SOFT}"})
                for w in words.split(",") if w.strip()]),
        ], style={"marginBottom": "16px"}))

    body.append(html.P(
        "TextBlob scores above zero as positive; the induced lexicon was built "
        "the other way up, so above zero there means negative. Both raw values "
        "are shown unchanged — the wording under each one is the translation.",
        style={"color": TEXT_FAINT, "fontSize": "11.5px", "lineHeight": "1.55",
               "margin": "0"}))

    return body


# ------------------------------------------------------------- layout

def build_review_explorer():
    """the card. returns an empty Div if review_examples.csv isnt there yet,
    so a page carrying this doesnt break before the script has been run."""
    try:
        df = load_review_examples()
    except Exception:
        return html.Div()

    return html.Div([
        html.Div([
            html.P("The reviews themselves", className="card-title"),
            html.P("Click any row to read the full comment with both scores "
                   "against it", className="card-sub"),

            dcc.Checklist(
                id="rev-only-zero",
                options=[{"label": "  Only comments TextBlob scored exactly 0.0",
                          "value": "zero"}],
                value=[],
                style={"marginBottom": "14px"},
                # the label colour has to go on labelStyle - dash 4 renders the
                # text inside its own <label>, which does not inherit a colour
                # set on the Checklist wrapper
                labelStyle={"fontSize": "12.5px", "color": TEXT_DIM,
                            "cursor": "pointer", "display": "inline-flex",
                            "alignItems": "center"},
                inputStyle={"accentColor": VIOLET, "width": "13px",
                            "height": "13px", "cursor": "pointer"},
            ),

            html.Div(id="rev-table-wrap", children=build_table(df)),

            html.P(
                "A stratified sample across the five star ratings, drawn with a "
                "fixed seed — not filtered to the cases that favour either "
                "method. The 0.0 rows are a filter, not the sample.",
                className="chart-note"),
        ], className="card"),

        html.Div(html.Div([
            html.Button("✕", id="rev-close", n_clicks=0, style=CLOSE_BTN),
            html.Div(id="rev-modal-body"),
        ], style=PANEL), id="rev-modal", style=MODAL_HIDDEN),
    ])


# ---------------------------------------------------------- callbacks

@callback(
    Output("rev-table-wrap", "children"),
    Input("rev-only-zero", "value"),
)
def filter_table(value):
    df = load_review_examples()
    if value and "zero" in value:
        df = df[df["textblob_zero"].astype(bool)]
    if df.empty:
        return html.P("No reviews match that filter.",
                      style={"color": TEXT_FAINT, "fontSize": "13px",
                             "padding": "18px 0"})
    return build_table(df)


@callback(
    Output("rev-modal-body", "children"),
    Output("rev-modal", "style"),
    Input("rev-table", "active_cell"),
    Input("rev-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_modal(active_cell, close_clicks):
    from dash import ctx

    if ctx.triggered_id == "rev-close":
        return no_update, MODAL_HIDDEN

    if not active_cell:
        return no_update, MODAL_HIDDEN

    review_id = active_cell.get("row_id")
    if review_id is None:
        return no_update, MODAL_HIDDEN

    # the table only carries the display columns, so go back to the source
    # for the full comment text and the words that fired
    df = load_review_examples()
    match = df[df["review_id"].astype(str) == str(review_id)]
    if match.empty:
        return no_update, MODAL_HIDDEN

    return build_modal_body(match.iloc[0]), MODAL_SHOWN