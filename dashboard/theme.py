"""
theme.py

Palette + the whole stylesheet as one string. Its embedded straight
into app.py's index_string instead of living in assets/style.css -
had trouble with the assets folder not getting picked up, and this
way theres one less thing that can silently not load.

plotly cant read css variables so the colours are python constants
here and get interpolated into the css below. single source of truth.
"""

BG        = "#0a0b0f"
CARD      = "#13151d"
CARD_2    = "#181b25"
LINE      = "#232734"
LINE_SOFT = "#1b1e29"

TEXT      = "#eef0f6"
TEXT_DIM  = "#8b91a5"
TEXT_FAINT= "#5a5f72"

VIOLET    = "#7c5cff"
CYAN      = "#22d3ee"
AMBER     = "#fbbf24"
ROSE      = "#fb7185"
EMERALD   = "#34d399"

# risk bands - low / watch / high
LOW, MID, HIGH = CYAN, AMBER, ROSE

MONO = "ui-monospace, 'SF Mono', Consolas, monospace"
SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def rgba(hex_colour, alpha):
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {{ box-sizing: border-box; }}

html, body {{
    margin: 0; padding: 0;
    background: {BG};
    color: {TEXT};
    font-family: {SANS};
    -webkit-font-smoothing: antialiased;
}}

#react-entry-point, #_dash-app-content {{ background: {BG}; }}

.app-shell {{ display: flex; min-height: 100vh; background: {BG}; }}

/* ---------------- sidebar ---------------- */

.sidebar {{
    width: 232px; flex-shrink: 0;
    background: {CARD};
    border-right: 1px solid {LINE_SOFT};
    padding: 26px 0 22px;
    display: flex; flex-direction: column;
    position: sticky; top: 0; height: 100vh;
}}

.brand {{ display: flex; align-items: center; gap: 11px; padding: 0 22px 30px; }}

.brand-mark {{
    width: 32px; height: 32px; border-radius: 9px; flex-shrink: 0;
    background: linear-gradient(135deg, {VIOLET}, {CYAN});
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: #fff;
}}

.brand-line-1 {{ font-size: 13px; font-weight: 600; letter-spacing: -0.01em; }}
.brand-line-2 {{ font-size: 11px; color: {TEXT_FAINT}; margin-top: 1px; }}

.nav {{ display: flex; flex-direction: column; gap: 2px; padding: 0 12px; }}

.nav-link {{
    color: {TEXT_DIM}; text-decoration: none;
    padding: 10px 14px; font-size: 13.5px; font-weight: 500;
    border-radius: 9px;
    transition: background 130ms ease, color 130ms ease;
}}

.nav-link:hover {{ color: {TEXT}; background: {CARD_2}; }}
.nav-link:focus-visible {{ outline: 2px solid {VIOLET}; outline-offset: -2px; }}

.sidebar-footer {{
    margin-top: auto; padding: 16px 22px 0;
    border-top: 1px solid {LINE_SOFT};
    font-family: {MONO}; font-size: 10px;
    letter-spacing: 0.05em; color: {TEXT_FAINT};
}}

/* ---------------- content ---------------- */

.content {{ flex: 1; padding: 32px 34px 56px; max-width: 1500px; }}

.page-head {{ margin-bottom: 24px; }}

h1 {{ font-size: 22px; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 7px; }}

.page-sub {{
    color: {TEXT_DIM}; font-size: 13px; line-height: 1.6;
    margin: 0; max-width: 660px;
}}

.card-title {{
    font-size: 13px; font-weight: 600; color: {TEXT};
    margin: 0 0 2px;
}}

.card-sub {{ font-size: 11.5px; color: {TEXT_FAINT}; margin: 0 0 16px; }}

.chart-note {{
    color: {TEXT_FAINT}; font-size: 11.5px; line-height: 1.55;
    margin: 14px 0 0; max-width: 580px;
}}

/* ---------------- KPI ---------------- */

.kpi-row {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin-bottom: 14px;
}}

.kpi-card {{
    background: {CARD};
    border: 1px solid {LINE_SOFT};
    border-radius: 14px;
    padding: 18px 18px 16px;
    position: relative; overflow: hidden;
    height: 100%;
    display: flex; flex-direction: column;
}}

.kpi-top {{ display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }}

.kpi-chip {{
    width: 30px; height: 30px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; flex-shrink: 0;
}}

.kpi-label {{
    color: {TEXT_DIM}; font-size: 11.5px; font-weight: 500;
    letter-spacing: 0.01em;
}}

.kpi-value {{
    font-size: 27px; font-weight: 600; letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums; line-height: 1.1;
}}

.kpi-sub {{ font-size: 11px; color: {TEXT_FAINT}; margin-top: 7px; margin-bottom: 14px; }}

.kpi-bar {{
    height: 4px; border-radius: 3px;
    background: {LINE}; margin-top: auto; overflow: hidden;
}}

.kpi-bar-fill {{ height: 100%; border-radius: 3px; }}

/* ---------------- clickable KPI ---------------- */

.kpi-link {{ text-decoration: none; color: inherit; display: block; height: 100%; }}

.kpi-link .kpi-card {{
    cursor: pointer;
    transition: border-color 140ms ease, transform 140ms ease, background 140ms ease;
}}

.kpi-link:hover .kpi-card {{
    border-color: {LINE};
    background: {CARD_2};
    transform: translateY(-2px);
}}

.kpi-link:focus-visible .kpi-card {{ outline: 2px solid {VIOLET}; outline-offset: 2px; }}

.kpi-cta {{
    font-size: 10.5px; color: {TEXT_FAINT};
    margin-top: 10px; opacity: 0; min-height: 13px;
    transition: opacity 140ms ease;
}}

.kpi-link:hover .kpi-cta {{ opacity: 1; }}

/* ---------------- filter pills ---------------- */

.pill-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }}

.pill {{
    text-decoration: none;
    padding: 7px 14px; border-radius: 20px;
    font-size: 12.5px; font-weight: 500;
    border: 1px solid {LINE_SOFT};
    color: {TEXT_DIM}; background: {CARD};
    transition: border-color 130ms ease, color 130ms ease;
}}

.pill:hover {{ color: {TEXT}; border-color: {LINE}; }}

.pill-count {{
    font-variant-numeric: tabular-nums;
    margin-left: 7px; color: {TEXT_FAINT}; font-size: 11px;
}}

.back-link {{
    display: inline-block; text-decoration: none;
    color: {TEXT_DIM}; font-size: 12.5px; margin-bottom: 14px;
}}

.back-link:hover {{ color: {TEXT}; }}

/* ---------------- datatable chrome ---------------- */

.dash-table-container {{ font-family: {SANS}; }}

/* the native sort arrows render as dark-on-dark by default */
.dash-table-container .column-header--sort {{
    color: {TEXT_FAINT} !important;
    padding-right: 5px;
}}

.dash-table-container .column-header--sort:hover {{ color: {VIOLET} !important; }}

.dash-table-container th .column-header-name {{
    font-weight: 500;
    letter-spacing: 0.01em;
}}

.dash-table-container .previous-next-container {{ padding-top: 14px; }}

.dash-table-container .previous-next-container button,
.dash-table-container .previous-next-container .page-number,
.dash-table-container .previous-next-container input {{
    color: {TEXT_DIM} !important;
    font-size: 12px !important;
    background: transparent !important;
    border-color: {LINE_SOFT} !important;
}}

.dash-table-container .previous-next-container button:hover {{ color: {TEXT} !important; }}

.dash-table-container .dash-filter input {{
    color: {TEXT} !important;
    background: {CARD_2} !important;
    border-radius: 6px !important;
}}

/* ---------------- filter dropdowns ---------------- */

.filter-row {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr)) auto;
    gap: 10px; align-items: end;
    margin-bottom: 16px;
}}

.filter-field {{ min-width: 0; }}

.filter-label {{
    display: block; font-size: 11px; color: {TEXT_FAINT};
    margin-bottom: 7px; font-weight: 500;
}}

.filter-count {{
    font-size: 12.5px; color: {TEXT_DIM};
    white-space: nowrap; padding-bottom: 10px;
}}

.filter-count b {{ color: {TEXT}; font-variant-numeric: tabular-nums; }}

/* dcc.Dropdown in this dash version is a fully custom component - it does
   NOT use react-select, so none of the .Select-control / .Select__menu
   selectors you find online do anything. pulled the real class names out
   of dash/dcc/async-dropdown.js:

     .dash-dropdown-wrapper   outer
     .dash-dropdown-trigger   the closed control
     .dash-dropdown-value     the text inside it (+ -placeholder, -value-item)
     .dash-dropdown-content   the popup
     .dash-dropdown-search    the search box
     .dash-dropdown-actions   the Select All / Deselect All row
     .dash-dropdown-option    each row in the list

   the popup renders inside the wrapper (no portal) and carries no inline
   background, so plain css is enough - no !important arms race needed. */

.dash-dropdown {{ font-family: {SANS}; font-size: 12.5px; }}

.dash-dropdown-wrapper {{ color: {TEXT}; }}

.dash-dropdown-trigger {{
    background: {CARD_2};
    border: 1px solid {LINE_SOFT};
    border-radius: 9px;
    color: {TEXT};
    min-height: 38px;
}}

.dash-dropdown-trigger:hover {{ border-color: {LINE}; }}

.dash-dropdown-wrapper:focus-within .dash-dropdown-trigger {{
    border-color: {VIOLET};
    box-shadow: none;
}}

.dash-dropdown-value {{ color: {TEXT}; }}
.dash-dropdown-placeholder {{ color: {TEXT_DIM}; }}

.dash-dropdown-value-item {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 6px;
    color: {TEXT};
    font-size: 11.5px;
}}

.dash-dropdown-value-count {{
    background: {VIOLET};
    color: #fff;
    border-radius: 20px;
    font-size: 11px;
}}

.dash-dropdown-trigger-icon,
.dash-dropdown-search-icon {{ color: {TEXT_FAINT}; }}

.dash-dropdown-clear {{ color: {TEXT_FAINT}; }}
.dash-dropdown-clear:hover {{ color: #fb7185; }}

/* ---- the popup ---- */

.dash-dropdown-content {{
    background: {CARD_2};
    border: 1px solid {LINE};
    color: {TEXT};
    box-shadow: 0 14px 34px rgba(0,0,0,0.55);
}}

.dash-dropdown-search-container {{
    background: {CARD_2};
    border-bottom: 1px solid {LINE_SOFT};
}}

.dash-dropdown-search {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 7px;
    color: {TEXT};
    font-family: {SANS};
    font-size: 12.5px;
}}

.dash-dropdown-search::placeholder {{ color: {TEXT_FAINT}; }}

.dash-dropdown-search:focus {{
    outline: none;
    border-color: {VIOLET};
}}

.dash-dropdown-actions {{
    background: {CARD_2};
    border-bottom: 1px solid {LINE_SOFT};
}}

.dash-dropdown-action-button {{
    background: transparent;
    border: none;
    color: {VIOLET};
    font-family: {SANS};
    font-size: 11.5px;
    cursor: pointer;
}}

.dash-dropdown-action-button:hover {{ color: {TEXT}; }}

.dash-dropdown-options {{ background: {CARD_2}; }}

.dash-dropdown-option {{
    background: transparent;
    color: {TEXT_DIM};
    font-size: 12.5px;
}}

.dash-dropdown-option:hover,
.dash-dropdown-option[aria-selected="true"] {{
    background: {CARD};
    color: {TEXT};
}}

.dash-dropdown-option input[type="checkbox"] {{
    accent-color: {VIOLET};
    width: 13px; height: 13px;
}}

.dash-dropdown-content ::-webkit-scrollbar {{ width: 9px; }}
.dash-dropdown-content ::-webkit-scrollbar-track {{ background: {CARD_2}; }}
.dash-dropdown-content ::-webkit-scrollbar-thumb {{
    background: {LINE}; border-radius: 5px;
}}
.dash-dropdown-content ::-webkit-scrollbar-thumb:hover {{ background: #2e3446; }}

@media (max-width: 1000px) {{
    .filter-row {{ grid-template-columns: 1fr; }}
    .filter-count {{ padding-bottom: 0; }}
}}

/* ---------------- model strip ---------------- */

/* sits to the right of the page title as a joined group of cells rather
   than a full width bar - takes up less room and stops competing with
   the KPI row underneath it */

.page-head {{
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 28px;
    margin-bottom: 24px;
}}

.page-head-text {{ flex: 1 1 auto; min-width: 0; }}
.page-head-text .page-sub {{ margin-bottom: 0; }}

.model-strip {{
    flex: 0 0 auto;
    display: flex;
    background: {CARD};
    border: 1px solid {LINE_SOFT};
    border-radius: 10px;
    overflow: hidden;
}}

.model-stat {{
    display: flex; flex-direction: column; gap: 5px;
    padding: 11px 18px;
    border-right: 1px solid {LINE_SOFT};
}}

.model-stat:last-child {{ border-right: none; }}

.model-stat-label {{
    font-size: 9.5px; color: {TEXT_FAINT};
    text-transform: uppercase; letter-spacing: 0.08em;
    white-space: nowrap;
}}

.model-stat-value {{
    font-size: 14px; color: {TEXT}; font-weight: 500;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em; white-space: nowrap;
}}

.model-stat-value.accent {{ color: {VIOLET}; }}

@media (max-width: 1180px) {{
    .page-head {{ flex-direction: column; gap: 18px; }}
    .model-strip {{ width: 100%; }}
    .model-stat {{ flex: 1; }}
}}

@media (max-width: 640px) {{
    .model-strip {{ flex-wrap: wrap; }}
    .model-stat {{ flex: 1 0 45%; border-bottom: 1px solid {LINE_SOFT}; }}
}}

/* even two-column row (the split-row above is deliberately lopsided) */
.duo-row {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 14px; margin-bottom: 14px;
}}

.duo-row .card {{ margin-bottom: 0; }}

@media (max-width: 1280px) {{
    .duo-row {{ grid-template-columns: 1fr; }}
    .model-strip-note {{ margin-left: 0; width: 100%; }}
}}

/* ---------------- layout grids ---------------- */

.split-row {{
    display: grid; grid-template-columns: 380px 1fr;
    gap: 14px; margin-bottom: 14px;
}}

.card {{
    background: {CARD};
    border: 1px solid {LINE_SOFT};
    border-radius: 14px;
    padding: 20px 22px 22px;
    margin-bottom: 14px;
}}

.split-row .card {{ margin-bottom: 0; }}

/* ---------------- donut legend ---------------- */

.legend {{ display: flex; flex-direction: column; gap: 12px; margin-top: 4px; }}

.legend-row {{ display: flex; align-items: center; font-size: 12.5px; }}

.legend-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    margin-right: 10px; flex-shrink: 0;
}}

.legend-name {{ color: {TEXT_DIM}; }}

.legend-val {{
    margin-left: auto; font-variant-numeric: tabular-nums;
    font-weight: 600; font-size: 12.5px;
}}

.legend-pct {{
    color: {TEXT_FAINT}; font-size: 11px;
    margin-left: 9px; font-variant-numeric: tabular-nums;
    min-width: 40px; text-align: right;
}}

/* ---------------- table ---------------- */

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td {{
    background: {CARD_2} !important;
}}

.dash-table-container .dash-spreadsheet-inner td {{
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid {LINE_SOFT} !important;
}}

.dash-table-container .row {{ margin: 0; }}

.risk-pill {{
    display: inline-block; padding: 3px 9px;
    border-radius: 20px; font-size: 11px; font-weight: 600;
}}

/* ---------------- responsive ---------------- */

@media (max-width: 1280px) {{
    .split-row {{ grid-template-columns: 1fr; }}
}}

@media (max-width: 1000px) {{
    .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    .content {{ padding: 24px 20px 44px; }}
    .sidebar {{ width: 190px; }}
}}

@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
"""