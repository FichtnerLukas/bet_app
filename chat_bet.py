import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
from scipy.stats import poisson

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Fußball-Wettanalyse", layout="wide", initial_sidebar_state="expanded")

# --- Style Enhancements ---
# A more professional and visually appealing style for plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelcolor': '#333',
    'xtick.color': '#333',
    'ytick.color': '#333',
    'text.color': '#333',
    'axes.titleweight': 'bold',
    'axes.titlesize': 16,
    'axes.labelsize': 12,
})

# -----------------------------
# Configuration
# -----------------------------
BASE_OPTIONS = [
    "https://api.openligadb.de",
    "https://www.openligadb.de/api",
    "https://www.openligadb.de"
]

LEAGUES = {
    "1. Bundesliga": "bl1",
    "2. Bundesliga": "bl2"
}

# -----------------------------
# Low-level helpers
# -----------------------------
@st.cache_data(ttl=300)
def try_openliga_endpoint(endpoint: str) -> dict:
    """Try several base URLs until one returns valid JSON."""
    last_err = None
    for base in BASE_OPTIONS:
        url = f"{base.rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    last_err = f"JSON parse error for {url}"
                    continue
            else:
                last_err = f"Status {resp.status_code} for {url}"
        except requests.RequestException as e:
            last_err = str(e)
    st.warning(f"OpenLigaDB: Anfrage an '{endpoint}' fehlgeschlagen. Letzter Fehler: {last_err}")
    return {}

def normalize(obj):
    """Recursively lowercase dict keys to make parsing robust."""
    if isinstance(obj, dict):
        return {k.lower(): normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj

# -----------------------------
# API functions
# -----------------------------
@st.cache_data(ttl=600)
def get_matches(league_short: str, season: int) -> pd.DataFrame:
    """Fetch all match data for a league + season."""
    endpoints = [
        f"getmatchdata/{league_short}/{season}",
        f"api/getmatchdata/{league_short}/{season}",
    ]
    data = None
    for ep in endpoints:
        d = try_openliga_endpoint(ep)
        if d:
            data = d
            break
    if not data:
        return pd.DataFrame()

    rows = []
    for m in data:
        nm = normalize(m)

        def teamname(t):
            if isinstance(t, dict):
                return t.get('teamname')
            return t

        team1 = teamname(nm.get('team1'))
        team2 = teamname(nm.get('team2'))

        goals_home, goals_away = None, None
        mr = nm.get('matchresults', [])
        if mr:
            final_result = max(mr, key=lambda x: x.get('resultorderid', 0), default=None)
            if final_result:
                goals_home = final_result.get('pointsteam1')
                goals_away = final_result.get('pointsteam2')

        rows.append({
            'match_id': nm.get('matchid'),
            'date': nm.get('matchdatetime'),
            'team_home': team1,
            'team_away': team2,
            'goals_home': pd.to_numeric(goals_home, errors='coerce'),
            'goals_away': pd.to_numeric(goals_away, errors='coerce'),
            'finished': nm.get('matchisfinished', False),
        })
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df.dropna(subset=['date', 'team_home', 'team_away'])

@st.cache_data(ttl=600)
def get_table(league_short: str, season: int) -> pd.DataFrame:
    """Fetch league table data."""
    endpoint = f"getbltable/{league_short}/{season}"
    data = try_openliga_endpoint(endpoint)
    if not data:
        return pd.DataFrame()

    rows = [{'team': t.get('teamName'), 'rank': t.get('place', t.get('rank')), 'points': t.get('points')} for t in data]
    return pd.DataFrame(rows)

# -----------------------------
# Analysis helpers
# -----------------------------
def compute_form(matches_df: pd.DataFrame, team: str, last_n: int = 3) -> dict:
    """Computes form over the last N finished games."""
    team_matches = matches_df[
        matches_df['finished'] &
        ((matches_df['team_home'] == team) | (matches_df['team_away'] == team))
    ].sort_values('date', ascending=False).head(last_n)

    points = 0
    results = []
    for _, r in team_matches.iterrows():
        if pd.isna(r['goals_home']) or pd.isna(r['goals_away']): continue

        is_home = r['team_home'] == team
        home_goals, away_goals = r['goals_home'], r['goals_away']

        if (is_home and home_goals > away_goals) or (not is_home and away_goals > home_goals):
            points += 3; results.append('W')
        elif home_goals == away_goals:
            points += 1; results.append('D')
        else:
            results.append('L')

    return {'points': points, 'results': results, 'matches_considered': len(team_matches)}

def calculate_home_away_performance(matches_df: pd.DataFrame, team: str) -> dict:
    """NEW: Analyzes a team's performance at home vs. away."""
    df = matches_df[matches_df['finished']].copy()

    # Home stats
    home_games = df[df['team_home'] == team]
    home_wins = (home_games['goals_home'] > home_games['goals_away']).sum()
    home_draws = (home_games['goals_home'] == home_games['goals_away']).sum()
    home_losses = (home_games['goals_home'] < home_games['goals_away']).sum()
    home_scored = home_games['goals_home'].sum()
    home_conceded = home_games['goals_away'].sum()

    # Away stats
    away_games = df[df['team_away'] == team]
    away_wins = (away_games['goals_away'] > away_games['goals_home']).sum()
    away_draws = (away_games['goals_away'] == away_games['goals_home']).sum()
    away_losses = (away_games['goals_away'] < away_games['goals_home']).sum()
    away_scored = away_games['goals_away'].sum()
    away_conceded = away_games['goals_home'].sum()

    return {
        'home': {'W': int(home_wins), 'D': int(home_draws), 'L': int(home_losses), 'GF': int(home_scored), 'GA': int(home_conceded), 'GP': len(home_games)},
        'away': {'W': int(away_wins), 'D': int(away_draws), 'L': int(away_losses), 'GF': int(away_scored), 'GA': int(away_conceded), 'GP': len(away_games)}
    }

def head2head(matches_df: pd.DataFrame, home: str, away: str, last_n: int = 10) -> pd.DataFrame:
    cond = ((matches_df['team_home'] == home) & (matches_df['team_away'] == away)) | \
           ((matches_df['team_home'] == away) & (matches_df['team_away'] == home))
    return matches_df[cond].sort_values('date', ascending=False).head(last_n)

def predict_poisson(matches_df: pd.DataFrame, home: str, away: str) -> dict:
    """Poisson-based prediction using season-long data."""
    df = matches_df[matches_df['finished']].copy()

    home_avg_scored = df[df['team_home'] == home]['goals_home'].mean()
    away_avg_conceded = df[df['team_away'] == away]['goals_home'].mean()
    away_avg_scored = df[df['team_away'] == away]['goals_away'].mean()
    home_avg_conceded = df[df['team_home'] == home]['goals_away'].mean()

    # Fallback to league average if a team has no data
    league_avg_home_goals = df['goals_home'].mean()
    league_avg_away_goals = df['goals_away'].mean()

    lam_home = (home_avg_scored if not pd.isna(home_avg_scored) else league_avg_home_goals) * \
               (away_avg_conceded if not pd.isna(away_avg_conceded) else league_avg_away_goals) / league_avg_home_goals
    lam_away = (away_avg_scored if not pd.isna(away_avg_scored) else league_avg_away_goals) * \
               (home_avg_conceded if not pd.isna(home_avg_conceded) else league_avg_home_goals) / league_avg_away_goals

    max_goals = 10
    probs_home = [poisson.pmf(k, lam_home) for k in range(max_goals + 1)]
    probs_away = [poisson.pmf(k, lam_away) for k in range(max_goals + 1)]
    mat = np.outer(probs_home, probs_away)

    p_home = np.tril(mat, -1).sum()
    p_draw = np.trace(mat)
    p_away = np.triu(mat, 1).sum()

    score_probs = {f"{i}-{j}": mat[i, j] for i in range(6) for j in range(6)}

    total_goals_probs = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over_prob = sum(mat[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > threshold)
        total_goals_probs[f"over_{threshold}"] = over_prob
        total_goals_probs[f"under_{threshold}"] = 1 - over_prob

    return {'p_home': p_home, 'p_draw': p_draw, 'p_away': p_away, 'score_probs': score_probs, 'total_goals_probs': total_goals_probs}

def estimate_expected_goals(matches_df: pd.DataFrame, team: str) -> dict:
    """IMPROVED: Estimates xG based on all season matches."""
    df = matches_df[matches_df['finished']].copy()

    home_matches = df[df['team_home'] == team]
    away_matches = df[df['team_away'] == team]

    xg_for_home = home_matches['goals_home'].mean()
    xg_against_home = home_matches['goals_away'].mean()

    xg_for_away = away_matches['goals_away'].mean()
    xg_against_away = away_matches['goals_home'].mean()

    return {
        'home': {'for': xg_for_home if not pd.isna(xg_for_home) else 0, 'against': xg_against_home if not pd.isna(xg_against_home) else 0, 'count': len(home_matches)},
        'away': {'for': xg_for_away if not pd.isna(xg_for_away) else 0, 'against': xg_against_away if not pd.isna(xg_against_away) else 0, 'count': len(away_matches)}
    }

def kelly_fraction(p: float, odds: float) -> float:
    if odds <= 1 or p <= 0: return 0.0
    b = odds - 1
    f = (p * b - (1 - p)) / b
    return max(f, 0.0)

# -----------------------------
# Visualization functions (IMPROVED & FANCY)
# -----------------------------
def plot_match_outcome_probabilities(p_home, p_draw, p_away, home_team, away_team):
    """IMPROVED: Plots outcome probabilities with a modern design."""
    fig, ax = plt.subplots(figsize=(8, 5))
    outcomes = [f'Sieg {home_team}', 'Unentschieden', f'Sieg {away_team}']
    probs = [p_home * 100, p_draw * 100, p_away * 100]
    colors = ['#4A90E2', '#7F8C8D', '#E74C3C']

    bars = ax.barh(outcomes, probs, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel('Wahrscheinlichkeit (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Wahrscheinlichkeit des Spielausgangs', pad=20)
    ax.set_xlim(0, 100)

    # Remove spines
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)

    # Add labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', ha='left', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig

def plot_score_probabilities(score_probs):
    """IMPROVED: Plots top score probabilities as a clean bar chart."""
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    scores = [s[0] for s in top_scores]
    probs = [s[1] * 100 for s in top_scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(scores, probs, color='#4A90E2', width=0.6)

    ax.set_ylabel('Wahrscheinlichkeit (%)', fontsize=12, fontweight='bold')
    ax.set_title('Wahrscheinlichste Endergebnisse', pad=20)

    # Clean up aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, max(probs) * 1.15 if probs else 10)
    plt.tight_layout()
    return fig

def plot_total_goals_probabilities(total_goals_probs):
    """IMPROVED: Visualizes Over/Under probabilities with a modern aesthetic."""
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    over_probs = [total_goals_probs.get(f"over_{t}", 0) * 100 for t in thresholds]
    under_probs = [total_goals_probs.get(f"under_{t}", 0) * 100 for t in thresholds]

    x = np.arange(len(thresholds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, over_probs, width, label='Über', color='#2ECC71')
    bars2 = ax.bar(x + width/2, under_probs, width, label='Unter', color='#E74C3C')

    ax.set_xlabel('Tore', fontsize=12, fontweight='bold')
    ax.set_ylabel('Wahrscheinlichkeit (%)', fontsize=12, fontweight='bold')
    ax.set_title('Über/Unter-Tore Wahrscheinlichkeit', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in thresholds])
    ax.legend(frameon=False, fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    add_labels(bars1)
    add_labels(bars2)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    return fig

def plot_xg_comparison(xg_home, xg_away, home_team, away_team):
    """IMPROVED: Plots a stylish xG comparison chart."""
    categories = ['xG For (Heim)', 'xG Against (Heim)', 'xG For (Auswärts)', 'xG Against (Auswärts)']
    values = [xg_home['home']['for'], xg_home['home']['against'], xg_away['away']['for'], xg_away['away']['against']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, color=['#4A90E2', '#E74C3C', '#4A90E2', '#E74C3C'], alpha=0.8)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontweight='bold')

    ax.set_ylabel('Erwartete Tore pro Spiel', fontsize=12, fontweight='bold')
    ax.set_title('Vergleich der erwarteten Tore (xG)', pad=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.set_ylim(0, max(values) * 1.2 if values else 3)
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    return fig


# -----------------------------
# UI / Main App
# -----------------------------
st.title("⚽ Fußball-Wettanalyse")

# --- Sidebar ---
st.sidebar.header("Einstellungen")
league_name = st.sidebar.selectbox("Liga", list(LEAGUES.keys()))
league_short = LEAGUES[league_name]
current_year = datetime.datetime.now().year
season = st.sidebar.number_input("Saison (Startjahr)", min_value=2010, max_value=current_year + 1, value=current_year)
st.sidebar.markdown("---")

# --- Data Fetching ---
with st.spinner("Lade Spieldaten..."):
    matches_df = get_matches(league_short, season)
    table_df = get_table(league_short, season)

if matches_df.empty:
    st.error("Keine Spieldaten gefunden. Bitte Liga und Saison überprüfen.")
    st.stop()

# --- Match Selection in Sidebar ---
upcoming_matches = matches_df[~matches_df['finished']].sort_values('date')
if not upcoming_matches.empty:
    match_options = [f"{row['team_home']} vs {row['team_away']} ({row['date'].strftime('%d.%m.%y %H:%M')})" for _, row in upcoming_matches.iterrows()]
    selected_match_str = st.sidebar.selectbox("Wähle ein bevorstehendes Spiel", match_options, key="match_selector")

    match_idx = match_options.index(selected_match_str)
    match_data = upcoming_matches.iloc[match_idx]
    home_team, away_team = match_data['team_home'], match_data['team_away']
else:
    st.info("Keine bevorstehenden Spiele für die ausgewählte Saison gefunden.")
    st.stop()

# --- Main Page Content ---
st.header(f"{league_name} | Saison {season}/{season+1}")
st.subheader(f"Analyse: {home_team} vs. {away_team}")
st.markdown("---")

# --- Team Overview ---
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"#### {home_team} (Heim)")
    home_form = compute_form(matches_df, home_team, last_n=3)
    home_rank = table_df.loc[table_df['team'] == home_team, 'rank'].values
    home_rank_display = int(home_rank[0]) if len(home_rank) > 0 and not pd.isna(home_rank[0]) else "N/A"
    st.metric("Tabellenplatz", home_rank_display)
    st.metric(f"Form (letzte {home_form['matches_considered']} Spiele)", " ".join(home_form['results']) if home_form['results'] else "N/A")

with col2:
    st.markdown(f"#### {away_team} (Auswärts)")
    away_form = compute_form(matches_df, away_team, last_n=3)
    away_rank = table_df.loc[table_df['team'] == away_team, 'rank'].values
    away_rank_display = int(away_rank[0]) if len(away_rank) > 0 and not pd.isna(away_rank[0]) else "N/A"
    st.metric("Tabellenplatz", away_rank_display)
    st.metric(f"Form (letzte {away_form['matches_considered']} Spiele)", " ".join(away_form['results']) if away_form['results'] else "N/A")

st.markdown("---")


# --- NEW FEATURE: Home vs. Away Performance ---
st.subheader("🏠 Heim- vs. Auswärts-Performance")
home_perf = calculate_home_away_performance(matches_df, home_team)
away_perf = calculate_home_away_performance(matches_df, away_team)

perf_data = {
    'Team': [home_team, away_team],
    'Heimspiele': [f"{d['home']['GP']}" for d in [home_perf, away_perf]],
    'Heim-Bilanz (W-D-L)': [f"{d['home']['W']}-{d['home']['D']}-{d['home']['L']}" for d in [home_perf, away_perf]],
    'Heimtore (F-A)': [f"{d['home']['GF']}-{d['home']['GA']}" for d in [home_perf, away_perf]],
    'Auswärtsspiele': [f"{d['away']['GP']}" for d in [home_perf, away_perf]],
    'Auswärts-Bilanz (W-D-L)': [f"{d['away']['W']}-{d['away']['D']}-{d['away']['L']}" for d in [home_perf, away_perf]],
    'Auswärtstore (F-A)': [f"{d['away']['GF']}-{d['away']['GA']}" for d in [home_perf, away_perf]],
}
# FIX: Deprecation warning for use_container_width
st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)
st.markdown("---")


# --- Head-to-Head ---
st.subheader("Vergangene Begegnungen (H2H)")
h2h_df = head2head(matches_df, home_team, away_team)
if not h2h_df.empty:
    # BUG FIX: Drop rows where score is missing to prevent crash
    h2h_display = h2h_df.dropna(subset=['goals_home', 'goals_away']).copy()
    if not h2h_display.empty:
        h2h_display['Datum'] = h2h_display['date'].dt.strftime('%d.%m.%Y')
        h2h_display['Ergebnis'] = h2h_display.apply(lambda r: f"{int(r['goals_home'])} - {int(r['goals_away'])}", axis=1)
        # FIX: Deprecation warning for use_container_width
        st.dataframe(h2h_display[['Datum', 'team_home', 'Ergebnis', 'team_away']].rename(
            columns={'team_home': 'Heim', 'team_away': 'Auswärts'}), hide_index=True, use_container_width=True)
    else:
        st.info("Keine direkten vergangenen Begegnungen mit Ergebnisdaten gefunden.")
else:
    st.info("Keine direkten vergangenen Begegnungen in den Daten gefunden.")

st.markdown("---")

# --- Poisson Prediction & Visualizations ---
st.header("Statistische Vorhersage & Analyse")
prediction = predict_poisson(matches_df, home_team, away_team)

# Visuals first
st.pyplot(plot_match_outcome_probabilities(prediction['p_home'], prediction['p_draw'], prediction['p_away'], home_team, away_team))

tab1, tab2, tab3 = st.tabs(["Wahrscheinlichste Ergebnisse", "Über/Unter Tore", "Erwartete Tore (xG)"])
with tab1:
    st.pyplot(plot_score_probabilities(prediction['score_probs']))
with tab2:
    st.pyplot(plot_total_goals_probabilities(prediction['total_goals_probs']))
with tab3:
    xg_home = estimate_expected_goals(matches_df, home_team)
    xg_away = estimate_expected_goals(matches_df, away_team)
    st.pyplot(plot_xg_comparison(xg_home, xg_away, home_team, away_team))
    st.caption(f"xG-Daten basieren auf {xg_home['home']['count']} Heimspielen für {home_team} und {xg_away['away']['count']} Auswärtsspielen für {away_team} in dieser Saison.")

st.markdown("---")

# --- Betting Analysis ---
st.header("💰 Wettanalyse & Empfehlungen")
with st.expander("Wettquoten & Bankroll eingeben"):
    col1, col2, col3 = st.columns(3)
    with col1:
        home_odds = st.number_input("Heimsieg-Quote (1)", min_value=1.0, value=2.5, step=0.05)
    with col2:
        draw_odds = st.number_input("Unentschieden-Quote (X)", min_value=1.0, value=3.5, step=0.05)
    with col3:
        away_odds = st.number_input("Auswärtssieg-Quote (2)", min_value=1.0, value=3.0, step=0.05)
    bankroll = st.number_input("Bankroll (€)", min_value=1, value=100)

# Calculations
ev_home = prediction['p_home'] * home_odds - 1
ev_draw = prediction['p_draw'] * draw_odds - 1
ev_away = prediction['p_away'] * away_odds - 1
kelly_home = kelly_fraction(prediction['p_home'], home_odds)
kelly_draw = kelly_fraction(prediction['p_draw'], draw_odds)
kelly_away = kelly_fraction(prediction['p_away'], away_odds)

st.subheader("Analyse der Wettoptionen")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("EV Heimsieg", f"{ev_home*100:.2f}%", delta=f"{ev_home*100:.2f}%" if ev_home > 0 else None)
    st.metric("Kelly Heimsieg", f"{kelly_home*100:.1f}%", f"Einsatz: €{bankroll * kelly_home:.2f}")
with col2:
    st.metric("EV Unentschieden", f"{ev_draw*100:.2f}%", delta=f"{ev_draw*100:.2f}%" if ev_draw > 0 else None)
    st.metric("Kelly Unentschieden", f"{kelly_draw*100:.1f}%", f"Einsatz: €{bankroll * kelly_draw:.2f}")
with col3:
    st.metric("EV Auswärtssieg", f"{ev_away*100:.2f}%", delta=f"{ev_away*100:.2f}%" if ev_away > 0 else None)
    st.metric("Kelly Auswärtssieg", f"{kelly_away*100:.1f}%", f"Einsatz: €{bankroll * kelly_away:.2f}")

st.subheader("Wettempfehlungen (basierend auf positivem Erwartungswert)")
recommendations = []
if ev_home > 0: recommendations.append(f"**Heimsieg ({home_team})** zu Quote {home_odds} (EV: {ev_home*100:.2f}%)")
if ev_draw > 0: recommendations.append(f"**Unentschieden** zu Quote {draw_odds} (EV: {ev_draw*100:.2f}%)")
if ev_away > 0: recommendations.append(f"**Auswärtssieg ({away_team})** zu Quote {away_odds} (EV: {ev_away*100:.2f}%)")

if recommendations:
    for rec in recommendations:
        st.success(f"📈 {rec}")
else:
    st.warning("Keine Wetten mit positivem Erwartungswert bei den aktuellen Quoten gefunden.")

st.sidebar.markdown("---")
if not table_df.empty:
    st.sidebar.subheader("Aktuelle Tabelle")
    st.sidebar.dataframe(table_df, hide_index=True, height=600)