import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
import math
from scipy.stats import poisson

st.set_page_config(page_title="OpenLigaDB — Wettanalyse (Bundesliga & DEL)", layout="wide")

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
    "2. Bundesliga": "bl2",
    "DEL (Eishockey)": "del"
}

# -----------------------------
# Low-level helpers
# -----------------------------
@st.cache_data(ttl=300)
def try_openliga_endpoint(endpoint: str) -> dict:
    """Try several base URLs until one returns valid JSON.
    Returns parsed JSON (or empty dict on failure).
    """
    last_err = None
    for base in BASE_OPTIONS:
        url = base.rstrip('/') + '/' + endpoint.lstrip('/')
        try:
            resp = requests.get(url, timeout=10)
            # follow redirects; treat 200 as success
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
    """Recursively lowercase dict keys to make parsing robust against keycase variations."""
    if isinstance(obj, dict):
        return {k.lower(): normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj

# -----------------------------
# API functions (with fallbacks)
# -----------------------------
@st.cache_data(ttl=600)
def get_matches(league_short: str, season: int) -> pd.DataFrame:
    """Fetch match data for a league + season. Tries multiple endpoint variants.
    Returns DataFrame with normalized columns.
    """
    endpoints = [
        f"getmatchdata/{league_short}/{season}",
        f"api/getmatchdata/{league_short}/{season}",
        f"getmatchdata/{league_short}",
        f"api/getmatchdata/{league_short}"
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
        match_id = nm.get('matchid') or nm.get('matchid')
        date = nm.get('matchdatetime') or nm.get('matchdateutc') or nm.get('matchdate')
        # team objects can be nested or direct strings depending on endpoint
        def teamname(t):
            if not t:
                return None
            if isinstance(t, dict):
                return t.get('teamname') or t.get('name')
            return t
        team1 = teamname(nm.get('team1'))
        team2 = teamname(nm.get('team2'))

        # matchresults: choose "final" / highest order id or last element
        goals_home = None
        goals_away = None
        mr = nm.get('matchresults') or []
        if isinstance(mr, list) and mr:
            # pick element with highest resultorderid if present
            try:
                chosen = max(mr, key=lambda x: x.get('resultorderid') if isinstance(x, dict) else 0)
            except Exception:
                chosen = mr[-1]
            cm = normalize(chosen)
            goals_home = cm.get('pointsteam1') or cm.get('points_team1') or cm.get('points') and None
            goals_away = cm.get('pointsteam2') or cm.get('points_team2') or None
            # try convert to int if possible
            try:
                goals_home = int(goals_home) if goals_home is not None else None
            except Exception:
                goals_home = None
            try:
                goals_away = int(goals_away) if goals_away is not None else None
            except Exception:
                goals_away = None

        finished = nm.get('matchisfinished') or nm.get('matchfinished') or nm.get('matchidfinished') or False
        # group info
        group = nm.get('group') or {}
        group_name = group.get('groupname') if isinstance(group, dict) else None
        group_order = group.get('grouporderid') if isinstance(group, dict) else None

        rows.append({
            'match_id': match_id,
            'date': date,
            'team_home': team1,
            'team_away': team2,
            'goals_home': goals_home,
            'goals_away': goals_away,
            'finished': bool(finished),
            'group_name': group_name,
            'group_order': group_order,
            'raw': m
        })
    df = pd.DataFrame(rows)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

@st.cache_data(ttl=600)
def get_table(league_short: str, season: int) -> pd.DataFrame:
    endpoints = [
        f"getbltable/{league_short}/{season}",
        f"gettable/{league_short}/{season}",
        f"api/getbltable/{league_short}/{season}",
        f"api/gettable/{league_short}/{season}"
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
    for t in data:
        nt = normalize(t)
        # team might be nested
        team = None
        if isinstance(nt.get('team'), dict):
            team = nt['team'].get('teamname')
        else:
            team = nt.get('teamname') or nt.get('team')
        rows.append({
            'team': team,
            'rank': nt.get('rank'),
            'points': nt.get('points'),
            'goals': nt.get('goals'),
            'opponent_goals': nt.get('opponentgoals'),
            'diff': nt.get('goaldiff')
        })
    return pd.DataFrame(rows)

# -----------------------------
# Analysis helpers
# -----------------------------

def compute_form(matches_df: pd.DataFrame, team: str, last_n: int = 5) -> dict:
    df = matches_df.copy()
    df = df[df['finished'] == True]
    df = df[((df['team_home'] == team) | (df['team_away'] == team))].sort_values('date', ascending=False).head(last_n)
    points = 0
    results = []
    for _, r in df.iterrows():
        if pd.isna(r['goals_home']) or pd.isna(r['goals_away']):
            continue
        if r['team_home'] == team:
            if r['goals_home'] > r['goals_away']:
                points += 3; results.append('W')
            elif r['goals_home'] == r['goals_away']:
                points += 1; results.append('D')
            else:
                results.append('L')
        else:
            if r['goals_away'] > r['goals_home']:
                points += 3; results.append('W')
            elif r['goals_away'] == r['goals_home']:
                points += 1; results.append('D')
            else:
                results.append('L')
    return {'points': points, 'results': results, 'matches_considered': len(df)}


def head2head(matches_df: pd.DataFrame, home: str, away: str, last_n: int = 10) -> pd.DataFrame:
    df = matches_df.copy()
    cond = ((df['team_home'] == home) & (df['team_away'] == away)) | ((df['team_home'] == away) & (df['team_away'] == home))
    h2h = df[cond].sort_values('date', ascending=False).head(last_n)
    return h2h

# Poisson-based prediction

def poisson_prob_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    probs_home = [poisson.pmf(k, lam_home) for k in range(max_goals+1)]
    probs_away = [poisson.pmf(k, lam_away) for k in range(max_goals+1)]
    mat = np.outer(probs_home, probs_away)
    return mat


def predict_poisson(matches_df: pd.DataFrame, home: str, away: str) -> dict:
    # compute averages
    df = matches_df[matches_df['finished'] == True]
    # home goals per home-game
    home_scored_home = df[df['team_home'] == home]['goals_home'].dropna()
    away_conceded_away = df[df['team_away'] == away]['goals_home'].dropna()  # careful with columns; fallback below
    # safer: compute team's avg scored at home and conceded at home/away properly
    try:
        home_avg_scored_home = float(home_scored_home.mean()) if not home_scored_home.empty else 1.2
    except Exception:
        home_avg_scored_home = 1.2
    # away_avg_scored_away
    away_scored_away = df[df['team_away'] == away]['goals_away'].dropna()
    try:
        away_avg_scored_away = float(away_scored_away.mean()) if not away_scored_away.empty else 1.0
    except Exception:
        away_avg_scored_away = 1.0

    # conceded averages
    home_conceded_home = df[df['team_home'] == home]['goals_away'].dropna()
    away_conceded_away = df[df['team_away'] == away]['goals_home'].dropna()
    try:
        home_avg_conceded = float(home_conceded_home.mean()) if not home_conceded_home.empty else 1.0
    except Exception:
        home_avg_conceded = 1.0
    try:
        away_avg_conceded = float(away_conceded_away.mean()) if not away_conceded_away.empty else 1.1
    except Exception:
        away_avg_conceded = 1.1

    # simple lambda estimators (blend attack and opponent defense)
    lam_home = (home_avg_scored_home + away_avg_conceded) / 2
    lam_away = (away_avg_scored_away + home_avg_conceded) / 2

    mat = poisson_prob_matrix(lam_home, lam_away, max_goals=10)
    p_home = np.tril(mat, -1).sum()  # sum where i>j
    p_draw = np.trace(mat)
    p_away = np.triu(mat, 1).sum()
    
    # Calculate exact score probabilities
    score_probs = {}
    for i in range(0, 6):  # Home goals
        for j in range(0, 6):  # Away goals
            score_probs[f"{i}-{j}"] = mat[i, j]
    
    # Calculate over/under probabilities
    total_goals_probs = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        over_prob = 0
        under_prob = 0
        for i in range(0, 11):  # Home goals
            for j in range(0, 11):  # Away goals
                total = i + j
                if total > threshold:
                    over_prob += mat[i, j]
                elif total < threshold:
                    under_prob += mat[i, j]
        total_goals_probs[f"over_{threshold}"] = over_prob
        total_goals_probs[f"under_{threshold}"] = under_prob

    return {
        'lam_home': lam_home,
        'lam_away': lam_away,
        'p_home': float(p_home),
        'p_draw': float(p_draw),
        'p_away': float(p_away),
        'score_probs': score_probs,
        'total_goals_probs': total_goals_probs,
        'probability_matrix': mat
    }

# Expected goals (xG) estimation (simplified model)
def estimate_expected_goals(matches_df: pd.DataFrame, team: str, is_home: bool) -> dict:
    df = matches_df[matches_df['finished'] == True]
    
    if is_home:
        team_matches = df[df['team_home'] == team]
        goals = team_matches['goals_home'].dropna()
        conceded = team_matches['goals_away'].dropna()
    else:
        team_matches = df[df['team_away'] == team]
        goals = team_matches['goals_away'].dropna()
        conceded = team_matches['goals_home'].dropna()
    
    # Simple xG estimation based on historical performance
    xg_for = goals.mean() if not goals.empty else 1.0
    xg_against = conceded.mean() if not conceded.empty else 1.0
    
    # Calculate xG per match type (home/away)
    xg_per_match = {
        'xg_for': xg_for,
        'xg_against': xg_against,
        'matches_considered': len(goals)
    }
    
    return xg_per_match

# -----------------------------
# Bankroll / Kelly
# -----------------------------

def kelly_fraction(p: float, odds: float) -> float:
    """Return Kelly fraction (full Kelly). odds = decimal odds (>=1)
    If odds <=1 or p <=0: return 0.
    """
    if odds <= 1 or p <= 0:
        return 0.0
    b = odds - 1
    f = (p * b - (1 - p)) / b
    return max(f, 0.0)

# -----------------------------
# Visualization functions
# -----------------------------

def plot_score_probabilities(score_probs, max_scores=5):
    """Plot a heatmap of the most probable scorelines"""
    # Extract the top scores by probability
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    top_scores = sorted_scores[:max_scores]
    
    scores = [f"{score[0]}" for score in top_scores]
    probabilities = [score[1] * 100 for score in top_scores]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(scores, probabilities, color='skyblue')
    ax.set_xlabel('Ergebnis')
    ax.set_ylabel('Wahrscheinlichkeit (%)')
    ax.set_title('Wahrscheinlichste Spielergebnisse')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_total_goals_probabilities(total_goals_probs):
    """Plot over/under probabilities for different thresholds"""
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    over_probs = [total_goals_probs[f"over_{t}"] * 100 for t in thresholds]
    under_probs = [total_goals_probs[f"under_{t}"] * 100 for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, over_probs, width, label='Über', color='lightgreen')
    bars2 = ax.bar(x + width/2, under_probs, width, label='Unter', color='lightcoral')
    
    ax.set_xlabel('Torgrenze')
    ax.set_ylabel('Wahrscheinlichkeit (%)')
    ax.set_title('Wahrscheinlichkeit für Gesamttore (Über/Unter)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}" for t in thresholds])
    ax.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    return fig

def plot_xg_comparison(xg_home, xg_away, home_team, away_team):
    """Plot xG comparison between teams"""
    categories = ['xG Für', 'xG Gegen']
    home_values = [xg_home['xg_for'], xg_home['xg_against']]
    away_values = [xg_away['xg_for'], xg_away['xg_against']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, home_values, width, label=home_team, color='lightblue')
    bars2 = ax.bar(x + width/2, away_values, width, label=away_team, color='lightcoral')
    
    ax.set_xlabel('Metrik')
    ax.set_ylabel('Erwartete Tore')
    ax.set_title('Erwartete Tore (xG) Vergleich')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    return fig

def plot_match_outcome_probabilities(p_home, p_draw, p_away, home_team, away_team):
    """Plot probabilities for match outcomes"""
    outcomes = ['Heimsieg', 'Unentschieden', 'Auswärtssieg']
    probabilities = [p_home * 100, p_draw * 100, p_away * 100]
    colors = ['lightblue', 'lightgray', 'lightcoral']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(outcomes, probabilities, color=colors)
    ax.set_xlabel('Ergebnis')
    ax.set_ylabel('Wahrscheinlichkeit (%)')
    ax.set_title(f'Spielausgang: {home_team} vs {away_team}')
    ax.set_ylim(0, max(probabilities) * 1.1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("⚽ OpenLigaDB — Wettanalyse (Bundesliga & DEL)")

# Sidebar
st.sidebar.header("Konfiguration")
league_name = st.sidebar.selectbox("Liga", list(LEAGUES.keys()))
league_short = LEAGUES[league_name]
season = st.sidebar.number_input("Saison", min_value=2000, max_value=datetime.datetime.now().year + 1, value=datetime.datetime.now().year)

# Main content
st.header(f"{league_name} — Saison {season}/{season+1}")

# Fetch data
with st.spinner("Lade Daten..."):
    matches_df = get_matches(league_short, season)
    table_df = get_table(league_short, season)

if matches_df.empty:
    st.error("Keine Spieldaten gefunden. Bitte überprüfen Sie Liga und Saison.")
    st.stop()

# Filter matches
st.subheader("Spielanalyse")
upcoming_matches = matches_df[matches_df['finished'] == False].sort_values('date')
if not upcoming_matches.empty:
    match_options = [f"{row['team_home']} vs {row['team_away']} ({row['date'].strftime('%d.%m.%Y %H:%M')})" for _, row in upcoming_matches.iterrows()]
    selected_match = st.selectbox("Wähle ein bevorstehendes Spiel", match_options)
    
    # Extract teams from selected match
    match_idx = match_options.index(selected_match)
    match_data = upcoming_matches.iloc[match_idx]
    home_team = match_data['team_home']
    away_team = match_data['team_away']
    
    st.write(f"**Ausgewähltes Spiel:** {home_team} vs {away_team}")
    
    # Team information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{home_team}")
        home_form = compute_form(matches_df, home_team)
        st.write(f"Form (letzte {home_form['matches_considered']} Spiele): {', '.join(home_form['results'])}")
        st.write(f"Punkte: {home_form['points']}/{home_form['matches_considered']*3}")
        
        # Get team position from table
        home_position = table_df[table_df['team'] == home_team]['rank'].iloc[0] if not table_df.empty and home_team in table_df['team'].values else "N/A"
        st.write(f"Tabellenposition: {home_position}")
    
    with col2:
        st.subheader(f"{away_team}")
        away_form = compute_form(matches_df, away_team)
        st.write(f"Form (letzte {away_form['matches_considered']} Spiele): {', '.join(away_form['results'])}")
        st.write(f"Punkte: {away_form['points']}/{away_form['matches_considered']*3}")
        
        # Get team position from table
        away_position = table_df[table_df['team'] == away_team]['rank'].iloc[0] if not table_df.empty and away_team in table_df['team'].values else "N/A"
        st.write(f"Tabellenposition: {away_position}")
    
    # Head-to-head
    st.subheader("Vergangene Begegnungen")
    h2h = head2head(matches_df, home_team, away_team)
    if not h2h.empty:
        h2h_display = h2h.copy()
        h2h_display['Datum'] = h2h_display['date'].dt.strftime('%d.%m.%Y')
        h2h_display['Ergebnis'] = h2h_display.apply(lambda x: f"{x['goals_home']} - {x['goals_away']}", axis=1)
        st.dataframe(h2h_display[['Datum', 'team_home', 'team_away', 'Ergebnis']].rename(
            columns={'team_home': 'Heim', 'team_away': 'Auswärts'}), hide_index=True)
    else:
        st.info("Keine vergangenen Begegnungen gefunden.")
    
    # Prediction
    st.subheader("Vorhersage")
    prediction = predict_poisson(matches_df, home_team, away_team)
    
    # Display outcome probabilities
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Heimsieg", f"{prediction['p_home']*100:.1f}%")
    with col2:
        st.metric("Unentschieden", f"{prediction['p_draw']*100:.1f}%")
    with col3:
        st.metric("Auswärtssieg", f"{prediction['p_away']*100:.1f}%")
    
    # Visualizations
    st.subheader("Visualisierungen")
    
    # Match outcome probabilities
    fig1 = plot_match_outcome_probabilities(
        prediction['p_home'], prediction['p_draw'], prediction['p_away'],
        home_team, away_team
    )
    st.pyplot(fig1)
    
    # Score probabilities
    fig2 = plot_score_probabilities(prediction['score_probs'])
    st.pyplot(fig2)
    
    # Total goals probabilities
    fig3 = plot_total_goals_probabilities(prediction['total_goals_probs'])
    st.pyplot(fig3)
    
    # Expected goals analysis
    st.subheader("Erwartete Tore (xG) Analyse")
    xg_home = estimate_expected_goals(matches_df, home_team, is_home=True)
    xg_away = estimate_expected_goals(matches_df, away_team, is_home=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{home_team} (Heim):**")
        st.write(f"xG Für: {xg_home['xg_for']:.2f}")
        st.write(f"xG Gegen: {xg_home['xg_against']:.2f}")
        st.write(f"Basierend auf {xg_home['matches_considered']} Spielen")
    
    with col2:
        st.write(f"**{away_team} (Auswärts):**")
        st.write(f"xG Für: {xg_away['xg_for']:.2f}")
        st.write(f"xG Gegen: {xg_away['xg_against']:.2f}")
        st.write(f"Basierend auf {xg_away['matches_considered']} Spielen")
    
    # xG comparison chart
    fig4 = plot_xg_comparison(xg_home, xg_away, home_team, away_team)
    st.pyplot(fig4)
    
    # Betting analysis
    st.subheader("Wettanalyse")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_odds = st.number_input("Heimsieg-Quote", min_value=1.0, max_value=100.0, value=2.0, step=0.1)
    with col2:
        draw_odds = st.number_input("Unentschieden-Quote", min_value=1.0, max_value=100.0, value=3.5, step=0.1)
    with col3:
        away_odds = st.number_input("Auswärtssieg-Quote", min_value=1.0, max_value=100.0, value=4.0, step=0.1)
    
    bankroll = st.number_input("Bankroll (€)", min_value=1, max_value=100000, value=1000)
    
    kelly_home = kelly_fraction(prediction['p_home'], home_odds)
    kelly_draw = kelly_fraction(prediction['p_draw'], draw_odds)
    kelly_away = kelly_fraction(prediction['p_away'], away_odds)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Kelly Heimsieg", f"{kelly_home*100:.1f}%", f"€{bankroll * kelly_home:.2f}")
    with col2:
        st.metric("Kelly Unentschieden", f"{kelly_draw*100:.1f}%", f"€{bankroll * kelly_draw:.2f}")
    with col3:
        st.metric("Kelly Auswärtssieg", f"{kelly_away*100:.1f}%", f"€{bankroll * kelly_away:.2f}")
    
    # Expected value calculation
    ev_home = (prediction['p_home'] * (home_odds - 1) - (1 - prediction['p_home'])) * 100
    ev_draw = (prediction['p_draw'] * (draw_odds - 1) - (1 - prediction['p_draw'])) * 100
    ev_away = (prediction['p_away'] * (away_odds - 1) - (1 - prediction['p_away'])) * 100
    
    st.write("**Erwartungswert (EV):**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("EV Heimsieg", f"{ev_home:.1f}%", "positiv" if ev_home > 0 else "negativ")
    with col2:
        st.metric("EV Unentschieden", f"{ev_draw:.1f}%", "positiv" if ev_draw > 0 else "negativ")
    with col3:
        st.metric("EV Auswärtssieg", f"{ev_away:.1f}%", "positiv" if ev_away > 0 else "negativ")
    
    # Betting recommendations
    st.subheader("Wettempfehlungen")
    recommendations = []
    if ev_home > 0:
        recommendations.append(f"Heimsieg: {home_team} zu {home_odds} (EV: {ev_home:.1f}%)")
    if ev_draw > 0:
        recommendations.append(f"Unentschieden zu {draw_odds} (EV: {ev_draw:.1f}%)")
    if ev_away > 0:
        recommendations.append(f"Auswärtssieg: {away_team} zu {away_odds} (EV: {ev_away:.1f}%)")
    
    if recommendations:
        for rec in recommendations:
            st.success(rec)
    else:
        st.warning("Keine positiven Erwartungswerte gefunden. Keine Wettempfehlung.")
    
else:
    st.info("Keine bevorstehenden Spiele gefunden.")

# Show league table
if not table_df.empty:
    st.subheader("Tabelle")
    st.dataframe(table_df, hide_index=True)