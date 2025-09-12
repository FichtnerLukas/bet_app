import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
import math

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

def poisson_prob_matrix(lam_home: float, lam_away: float, max_goals: int = 6) -> np.ndarray:
    probs_home = [math.exp(-lam_home) * (lam_home**k) / math.factorial(k) for k in range(max_goals+1)]
    probs_away = [math.exp(-lam_away) * (lam_away**k) / math.factorial(k) for k in range(max_goals+1)]
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

    mat = poisson_prob_matrix(lam_home, lam_away, max_goals=6)
    p_home = np.tril(mat, -1).sum()  # sum where i>j
    p_draw = np.trace(mat)
    p_away = np.triu(mat, 1).sum()

    return {
        'lam_home': lam_home,
        'lam_away': lam_away,
        'p_home': float(p_home),
        'p_draw': float(p_draw),
        'p_away': float(p_away)
    }

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
# Streamlit UI
# -----------------------------

def sidebar_controls():
    st.sidebar.title("Einstellungen")
    league_label = st.sidebar.selectbox("Liga", list(LEAGUES.keys()))
    league = LEAGUES[league_label]
    # Try to get sensible default season: current year or detect via available leagues
    current_year = datetime.datetime.now().year
    season = st.sidebar.number_input("Saison (Jahr)", min_value=2000, max_value=current_year, value=current_year)
    bankroll = st.sidebar.number_input("Bankroll (€)", min_value=1.0, value=100.0, step=1.0)
    return league_label, league, season, bankroll


def main():
    st.title("Wettanalyse — Bundesliga & DEL (OpenLigaDB)")

    league_label, league, season, bankroll = sidebar_controls()

    st.info(f"Lade Spiele für {league_label} — Saison {season}...")
    matches = get_matches(league, season)
    if matches.empty:
        st.error("Keine Spieldaten gefunden. Bitte Saison / Liga prüfen oder später erneut versuchen.")
        return

    # show next upcoming matchday (matches with future date)
    now = pd.Timestamp.now(tz=None)
    upcoming = matches[matches['date'] >= now].sort_values('date')
    if upcoming.empty:
        st.warning("Keine zukünftigen Matches in den Daten — zeige alle Matches dieser Saison.")
        upcoming = matches.sort_values('date')

    st.subheader("Begegnungen (Auswahl)")
    display_df = upcoming[['date','team_home','team_away','finished']].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(display_df)

    # select match
    upcoming['label'] = upcoming['team_home'] + ' vs ' + upcoming['team_away'] + ' — ' + upcoming['date'].dt.strftime('%Y-%m-%d %H:%M')
    selection = st.selectbox("Spiel auswählen", options=upcoming['label'].tolist())
    match_row = upcoming[upcoming['label'] == selection].iloc[0]

    st.header(f"Analyse: {match_row['team_home']} vs {match_row['team_away']}")
    st.write(f"Datum: {match_row['date']}")

    # Form
    st.subheader("Teamform (letzte 5 Spiele)")
    f_home = compute_form(matches, match_row['team_home'], last_n=5)
    f_away = compute_form(matches, match_row['team_away'], last_n=5)
    st.write(f"{match_row['team_home']}: {f_home['points']} Punkte — Ergebnisse: {', '.join(f_home['results']) if f_home['results'] else 'keine'}")
    st.write(f"{match_row['team_away']}: {f_away['points']} Punkte — Ergebnisse: {', '.join(f_away['results']) if f_away['results'] else 'keine'}")

    # Table
    st.subheader("Tabellenplatz")
    table = get_table(league, season)
    if not table.empty:
        subset = table[table['team'].isin([match_row['team_home'], match_row['team_away']])]
        if not subset.empty:
            st.dataframe(subset)
        else:
            st.write("Teams nicht in Tabellen-Daten gefunden.")
    else:
        st.write("Tabellen-Daten nicht verfügbar.")

    # Head-to-head
    st.subheader("Direkter Vergleich (Head-to-Head)")
    h2h = head2head(matches, match_row['team_home'], match_row['team_away'], last_n=20)
    if not h2h.empty:
        st.dataframe(h2h[['date','team_home','team_away','goals_home','goals_away','finished']])
        # summary
        played = len(h2h)
        home_wins = 0; away_wins = 0; draws = 0
        gh = 0; ga = 0
        for _, r in h2h.iterrows():
            if pd.isna(r['goals_home']) or pd.isna(r['goals_away']):
                continue
            gh += r['goals_home'] if r['team_home']==match_row['team_home'] else r['goals_away']
            ga += r['goals_away'] if r['team_away']==match_row['team_away'] else r['goals_home']
            if (r['team_home']==match_row['team_home'] and r['goals_home']>r['goals_away']) or (r['team_away']==match_row['team_home'] and r['goals_away']>r['goals_home']):
                home_wins += 1
            elif r['goals_home']==r['goals_away']:
                draws += 1
            else:
                away_wins += 1
        st.write(f"Letzte {played} Duelle: {match_row['team_home']} Siege: {home_wins}, Unentschieden: {draws}, {match_row['team_away']} Siege: {away_wins}")
    else:
        st.write("Keine direkten Begegnungen in den geladenen Daten.")

    # Prediction
    st.subheader("Vorhersage (Poisson-basierter Ansatz)")
    pred = predict_poisson(matches, match_row['team_home'], match_row['team_away'])
    st.metric("Geschätzte Erwartung Heimsiege (p)", f"{pred['p_home']*100:.1f}%")
    st.metric("Geschätzte Erwartung Unentschieden (p)", f"{pred['p_draw']*100:.1f}%")
    st.metric("Geschätzte Erwartung Auswärtssieg (p)", f"{pred['p_away']*100:.1f}%")

    # Visual
    fig1, ax1 = plt.subplots()
    ax1.bar([match_row['team_home'], 'Unentschieden', match_row['team_away']], [pred['p_home'], pred['p_draw'], pred['p_away']])
    ax1.set_ylabel('Wahrscheinlichkeit')
    st.pyplot(fig1)

    # Bankroll & Odds input
    st.subheader("Bankroll & Wettempfehlung")
    col1, col2 = st.columns(2)
    with col1:
        user_bankroll = st.number_input("Bankroll (€)", value=float(bankroll), min_value=1.0)
        odds_home = st.number_input(f"Quote Heimsieg ({match_row['team_home']})", min_value=1.01, value=0.0, format="%.2f")
        odds_draw = st.number_input("Quote Unentschieden", min_value=1.01, value=0.0, format="%.2f")
        odds_away = st.number_input(f"Quote Auswärtssieg ({match_row['team_away']})", min_value=1.01, value=0.0, format="%.2f")
    with col2:
        kelly_fraction_input = st.slider("Kelly-Fraction (Bruchteil, konservativ)", min_value=0.0, max_value=1.0, value=0.25)
        st.write("Wenn keine Quoten angegeben werden, wird ein konservativer Flat-Stake-Vorschlag gemacht.")

    # Recommendation logic
    outcomes = [
        ("1", pred['p_home'], odds_home if odds_home>=1.01 else None),
        ("X", pred['p_draw'], odds_draw if odds_draw>=1.01 else None),
        ("2", pred['p_away'], odds_away if odds_away>=1.01 else None)
    ]

    # If market odds provided, compute Kelly for each
    kelly_results = []
    for label, p, odds in outcomes:
        if odds is not None:
            f = kelly_fraction(p, odds)
            f_adj = f * kelly_fraction_input
            stake = round(user_bankroll * f_adj, 2)
            kelly_results.append({'label': label, 'p': p, 'odds': odds, 'kelly_f': f, 'kelly_f_adj': f_adj, 'stake': stake})

    st.subheader("Vorschläge")
    if kelly_results:
        # pick best edge stake > 0
        best = max(kelly_results, key=lambda x: x['stake'])
        st.write("**Kelly-basierte Empfehlung (unter Nutzung eingegebener Quoten)**")
        st.write(pd.DataFrame(kelly_results))
        if best['stake'] > 0:
            risk = 'niedrig' if best['p']>0.6 else ('mittel' if best['p']>0.45 else 'hoch')
            st.success(f"Empfohlen: {best['label']} — Einsatz: €{best['stake']} (Kelly-Full {best['kelly_f']:.3f}, adj {best['kelly_f_adj']:.3f}) — Risiko: {risk}")
        else:
            st.info("Keine positive Kelly-Empfehlung mit den angegebenen Quoten (kein Value).")
    else:
        # No market odds — give conservative flat-stake suggestion based on model probability
        p_vals = {'1': pred['p_home'], 'X': pred['p_draw'], '2': pred['p_away']}
        best_label = max(p_vals, key=lambda k: p_vals[k])
        best_p = p_vals[best_label]
        # simple stake rules
        if best_p > 0.65:
            stake = round(user_bankroll * 0.03, 2)
            risk = 'niedrig'
        elif best_p > 0.55:
            stake = round(user_bankroll * 0.02, 2)
            risk = 'mittel'
        elif best_p > 0.45:
            stake = round(user_bankroll * 0.01, 2)
            risk = 'mittel'
        else:
            stake = round(user_bankroll * 0.005, 2)
            risk = 'hoch'
        st.write("**Conservative Flat-Stake Empfehlung (ohne Bookmaker-Quoten)**")
        st.write(f"Empfohlen: {best_label}, geschätzte Gewinnwahrscheinlichkeit: {best_p*100:.1f}%, Einsatztipp: €{stake} — Risiko: {risk}")

    st.markdown("---")
    st.caption("Hinweis: Modell ist einfach und dient nur zu Analyse-/Lernzwecken. Wetten bergen Risiko.")

if __name__ == '__main__':
    main()