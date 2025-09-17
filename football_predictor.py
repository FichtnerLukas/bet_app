import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
import math
from scipy.stats import poisson
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Bundesliga Wettanalyse", layout="wide", page_icon="⚽")

# -----------------------------
# Configuration
# -----------------------------
# API Keys
API_FOOTBALL_KEY = "ddf9e34021ab3cea7823b028cc2bdd0b"
OPENWEATHER_API_KEY = "5e46b576669e120f9b2990ce341c8625"

# API Endpoints
API_FOOTBALL_URL = "https://api-football-v1.p.rapidapi.com/v3"
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5"

# Headers for API-Football
API_FOOTBALL_HEADERS = {
    "X-RapidAPI-Key": API_FOOTBALL_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# API Request Counter
if 'api_requests_count' not in st.session_state:
    st.session_state.api_requests_count = 0
    st.session_state.last_request_time = time.time()

# League configuration - Only 1. Bundesliga
LEAGUES = {
    "1. Bundesliga": {"openliga": "bl1", "api_football": 78}
}

# Team colors for visualizations
TEAM_COLORS = {
    "Bayern Munich": "#DC052D",
    "Borussia Dortmund": "#FDE100",
    "RB Leipzig": "#DD0F2D",
    "Bayer Leverkusen": "#E32219",
    "Eintracht Frankfurt": "#E20000",
    "VfL Wolfsburg": "#65C42D",
    "Union Berlin": "#E3010F",
    "SC Freiburg": "#E32219",
    "Borussia Mönchengladbach": "#000000",
    "1. FC Köln": "#ED1C24",
    "TSG Hoffenheim": "#1F5EBD",
    "Mainz 05": "#ED1C24",
    "Augsburg": "#BA3733",
    "VfB Stuttgart": "#E32219",
    "Bochum": "#005CA9",
    "Hertha Berlin": "#005CA9",
    "Schalke 04": "#004D9D",
    "Werder Bremen": "#1D9053"
}

# Default colors for teams not in the list
DEFAULT_HOME_COLOR = "#1f77b4"
DEFAULT_AWAY_COLOR = "#ff7f0e"

# -----------------------------
# API Functions with Request Limiting
# -----------------------------
def make_api_request(url, headers=None, params=None, timeout=10):
    """Make API request with rate limiting"""
    # Check if we've reached the daily limit
    if st.session_state.api_requests_count >= 100:
        st.error("Tägliches API-Limit von 100 Anfragen erreicht. Bitte versuchen Sie es morgen wieder.")
        return None
    
    # Ensure at least 1 second between requests
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_request_time
    if time_since_last < 1:
        time.sleep(1 - time_since_last)
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        st.session_state.api_requests_count += 1
        st.session_state.last_request_time = time.time()
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"API Connection Error: {str(e)}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_api_football_data(endpoint, params=None):
    """Fetch data from API-Football with error handling"""
    url = f"{API_FOOTBALL_URL}/{endpoint}"
    return make_api_request(url, headers=API_FOOTBALL_HEADERS, params=params)

@st.cache_data(ttl=3600)
def try_openliga_endpoint(endpoint: str) -> dict:
    """Try several base URLs until one returns valid JSON."""
    BASE_OPTIONS = [
        "https://api.openligadb.de",
        "https://www.openligadb.de/api",
        "https://www.openligadb.de"
    ]
    
    last_err = None
    for base in BASE_OPTIONS:
        url = base.rstrip('/') + '/' + endpoint.lstrip('/')
        try:
            response = make_api_request(url, timeout=10)
            if response is not None:
                return response
            else:
                last_err = f"Request failed for {url}"
        except Exception as e:
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
# Data Fetching Functions
# -----------------------------
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_matches(league_short: str, season: int) -> pd.DataFrame:
    """Fetch match data from API-Football"""
    # Try API-Football first
    league_id = LEAGUES["1. Bundesliga"]["api_football"]
    
    response = get_api_football_data("fixtures", {"league": league_id, "season": season})
    if response and "response" in response:
        rows = []
        for fixture in response["response"]:
            fixture_data = fixture["fixture"]
            teams_data = fixture["teams"]
            score_data = fixture["score"]
            goals_data = fixture["goals"]
            
            # Determine if match is finished
            status = fixture_data["status"]["short"]
            finished = status in ["FT", "AET", "PEN"]
            
            # Get scores
            goals_home = goals_data["home"] if goals_data["home"] is not None else None
            goals_away = goals_data["away"] if goals_data["away"] is not None else None
            
            rows.append({
                'match_id': fixture_data["id"],
                'date': fixture_data["date"],
                'team_home': teams_data["home"]["name"],
                'team_away': teams_data["away"]["name"],
                'goals_home': goals_home,
                'goals_away': goals_away,
                'finished': finished,
                'venue': fixture_data["venue"]["name"] if "venue" in fixture_data and fixture_data["venue"] else "Unknown",
                'referee': fixture_data["referee"] if "referee" in fixture_data else "Unknown",
                'status': status,
                'raw': fixture
            })
        
        df = pd.DataFrame(rows)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    
    # Fallback to OpenLigaDB if API-Football fails
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
        match_id = nm.get('matchid') or nm.get('matchid')
        date = nm.get('matchdatetime') or nm.get('matchdateutc') or nm.get('matchdate')
        
        def teamname(t):
            if not t:
                return None
            if isinstance(t, dict):
                return t.get('teamname') or t.get('name')
            return t
        
        team1 = teamname(nm.get('team1'))
        team2 = teamname(nm.get('team2'))

        goals_home = None
        goals_away = None
        mr = nm.get('matchresults') or []
        if isinstance(mr, list) and mr:
            try:
                chosen = max(mr, key=lambda x: x.get('resultorderid') if isinstance(x, dict) else 0)
            except Exception:
                chosen = mr[-1]
            cm = normalize(chosen)
            goals_home = cm.get('pointsteam1') or cm.get('points_team1') or cm.get('points') and None
            goals_away = cm.get('pointsteam2') or cm.get('points_team2') or None
            try:
                goals_home = int(goals_home) if goals_home is not None else None
            except Exception:
                goals_home = None
            try:
                goals_away = int(goals_away) if goals_away is not None else None
            except Exception:
                goals_away = None

        finished = nm.get('matchisfinished') or nm.get('matchfinished') or nm.get('matchidfinished') or False

        rows.append({
            'match_id': match_id,
            'date': date,
            'team_home': team1,
            'team_away': team2,
            'goals_home': goals_home,
            'goals_away': goals_away,
            'finished': bool(finished),
            'venue': 'Unknown',
            'referee': 'Unknown',
            'status': 'Finished' if finished else 'Scheduled',
            'raw': m
        })
    
    df = pd.DataFrame(rows)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_table(league_short: str, season: int) -> pd.DataFrame:
    """Fetch league table from API-Football"""
    league_id = LEAGUES["1. Bundesliga"]["api_football"]
    
    response = get_api_football_data("standings", {"league": league_id, "season": season})
    if response and "response" in response:
        rows = []
        for standing in response["response"]:
            if "league" in standing and "standings" in standing["league"]:
                for table in standing["league"]["standings"]:
                    for team in table:
                        rows.append({
                            'team': team["team"]["name"],
                            'rank': team["rank"],
                            'points': team["points"],
                            'goals': team["all"]["goals"]["for"],
                            'opponent_goals': team["all"]["goals"]["against"],
                            'diff': team["goalsDiff"],
                            'form': team["form"],
                            'played': team["all"]["played"]
                        })
        return pd.DataFrame(rows)
    
    # Fallback to OpenLigaDB if API-Football fails
    endpoints = [
        f"getbltable/{league_short}/{season}",
        f"api/getbltable/{league_short}/{season}",
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
            'diff': nt.get('goaldiff'),
            'form': '',
            'played': nt.get('played', 0)
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def get_weather_data(lat: float, lon: float, date: datetime.datetime) -> dict:
    """Get weather data for a specific location and date"""
    # For historical data (past dates)
    if date.date() < datetime.datetime.now().date():
        url = f"{OPENWEATHER_URL}/weather"
    else:
        # For future dates, use forecast
        url = f"{OPENWEATHER_URL}/forecast"
    
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "de"
    }
    
    try:
        response = make_api_request(url, params=params)
        if response is None:
            return {}
            
        if "list" in response:  # Forecast data
            # Find the closest forecast to our match time
            closest_forecast = min(response["list"], key=lambda x: abs(pd.to_datetime(x["dt_txt"]) - date))
            weather = {
                "temp": closest_forecast["main"]["temp"],
                "feels_like": closest_forecast["main"]["feels_like"],
                "humidity": closest_forecast["main"]["humidity"],
                "conditions": closest_forecast["weather"][0]["description"],
                "wind_speed": closest_forecast["wind"]["speed"],
                "precipitation": closest_forecast.get("rain", {}).get("3h", 0) if "rain" in closest_forecast else 0
            }
        else:  # Current weather
            weather = {
                "temp": response["main"]["temp"],
                "feels_like": response["main"]["feels_like"],
                "humidity": response["main"]["humidity"],
                "conditions": response["weather"][0]["description"],
                "wind_speed": response["wind"]["speed"],
                "precipitation": response.get("rain", {}).get("1h", 0) if "rain" in response else 0
            }
        
        return weather
    except Exception as e:
        st.error(f"Weather API Error: {str(e)}")
        return {}

# -----------------------------
# Analysis Helpers (Enhanced)
# -----------------------------
def compute_form(matches_df: pd.DataFrame, team: str, last_n: int = 5) -> dict:
    df = matches_df.copy()
    df = df[df['finished'] == True]
    df = df[((df['team_home'] == team) | (df['team_away'] == team))].sort_values('date', ascending=False).head(last_n)
    points = 0
    results = []
    goals_for = 0
    goals_against = 0
    
    for _, r in df.iterrows():
        if pd.isna(r['goals_home']) or pd.isna(r['goals_away']):
            continue
        if r['team_home'] == team:
            goals_for += r['goals_home']
            goals_against += r['goals_away']
            if r['goals_home'] > r['goals_away']:
                points += 3; results.append('W')
            elif r['goals_home'] == r['goals_away']:
                points += 1; results.append('D')
            else:
                results.append('L')
        else:
            goals_for += r['goals_away']
            goals_against += r['goals_home']
            if r['goals_away'] > r['goals_home']:
                points += 3; results.append('W')
            elif r['goals_away'] == r['goals_home']:
                points += 1; results.append('D')
            else:
                results.append('L')
    
    return {
        'points': points, 
        'results': results, 
        'matches_considered': len(df),
        'goals_for': goals_for,
        'goals_against': goals_against,
        'goal_difference': goals_for - goals_against
    }

def head2head(matches_df: pd.DataFrame, home: str, away: str, last_n: int = 10) -> pd.DataFrame:
    df = matches_df.copy()
    cond = ((df['team_home'] == home) & (df['team_away'] == away)) | ((df['team_home'] == away) & (df['team_away'] == home))
    h2h = df[cond].sort_values('date', ascending=False).head(last_n)
    return h2h

def poisson_prob_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    probs_home = [poisson.pmf(k, lam_home) for k in range(max_goals+1)]
    probs_away = [poisson.pmf(k, lam_away) for k in range(max_goals+1)]
    mat = np.outer(probs_home, probs_away)
    return mat

def predict_poisson(matches_df: pd.DataFrame, home: str, away: str) -> dict:
    # compute averages
    df = matches_df[matches_df['finished'] == True]
    
    # Home team stats
    home_home_games = df[(df['team_home'] == home)]
    home_away_games = df[(df['team_away'] == home)]
    
    # Away team stats
    away_away_games = df[(df['team_away'] == away)]
    away_home_games = df[(df['team_home'] == away)]
    
    # Calculate attack and defense strengths
    try:
        home_attack = home_home_games['goals_home'].mean() if not home_home_games.empty else 1.5
        home_defense = home_home_games['goals_away'].mean() if not home_home_games.empty else 1.2
        
        away_attack = away_away_games['goals_away'].mean() if not away_away_games.empty else 1.2
        away_defense = away_away_games['goals_home'].mean() if not away_away_games.empty else 1.5
        
        # League averages
        league_home_avg = df['goals_home'].mean()
        league_away_avg = df['goals_away'].mean()
        
        # Calculate expected goals
        home_xg = (home_attack / league_home_avg) * (away_defense / league_away_avg) * league_home_avg
        away_xg = (away_attack / league_away_avg) * (home_defense / league_home_avg) * league_away_avg
        
        # Adjust for small sample sizes
        home_xg = max(0.2, min(4.0, home_xg))
        away_xg = max(0.2, min(4.0, away_xg))
        
    except Exception:
        home_xg, away_xg = 1.5, 1.2

    mat = poisson_prob_matrix(home_xg, away_xg, max_goals=10)
    p_home = np.tril(mat, -1).sum()
    p_draw = np.trace(mat)
    p_away = np.triu(mat, 1).sum()
    
    # Calculate exact score probabilities
    score_probs = {}
    for i in range(0, 6):
        for j in range(0, 6):
            score_probs[f"{i}-{j}"] = mat[i, j]
    
    # Calculate over/under probabilities
    total_goals_probs = {}
    for threshold in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        over_prob = 0
        under_prob = 0
        for i in range(0, 11):
            for j in range(0, 11):
                total = i + j
                if total > threshold:
                    over_prob += mat[i, j]
                elif total < threshold:
                    under_prob += mat[i, j]
        total_goals_probs[f"over_{threshold}"] = over_prob
        total_goals_probs[f"under_{threshold}"] = under_prob

    return {
        'lam_home': home_xg,
        'lam_away': away_xg,
        'p_home': float(p_home),
        'p_draw': float(p_draw),
        'p_away': float(p_away),
        'score_probs': score_probs,
        'total_goals_probs': total_goals_probs,
        'probability_matrix': mat
    }

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
    
    return {
        'xg_for': xg_for,
        'xg_against': xg_against,
        'matches_considered': len(goals)
    }

# -----------------------------
# Enhanced Visualization functions with Plotly
# -----------------------------
def plot_score_probabilities(score_probs, max_scores=5):
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    top_scores = sorted_scores[:max_scores]
    
    scores = [f"{score[0]}" for score in top_scores]
    probabilities = [score[1] * 100 for score in top_scores]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=probabilities,
        marker_color='#1f77b4',
        text=[f'{p:.1f}%' for p in probabilities],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Wahrscheinlichste Spielergebnisse',
        xaxis_title='Ergebnis',
        yaxis_title='Wahrscheinlichkeit (%)',
        template='plotly_white',
        font=dict(size=12),
        height=400
    )
    
    return fig

def plot_total_goals_probabilities(total_goals_probs):
    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    over_probs = [total_goals_probs[f"over_{t}"] * 100 for t in thresholds]
    under_probs = [total_goals_probs[f"under_{t}"] * 100 for t in thresholds]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=thresholds,
        y=over_probs,
        name='Über',
        marker_color='#2ca02c',
        text=[f'{p:.1f}%' for p in over_probs],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        x=thresholds,
        y=under_probs,
        name='Unter',
        marker_color='#d62728',
        text=[f'{p:.1f}%' for p in under_probs],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Wahrscheinlichkeit für Gesamttore (Über/Unter)',
        xaxis_title='Torgrenze',
        yaxis_title='Wahrscheinlichkeit (%)',
        barmode='group',
        template='plotly_white',
        font=dict(size=12),
        height=400
    )
    
    return fig

def plot_xg_comparison(xg_home, xg_away, home_team, away_team):
    categories = ['xG Für', 'xG Gegen']
    home_values = [xg_home['xg_for'], xg_home['xg_against']]
    away_values = [xg_away['xg_for'], xg_away['xg_against']]
    
    home_color = TEAM_COLORS.get(home_team, DEFAULT_HOME_COLOR)
    away_color = TEAM_COLORS.get(away_team, DEFAULT_AWAY_COLOR)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=home_team,
        x=categories,
        y=home_values,
        marker_color=home_color,
        text=[f'{v:.2f}' for v in home_values],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name=away_team,
        x=categories,
        y=away_values,
        marker_color=away_color,
        text=[f'{v:.2f}' for v in away_values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Erwartete Tore (xG) Vergleich',
        yaxis_title='Erwartete Tore',
        barmode='group',
        template='plotly_white',
        font=dict(size=12),
        height=400
    )
    
    return fig

def plot_match_outcome_probabilities(p_home, p_draw, p_away, home_team, away_team):
    outcomes = ['Heimsieg', 'Unentschieden', 'Auswärtssieg']
    probabilities = [p_home * 100, p_draw * 100, p_away * 100]
    
    home_color = TEAM_COLORS.get(home_team, DEFAULT_HOME_COLOR)
    away_color = TEAM_COLORS.get(away_team, DEFAULT_AWAY_COLOR)
    
    colors = [home_color, 'gray', away_color]
    
    fig = go.Figure(go.Bar(
        x=outcomes,
        y=probabilities,
        marker_color=colors,
        text=[f'{p:.1f}%' for p in probabilities],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f'Spielausgang: {home_team} vs {away_team}',
        xaxis_title='Ergebnis',
        yaxis_title='Wahrscheinlichkeit (%)',
        template='plotly_white',
        font=dict(size=12),
        height=400
    )
    
    return fig

def plot_team_form(results, team_name):
    form_sequence = results[::-1]  # Reverse to show oldest first
    
    colors = {'W': '#2ca02c', 'D': '#ff7f0e', 'L': '#d62728'}
    symbols = {'W': '▲', 'D': '■', 'L': '▼'}
    
    fig = go.Figure()
    
    for i, result in enumerate(form_sequence):
        fig.add_trace(go.Scatter(
            x=[i],
            y=[0],
            mode='markers+text',
            marker=dict(size=25, color=colors[result]),
            text=symbols[result],
            textfont=dict(size=15, color='white'),
            textposition='middle center',
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Spiel {i+1}: {result}"
        ))
    
    fig.update_layout(
        title=f'Form von {team_name} (letzte {len(form_sequence)} Spiele)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        template='plotly_white',
        height=150,
        width=len(form_sequence) * 60,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig

def plot_radar_chart(home_stats, away_stats, home_team, away_team):
    categories = ['xG Für', 'xG Gegen', 'Form', 'Tordifferenz']
    
    # Normalize values for radar chart (0-1 scale)
    max_xg_for = max(home_stats['xg_for'], away_stats['xg_for'])
    max_xg_against = max(home_stats['xg_against'], away_stats['xg_against'])
    
    home_values = [
        home_stats['xg_for'] / max_xg_for if max_xg_for > 0 else 0,
        home_stats['xg_against'] / max_xg_against if max_xg_against > 0 else 0,
        home_stats['form_points'] / 15 if home_stats['form_points'] > 0 else 0,  # Max 15 points from 5 games
        (home_stats['goal_difference'] + 10) / 20 if home_stats['goal_difference'] > -10 else 0  # Normalize from -10 to +10
    ]
    
    away_values = [
        away_stats['xg_for'] / max_xg_for if max_xg_for > 0 else 0,
        away_stats['xg_against'] / max_xg_against if max_xg_against > 0 else 0,
        away_stats['form_points'] / 15 if away_stats['form_points'] > 0 else 0,
        (away_stats['goal_difference'] + 10) / 20 if away_stats['goal_difference'] > -10 else 0
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=home_values,
        theta=categories,
        fill='toself',
        name=home_team,
        line_color=TEAM_COLORS.get(home_team, DEFAULT_HOME_COLOR)
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=away_values,
        theta=categories,
        fill='toself',
        name=away_team,
        line_color=TEAM_COLORS.get(away_team, DEFAULT_AWAY_COLOR)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Team-Vergleich (Radar-Diagramm)',
        template='plotly_white',
        height=400
    )
    
    return fig

# -----------------------------
# Bankroll / Kelly
# -----------------------------
def kelly_fraction(p: float, odds: float) -> float:
    if odds <= 1 or p <= 0:
        return 0.0
    b = odds - 1
    f = (p * b - (1 - p)) / b
    return max(f, 0.0)

# -----------------------------
# UI
# -----------------------------
st.title("⚽ Advanced Bundesliga Wettanalyse")

# Sidebar
st.sidebar.header("Konfiguration")
league_name = "1. Bundesliga"
league_short = LEAGUES[league_name]["openliga"]
current_year = datetime.datetime.now().year
season = st.sidebar.number_input("Saison", min_value=2000, max_value=current_year + 1, value=current_year)

# API usage indicator
st.sidebar.subheader("API-Nutzung")
st.sidebar.progress(st.session_state.api_requests_count / 100)
st.sidebar.write(f"{st.session_state.api_requests_count}/100 Anfragen heute")

# Main content
st.header(f"{league_name} — Saison {season}/{season+1}")

# Fetch data
with st.spinner("Lade Daten..."):
    matches_df = get_matches(league_short, season)
    table_df = get_table(league_short, season)

if matches_df.empty:
    st.error("Keine Spieldaten gefunden. Bitte überprüfen Sie die Saison.")
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
    match_date = match_data['date']
    venue = match_data.get('venue', 'Unbekannt')
    
    # Display match info in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Heim")
        st.markdown(f"**{home_team}**")
    with col2:
        st.subheader("vs")
        st.markdown(f"**{match_date.strftime('%d.%m.%Y %H:%M')}**")
        st.markdown(f"*{venue}*")
    with col3:
        st.subheader("Auswärts")
        st.markdown(f"**{away_team}**")
    
    # Weather information
    st.subheader("Wettervorhersage")
    # Use Munich as default location
    weather_data = get_weather_data(48.1351, 11.5820, match_date)
    
    if weather_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Temperatur", f"{weather_data['temp']:.1f}°C")
        with col2:
            st.metric("Luftfeuchtigkeit", f"{weather_data['humidity']}%")
        with col3:
            st.metric("Windgeschwindigkeit", f"{weather_data['wind_speed']} m/s")
        with col4:
            st.metric("Niederschlag", f"{weather_data['precipitation']} mm")
        
        st.write(f"**Bedingungen:** {weather_data['conditions'].capitalize()}")
    else:
        st.info("Wetterdaten nicht verfügbar.")
    
    # Team information
    st.subheader("Team-Informationen")
    
    # Get form data
    home_form = compute_form(matches_df, home_team)
    away_form = compute_form(matches_df, away_team)
    
    # Get table positions
    home_position = table_df[table_df['team'] == home_team]['rank'].iloc[0] if not table_df.empty and home_team in table_df['team'].values else "N/A"
    away_position = table_df[table_df['team'] == away_team]['rank'].iloc[0] if not table_df.empty and away_team in table_df['team'].values else "N/A"
    
    # Create columns for team info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {home_team}")
        st.plotly_chart(plot_team_form(home_form['results'], home_team), use_container_width=True)
        
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("Tabellenplatz", home_position)
        with col1b:
            st.metric("Form", f"{home_form['points']}/{home_form['matches_considered']*3}")
        with col1c:
            st.metric("Tordiff.", home_form['goal_difference'])
    
    with col2:
        st.markdown(f"### {away_team}")
        st.plotly_chart(plot_team_form(away_form['results'], away_team), use_container_width=True)
        
        col2a, col2b, col2c = st.columns(3)
        with col2a:
            st.metric("Tabellenplatz", away_position)
        with col2b:
            st.metric("Form", f"{away_form['points']}/{away_form['matches_considered']*3}")
        with col2c:
            st.metric("Tordiff.", away_form['goal_difference'])
    
    # Head-to-head
    st.subheader("Vergangene Begegnungen")
    h2h = head2head(matches_df, home_team, away_team)
    if not h2h.empty:
        h2h_display = h2h.copy()
        h2h_display['Datum'] = h2h_display['date'].dt.strftime('%d.%m.%Y')
        h2h_display['Ergebnis'] = h2h_display.apply(lambda x: f"{x['goals_home']} - {x['goals_away']}", axis=1)
        
        # Style the DataFrame
        st.dataframe(
            h2h_display[['Datum', 'team_home', 'team_away', 'Ergebnis']].rename(
                columns={'team_home': 'Heim', 'team_away': 'Auswärts'}
            ),
            hide_index=True,
            use_container_width=True
        )
        
        # Calculate H2H stats
        home_wins = len(h2h[(h2h['team_home'] == home_team) & (h2h['goals_home'] > h2h['goals_away'])])
        away_wins = len(h2h[(h2h['team_away'] == away_team) & (h2h['goals_away'] > h2h['goals_home'])])
        draws = len(h2h[h2h['goals_home'] == h2h['goals_away']])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home_team} Siege", home_wins)
        with col2:
            st.metric("Unentschieden", draws)
        with col3:
            st.metric(f"{away_team} Siege", away_wins)
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
    st.plotly_chart(plot_match_outcome_probabilities(
        prediction['p_home'], prediction['p_draw'], prediction['p_away'],
        home_team, away_team
    ), use_container_width=True)
    
    # Score probabilities
    st.plotly_chart(plot_score_probabilities(prediction['score_probs']), use_container_width=True)
    
    # Total goals probabilities
    st.plotly_chart(plot_total_goals_probabilities(prediction['total_goals_probs']), use_container_width=True)
    
    # Expected goals analysis
    st.subheader("Erwartete Tore (xG) Analyse")
    xg_home = estimate_expected_goals(matches_df, home_team, is_home=True)
    xg_away = estimate_expected_goals(matches_df, away_team, is_home=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{home_team} (Heim):**")
        st.metric("xG Für", f"{xg_home['xg_for']:.2f}")
        st.metric("xG Gegen", f"{xg_home['xg_against']:.2f}")
        st.write(f"Basierend auf {xg_home['matches_considered']} Spielen")
    
    with col2:
        st.markdown(f"**{away_team} (Auswärts):**")
        st.metric("xG Für", f"{xg_away['xg_for']:.2f}")
        st.metric("xG Gegen", f"{xg_away['xg_against']:.2f}")
        st.write(f"Basierend auf {xg_away['matches_considered']} Spielen")
    
    # xG comparison chart
    st.plotly_chart(plot_xg_comparison(xg_home, xg_away, home_team, away_team), use_container_width=True)
    
    # Radar chart for team comparison
    home_stats = {
        'xg_for': xg_home['xg_for'],
        'xg_against': xg_home['xg_against'],
        'form_points': home_form['points'],
        'goal_difference': home_form['goal_difference']
    }
    
    away_stats = {
        'xg_for': xg_away['xg_for'],
        'xg_against': xg_away['xg_against'],
        'form_points': away_form['points'],
        'goal_difference': away_form['goal_difference']
    }
    
    st.plotly_chart(plot_radar_chart(home_stats, away_stats, home_team, away_team), use_container_width=True)
    
    # Betting analysis
    st.subheader("Wettanalyse")
    col1, col2, col3 = st.columns(3)
    with col1:
        home_odds = st.number_input("Heimsieg-Quote", min_value=1.0, max_value=100.0, value=2.0, step=0.1, key="home_odds")
    with col2:
        draw_odds = st.number_input("Unentschieden-Quote", min_value=1.0, max_value=100.0, value=3.5, step=0.1, key="draw_odds")
    with col3:
        away_odds = st.number_input("Auswärtssieg-Quote", min_value=1.0, max_value=100.0, value=4.0, step=0.1, key="away_odds")
    
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
        st.metric("EV Heimsieg", f"{ev_home:.1f}%", "positiv" if ev_home > 0 else "negativ", delta_color="inverse")
    with col2:
        st.metric("EV Unentschieden", f"{ev_draw:.1f}%", "positiv" if ev_draw > 0 else "negativ", delta_color="inverse")
    with col3:
        st.metric("EV Auswärtssieg", f"{ev_away:.1f}%", "positiv" if ev_away > 0 else "negativ", delta_color="inverse")
    
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
    st.subheader("Aktuelle Tabelle")
    st.dataframe(table_df, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray;'>API-Anfragen heute: {st.session_state.api_requests_count}/100</div>",
    unsafe_allow_html=True
)