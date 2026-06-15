import pandas as pd
import numpy as np
import sqlite3
from typing import Optional
from src.data.database import load_all_matches, get_connection

def get_rolling_stats(team: str, date: str, df: pd.DataFrame = None, n_matches: int = 5, metric: str = 'goals', conn: Optional[sqlite3.Connection] = None) -> tuple:
    """Gets rolling statistics (for and against) for a team before a given date.
    Supports metrics: 'goals', 'corners', 'cards'.
    For 'goals', queries 'matches' table or uses in-memory 'df'.
    For 'corners' and 'cards', queries 'match_stats' table first, and if not enough matches (or empty),
    falls back to squad_stats tournament averages. Reuses connection if provided.
    """
    if df is not None and not df.empty:
        # Fallback to in-memory DataFrame lookup for compatibility
        df_clean = df.dropna(subset=['home_score', 'away_score'])
        team_df = df_clean[
            ((df_clean['home_team'] == team) | (df_clean['away_team'] == team)) & 
            (df_clean['date'] < date)
        ].sort_values('date', ascending=False)
        
        if len(team_df) == 0:
            return 0.0, 0.0
            
        recent_matches = team_df.head(n_matches)
        val_for = []
        val_against = []
        
        for _, match in recent_matches.iterrows():
            if metric == 'goals':
                if match['home_team'] == team:
                    val_for.append(match['home_score'])
                    val_against.append(match['away_score'])
                else:
                    val_for.append(match['away_score'])
                    val_against.append(match['home_score'])
            elif metric == 'corners':
                h_corn = match.get('home_corners', None)
                a_corn = match.get('away_corners', None)
                if h_corn is not None and a_corn is not None:
                    if match['home_team'] == team:
                        val_for.append(h_corn)
                        val_against.append(a_corn)
                    else:
                        val_for.append(a_corn)
                        val_against.append(h_corn)
            elif metric == 'cards':
                h_yc = match.get('home_yellow_cards', 0)
                a_yc = match.get('away_yellow_cards', 0)
                h_rc = match.get('home_red_cards', 0)
                a_rc = match.get('away_red_cards', 0)
                if match['home_team'] == team:
                    val_for.append(h_yc + h_rc)
                    val_against.append(a_yc + a_rc)
                else:
                    val_for.append(a_yc + a_rc)
                    val_against.append(h_yc + h_rc)
                    
        if len(val_for) > 0:
            return float(np.mean(val_for)), float(np.mean(val_against))
            
    # Database query logic
    conn_to_use = conn if conn is not None else get_connection()
    try:
        cursor = conn_to_use.cursor()
        
        if metric == 'goals':
            cursor.execute("""
                SELECT home_team, away_team, home_score, away_score 
                FROM matches 
                WHERE (home_team = ? OR away_team = ?) 
                  AND date < ? 
                  AND home_score IS NOT NULL 
                  AND away_score IS NOT NULL 
                ORDER BY date DESC LIMIT ?
            """, (team, team, date, n_matches))
            rows = cursor.fetchall()
            
            if not rows:
                return 0.0, 0.0
                
            gf = []
            ga = []
            for r in rows:
                if r['home_team'] == team:
                    gf.append(r['home_score'])
                    ga.append(r['away_score'])
                else:
                    gf.append(r['away_score'])
                    ga.append(r['home_score'])
            return float(np.mean(gf)), float(np.mean(ga))
            
        elif metric == 'corners':
            cursor.execute("""
                SELECT home_team, away_team, home_corners, away_corners 
                FROM match_stats 
                WHERE (home_team = ? OR away_team = ?) 
                  AND date < ? 
                  AND home_corners IS NOT NULL 
                  AND away_corners IS NOT NULL 
                ORDER BY date DESC LIMIT ?
            """, (team, team, date, n_matches))
            rows = cursor.fetchall()
            
            if rows:
                cf = []
                ca = []
                for r in rows:
                    if r['home_team'] == team:
                        cf.append(r['home_corners'])
                        ca.append(r['away_corners'])
                    else:
                        cf.append(r['away_corners'])
                        ca.append(r['home_corners'])
                return float(np.mean(cf)), float(np.mean(ca))
            else:
                # Fallback to squad_stats
                cursor.execute("""
                    SELECT corners_per_match 
                    FROM squad_stats 
                    WHERE team = ? COLLATE NOCASE
                    ORDER BY year DESC LIMIT 1
                """, (team,))
                row = cursor.fetchone()
                if row:
                    return float(row['corners_per_match']), 4.5
                else:
                    return 4.5, 4.5
                    
        elif metric == 'cards':
            cursor.execute("""
                SELECT home_team, away_team, 
                       home_yellow_cards, away_yellow_cards,
                       home_red_cards, away_red_cards
                FROM match_stats 
                WHERE (home_team = ? OR away_team = ?) 
                  AND date < ? 
                  AND home_yellow_cards IS NOT NULL 
                ORDER BY date DESC LIMIT ?
            """, (team, team, date, n_matches))
            rows = cursor.fetchall()
            
            if rows:
                cards_for = []
                cards_against = []
                for r in rows:
                    h_yc = r['home_yellow_cards'] or 0
                    a_yc = r['away_yellow_cards'] or 0
                    h_rc = r['home_red_cards'] or 0
                    a_rc = r['away_red_cards'] or 0
                    
                    if r['home_team'] == team:
                        cards_for.append(h_yc + h_rc)
                        cards_against.append(a_yc + a_rc)
                    else:
                        cards_for.append(a_yc + a_rc)
                        cards_against.append(h_yc + h_rc)
                return float(np.mean(cards_for)), float(np.mean(cards_against))
            else:
                # Fallback to squad_stats
                cursor.execute("""
                    SELECT yellow_cards_per_match, red_cards_per_match 
                    FROM squad_stats 
                    WHERE team = ? COLLATE NOCASE
                    ORDER BY year DESC LIMIT 1
                """, (team,))
                row = cursor.fetchone()
                if row:
                    yc = float(row['yellow_cards_per_match']) if row['yellow_cards_per_match'] is not None else 1.8
                    rc = float(row['red_cards_per_match']) if row['red_cards_per_match'] is not None else 0.05
                    return yc + rc, 1.85
                else:
                    return 1.85, 1.85
    finally:
        if conn is None:
            conn_to_use.close()
                
    return 0.0, 0.0

def build_features_dataset() -> pd.DataFrame:
    """Builds a symmetric dataset with ELO, market values, tactical stats (possession, xG),
    and rolling goal statistics for training the unified predictive model.
    Each match is split into two rows (perspective of self and opponent).
    Uses precomputed dictionaries in memory to optimize performance.
    """
    df_matches = load_all_matches()
    if len(df_matches) == 0:
        print("No matches in database to build features.")
        return pd.DataFrame()
        
    df_matches = df_matches.dropna(subset=['home_score', 'away_score'])
    
    print(f"Building symmetric features for {len(df_matches)} matches...")
    
    # Precompute team histories for O(1) lookups of goals
    team_history = {}
    for _, row in df_matches.iterrows():
        date = row['date']
        home = row['home_team']
        away = row['away_team']
        home_score = float(row['home_score'])
        away_score = float(row['away_score'])
        
        if home not in team_history:
            team_history[home] = []
        if away not in team_history:
            team_history[away] = []
            
        team_history[home].append({'date': date, 'gf': home_score, 'ga': away_score})
        team_history[away].append({'date': date, 'gf': away_score, 'ga': home_score})
        
    for team in team_history:
        team_history[team].sort(key=lambda x: x['date'])
        
    # Precompute ELO histories for O(1) lookups
    conn = get_connection()
    df_elo = pd.read_sql_query("SELECT team, date, elo FROM team_elo ORDER BY date ASC", conn)
    # Precompute squad values for O(1) lookups
    df_values = pd.read_sql_query("SELECT team, market_value_euros FROM squad_values", conn)
    # Precompute squad stats for O(1) tactical features lookups
    df_squad_stats = pd.read_sql_query("SELECT * FROM squad_stats", conn)
    conn.close()
    
    elo_cache = {}
    for _, row in df_elo.iterrows():
        team = row['team']
        date = row['date']
        elo = float(row['elo'])
        
        if team not in elo_cache:
            elo_cache[team] = []
        elo_cache[team].append((date, elo))
        
    squad_values_cache = {row['team'].lower(): float(row['market_value_euros']) for _, row in df_values.iterrows()}
    
    squad_stats_cache = {}
    for _, row in df_squad_stats.iterrows():
        t = row['team'].lower()
        if t not in squad_stats_cache:
            squad_stats_cache[t] = []
        squad_stats_cache[t].append(row.to_dict())
        
    for t in squad_stats_cache:
        squad_stats_cache[t].sort(key=lambda x: x['year'])
        
    def get_rolling_stats_opt(team: str, date: str, n_matches: int) -> tuple:
        if team not in team_history:
            return 0.0, 0.0
            
        past_matches = [m for m in team_history[team] if m['date'] < date]
        if not past_matches:
            return 0.0, 0.0
            
        recent = past_matches[-n_matches:]
        gf = [m['gf'] for m in recent]
        ga = [m['ga'] for m in recent]
        return float(np.mean(gf)), float(np.mean(ga))
        
    def get_latest_elo_opt(team: str, date: str) -> float:
        if team not in elo_cache:
            return 1500.0
        history = elo_cache[team]
        for past_date, elo in reversed(history):
            if past_date < date:
                return elo
        return 1500.0
        
    def get_squad_value_opt(team: str) -> float:
        return squad_values_cache.get(team.lower(), 50000000.0)
        
    def get_latest_squad_stats_opt(team: str, year: int) -> dict:
        t = team.lower()
        if t not in squad_stats_cache:
            return {
                'possession': 50.0,
                'xg_per_match': 1.2
            }
        history = squad_stats_cache[t]
        latest = None
        for record in history:
            if record['year'] < year:
                latest = record
            else:
                break
        if latest:
            return {
                'possession': float(latest['possession']),
                'xg_per_match': float(latest['xg_per_match']) if latest['xg_per_match'] is not None else 1.2
            }
        return {
            'possession': float(history[0]['possession']),
            'xg_per_match': float(history[0]['xg_per_match']) if history[0]['xg_per_match'] is not None else 1.2
        }
        
    def get_conf_coeff(team_name: str) -> float:
        team_lower = team_name.lower()
        conmebol = {'argentina', 'brazil', 'uruguay', 'colombia', 'ecuador', 'paraguay', 'chile', 'peru', 'bolivia', 'venezuela'}
        uefa = {'germany', 'france', 'spain', 'portugal', 'england', 'belgium', 'netherlands', 'croatia', 'italy', 'switzerland', 
                'austria', 'sweden', 'scotland', 'czech republic', 'poland', 'slovakia', 'slovenia', 'romania', 'georgia', 
                'hungary', 'albania', 'norway', 'denmark', 'ukraine', 'bosnia and herzegovina', 'russia', 'turkey', 'greece', 'wales',
                'republic of ireland'}
        if team_lower in conmebol or team_lower in uefa:
            return 1.0
        concacaf = {'united states', 'usa', 'mexico', 'canada', 'jamaica', 'panama', 'costa rica', 'haiti', 'curaçao', 'curacao', 'el salvador', 'honduras', 'trinidad and tobago'}
        if team_lower in concacaf:
            return 0.85
        afc = {'japan', 'iran', 'south korea', 'australia', 'qatar', 'saudi arabia', 'iraq', 'jordan', 'uzbekistan', 'china'}
        if team_lower in afc:
            return 0.85
        caf = {'morocco', 'senegal', 'nigeria', 'ivory coast', 'egypt', 'ghana', 'cameroon', 'tunisia', 'south africa', 'algeria', 
               'dr congo', 'angola', 'mali', 'guinea', 'mauritania', 'namibia', 'cape verde', 'cabo verde', 'equatorial guinea', 
               'gambia', 'guinea-bissau', 'mozambique', 'zambia', 'tanzania'}
        if team_lower in caf:
            return 0.85
        ofc = {'new zealand'}
        if team_lower in ofc:
            return 0.70
        return 0.80

    # Build features for matches since 2010
    train_matches = df_matches[df_matches['date'] >= '2010-01-01'].copy()
    
    features = []
    
    for idx, row in train_matches.iterrows():
        date = row['date']
        home = row['home_team']
        away = row['away_team']
        neutral = int(row['neutral'])
        
        home_score = float(row['home_score'])
        away_score = float(row['away_score'])
        
        match_year = pd.to_datetime(date).year
        
        # Get ELO ratings before the match
        elo_home = get_latest_elo_opt(home, date)
        elo_away = get_latest_elo_opt(away, date)
        
        # Get ELO trend (6 months ago)
        date_dt = pd.to_datetime(date)
        date_6m_ago = (date_dt - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        elo_home_6m = get_latest_elo_opt(home, date_6m_ago)
        elo_away_6m = get_latest_elo_opt(away, date_6m_ago)
        elo_trend_home = elo_home - elo_home_6m
        elo_trend_away = elo_away - elo_away_6m
        
        # Get market values
        market_value_home = get_squad_value_opt(home)
        market_value_away = get_squad_value_opt(away)
        
        # Market value ratio
        mv_ratio_home = market_value_home / market_value_away if market_value_away > 0 else 1.0
        mv_ratio_away = market_value_away / market_value_home if market_value_home > 0 else 1.0
        
        # Confederation coefficients
        conf_coeff_home = get_conf_coeff(home)
        conf_coeff_away = get_conf_coeff(away)
        
        # Get squad stats
        stats_home = get_latest_squad_stats_opt(home, match_year)
        stats_away = get_latest_squad_stats_opt(away, match_year)
        
        # Get rolling stats
        home_gf_l5, home_ga_l5 = get_rolling_stats_opt(home, date, 5)
        away_gf_l5, away_ga_l5 = get_rolling_stats_opt(away, date, 5)
        
        home_gf_l10, home_ga_l10 = get_rolling_stats_opt(home, date, 10)
        away_gf_l10, away_ga_l10 = get_rolling_stats_opt(away, date, 10)
        
        # ROW 1: Home team's perspective
        is_home_val_1 = 1.0 if not neutral else 0.0
        outcome_1 = 2 if home_score > away_score else (1 if home_score == away_score else 0)
        features.append({
            'date': date,
            'team': home,
            'opponent': away,
            'is_home': is_home_val_1,
            'elo_self': elo_home,
            'elo_opponent': elo_away,
            'elo_diff': elo_home - elo_away,
            'elo_trend_self': elo_trend_home,
            'elo_trend_opponent': elo_trend_away,
            'elo_trend_diff': elo_trend_home - elo_trend_away,
            'market_value_self': market_value_home,
            'market_value_opponent': market_value_away,
            'market_value_diff': market_value_home - market_value_away,
            'market_value_ratio': mv_ratio_home,
            'conf_coeff_self': conf_coeff_home,
            'conf_coeff_opponent': conf_coeff_away,
            'conf_coeff_diff': conf_coeff_home - conf_coeff_away,
            'poss_style_self': stats_home['possession'],
            'poss_style_opponent': stats_away['possession'],
            'xg_style_self': stats_home['xg_per_match'],
            'xg_style_opponent': stats_away['xg_per_match'],
            'rolling_gf_self_l5': home_gf_l5,
            'rolling_ga_self_l5': home_ga_l5,
            'rolling_gf_opp_l5': away_gf_l5,
            'rolling_ga_opp_l5': away_ga_l5,
            'rolling_gf_self_l10': home_gf_l10,
            'rolling_ga_self_l10': home_ga_l10,
            'rolling_gf_opp_l10': away_gf_l10,
            'rolling_ga_opp_l10': away_ga_l10,
            'goals_scored': home_score,
            'match_outcome': outcome_1
        })
        
        # ROW 2: Away team's perspective
        is_home_val_2 = 0.0  # Away team is never home (and neutral is also 0.0)
        outcome_2 = 2 if away_score > home_score else (1 if home_score == away_score else 0)
        features.append({
            'date': date,
            'team': away,
            'opponent': home,
            'is_home': is_home_val_2,
            'elo_self': elo_away,
            'elo_opponent': elo_home,
            'elo_diff': elo_away - elo_home,
            'elo_trend_self': elo_trend_away,
            'elo_trend_opponent': elo_trend_home,
            'elo_trend_diff': elo_trend_away - elo_trend_home,
            'market_value_self': market_value_away,
            'market_value_opponent': market_value_home,
            'market_value_diff': market_value_away - market_value_home,
            'market_value_ratio': mv_ratio_away,
            'conf_coeff_self': conf_coeff_away,
            'conf_coeff_opponent': conf_coeff_home,
            'conf_coeff_diff': conf_coeff_away - conf_coeff_home,
            'poss_style_self': stats_away['possession'],
            'poss_style_opponent': stats_home['possession'],
            'xg_style_self': stats_away['xg_per_match'],
            'xg_style_opponent': stats_home['xg_per_match'],
            'rolling_gf_self_l5': away_gf_l5,
            'rolling_ga_self_l5': away_ga_l5,
            'rolling_gf_opp_l5': home_gf_l5,
            'rolling_ga_opp_l5': home_ga_l5,
            'rolling_gf_self_l10': away_gf_l10,
            'rolling_ga_self_l10': away_ga_l10,
            'rolling_gf_opp_l10': home_gf_l10,
            'rolling_ga_opp_l10': home_ga_l10,
            'goals_scored': away_score,
            'match_outcome': outcome_2
        })
        
        if len(features) % 10000 == 0:
            print(f"Processed {len(features) // 2} matches...")
            
    features_df = pd.DataFrame(features)
    print(f"Symmetric features dataset built successfully with {len(features_df)} rows.")
    return features_df

if __name__ == "__main__":
    df = build_features_dataset()
    if not df.empty:
        print(df.head())
