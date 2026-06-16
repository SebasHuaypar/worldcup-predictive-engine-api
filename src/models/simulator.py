import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.data.database import (
    get_connection, get_latest_elo, load_all_matches,
    get_squad_value, get_squad_absences, get_city_altitude, get_referee_stats
)
from src.features.builder import get_rolling_stats

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model_goals.pkl")
MODEL_OUTCOME_PATH = os.path.join(MODELS_DIR, "model_outcome_classifier.pkl")

def load_model() -> Any:
    """Loads the trained unified goal model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Goal prediction model not found. Please run training first.")
    return joblib.load(MODEL_PATH)

def load_outcome_model() -> Any:
    """Loads the trained outcome classifier model from disk."""
    if not os.path.exists(MODEL_OUTCOME_PATH):
        raise FileNotFoundError("Outcome classifier model not found. Please run training first.")
    return joblib.load(MODEL_OUTCOME_PATH)

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

def get_latest_squad_stats(team: str, conn: Optional[sqlite3.Connection] = None) -> Dict[str, float]:
    """Fetches the latest tournament stats for a team from the database.
    Returns defaults if the team has no recorded tournament statistics.
    Reuses connection if provided. Uses case-insensitive match.
    """
    def _execute(c):
        cursor = c.cursor()
        cursor.execute("""
            SELECT possession, shots_per_match, sot_per_match, 
                   corners_per_match, yellow_cards_per_match, red_cards_per_match, xg_per_match
            FROM squad_stats
            WHERE team = ? COLLATE NOCASE
            ORDER BY year DESC LIMIT 1
        """, (team,))
        row = cursor.fetchone()
        if row:
            return {
                'possession': row['possession'],
                'shots_per_match': row['shots_per_match'],
                'sot_per_match': row['sot_per_match'],
                'corners_per_match': row['corners_per_match'],
                'yellow_cards_per_match': row['yellow_cards_per_match'],
                'red_cards_per_match': row['red_cards_per_match'],
                'xg_per_match': row['xg_per_match'] if row['xg_per_match'] is not None else 1.2
            }
        else:
            return {
                'possession': 50.0,
                'shots_per_match': 11.5,
                'sot_per_match': 3.8,
                'corners_per_match': 4.5,
                'yellow_cards_per_match': 1.8,
                'red_cards_per_match': 0.05,
                'xg_per_match': 1.2
            }

    if conn is not None:
        return _execute(conn)
    from contextlib import closing
    with closing(get_connection()) as conn_new:
        return _execute(conn_new)

def predict_match_expectations(
    team_a: str, team_b: str, neutral: bool = True,
    missing_players_a: int = 0, missing_players_b: int = 0,
    city: str = None, country: str = None, referee: str = None
) -> Dict[str, Any]:
    """Predicts the expected goals, corners, and cards for both teams.
    Uses a single symmetric XGBoost model.
    Adjusts ELO rating based on missing players (weighted by severity).
    Incorporates altitude and referee card adjustments. Reuses connection.
    """
    from src.data.database import get_connection
    from contextlib import closing
    
    model = load_model()
    outcome_model = load_outcome_model()
    
    with closing(get_connection()) as conn:
        # Get ELO ratings (latest available)
        elo_a_raw = get_latest_elo(team_a, '9999-12-31', conn=conn)
        elo_b_raw = get_latest_elo(team_b, '9999-12-31', conn=conn)

        # Get active squad absences from DB
        abs_a = get_squad_absences(team_a, conn=conn)
        abs_b = get_squad_absences(team_b, conn=conn)
        
        # Apply injury/absence ELO reduction (weighted by severity)
        def calc_absence_penalty(abs_list):
            penalty = 0.0
            for ab in abs_list:
                sev = ab.get('severity', '').lower()
                if sev == 'star_player':
                    penalty += 0.020 # Moderate penalty for star player (2.0%)
                elif sev == 'major':
                    penalty += 0.010 # 1.0%
                else:
                    penalty += 0.005 # 0.5%
            return penalty
            
        elo_a = elo_a_raw * (1.0 - calc_absence_penalty(abs_a) - 0.0075 * missing_players_a)
        elo_b = elo_b_raw * (1.0 - calc_absence_penalty(abs_b) - 0.0075 * missing_players_b)
        # Prevent ELO from dropping below 60% of raw ELO
        elo_a = max(elo_a_raw * 0.60, elo_a)
        elo_b = max(elo_b_raw * 0.60, elo_b)
        
        # Get market values
        mv_a = get_squad_value(team_a, conn=conn)
        mv_b = get_squad_value(team_b, conn=conn)
        
        # Get squad stats for tactical styling features
        stats_a = get_latest_squad_stats(team_a, conn=conn)
        stats_b = get_latest_squad_stats(team_b, conn=conn)
        
        # Get rolling stats (corners and cards)
        roll_corners_a_for, _ = get_rolling_stats(team_a, '9999-12-31', None, 5, 'corners', conn=conn)
        roll_corners_b_for, _ = get_rolling_stats(team_b, '9999-12-31', None, 5, 'corners', conn=conn)
        
        roll_cards_a_for, _ = get_rolling_stats(team_a, '9999-12-31', None, 5, 'cards', conn=conn)
        roll_cards_b_for, _ = get_rolling_stats(team_b, '9999-12-31', None, 5, 'cards', conn=conn)
        
        # Helper to construct features for a team from their perspective
        def get_features_for_team(team, opponent, is_home_val, h_elo, a_elo, s_self, s_opp):
            home_gf_l5, home_ga_l5 = get_rolling_stats(team, '9999-12-31', None, 5, conn=conn)
            away_gf_l5, away_ga_l5 = get_rolling_stats(opponent, '9999-12-31', None, 5, conn=conn)
            
            home_gf_l10, home_ga_l10 = get_rolling_stats(team, '9999-12-31', None, 10, conn=conn)
            away_gf_l10, away_ga_l10 = get_rolling_stats(opponent, '9999-12-31', None, 10, conn=conn)
            
            mv_self = get_squad_value(team, conn=conn)
            mv_opp = get_squad_value(opponent, conn=conn)
            mv_ratio = mv_self / mv_opp if mv_opp > 0 else 1.0
            
            # 6 months ago date relative to max date in team_elo (excluding dummy future dates)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(date) as max_date FROM team_elo WHERE date < '9999-12-31'")
            max_date_row = cursor.fetchone()
            ref_date = max_date_row['max_date'] if (max_date_row and max_date_row['max_date']) else '2026-06-15'
            ref_date_dt = pd.to_datetime(ref_date)
            date_6m_ago = (ref_date_dt - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
            
            elo_self_6m = get_latest_elo(team, date_6m_ago, conn=conn)
            elo_opp_6m = get_latest_elo(opponent, date_6m_ago, conn=conn)
            
            raw_elo_self = get_latest_elo(team, '9999-12-31', conn=conn)
            raw_elo_opp = get_latest_elo(opponent, '9999-12-31', conn=conn)
            
            elo_trend_self_val = raw_elo_self - elo_self_6m
            elo_trend_opp_val = raw_elo_opp - elo_opp_6m
            
            conf_coeff_self_val = get_conf_coeff(team)
            conf_coeff_opp_val = get_conf_coeff(opponent)
            
            feature_cols = [
                'is_home', 'elo_self', 'elo_opponent', 'elo_diff',
                'elo_trend_self', 'elo_trend_opponent', 'elo_trend_diff',
                'market_value_self', 'market_value_opponent', 'market_value_diff', 'market_value_ratio',
                'conf_coeff_self', 'conf_coeff_opponent', 'conf_coeff_diff',
                'poss_style_self', 'poss_style_opponent', 'xg_style_self', 'xg_style_opponent',
                'rolling_gf_self_l5', 'rolling_ga_self_l5', 'rolling_gf_opp_l5', 'rolling_ga_opp_l5',
                'rolling_gf_self_l10', 'rolling_ga_self_l10', 'rolling_gf_opp_l10', 'rolling_ga_opp_l10'
            ]
            
            df_feat = pd.DataFrame([{
                'is_home': is_home_val,
                'elo_self': h_elo,
                'elo_opponent': a_elo,
                'elo_diff': h_elo - a_elo,
                'elo_trend_self': elo_trend_self_val,
                'elo_trend_opponent': elo_trend_opp_val,
                'elo_trend_diff': elo_trend_self_val - elo_trend_opp_val,
                'market_value_self': mv_self,
                'market_value_opponent': mv_opp,
                'market_value_diff': mv_self - mv_opp,
                'market_value_ratio': mv_ratio,
                'conf_coeff_self': conf_coeff_self_val,
                'conf_coeff_opponent': conf_coeff_opp_val,
                'conf_coeff_diff': conf_coeff_self_val - conf_coeff_opp_val,
                'poss_style_self': s_self['possession'],
                'poss_style_opponent': s_opp['possession'],
                'xg_style_self': s_self['xg_per_match'],
                'xg_style_opponent': s_opp['xg_per_match'],
                'rolling_gf_self_l5': home_gf_l5,
                'rolling_ga_self_l5': home_ga_l5,
                'rolling_gf_opp_l5': away_gf_l5,
                'rolling_ga_opp_l5': away_ga_l5,
                'rolling_gf_self_l10': home_gf_l10,
                'rolling_ga_self_l10': home_ga_l10,
                'rolling_gf_opp_l10': away_gf_l10,
                'rolling_ga_opp_l10': away_ga_l10
            }])
            return df_feat[feature_cols]
            
        # Run predictions simmetrically
        if neutral:
            # Setting is_home = 0.0 since it is neutral for both
            feat_a = get_features_for_team(team_a, team_b, 0.0, elo_a, elo_b, stats_a, stats_b)
            feat_b = get_features_for_team(team_b, team_a, 0.0, elo_b, elo_a, stats_b, stats_a)
        else:
            # Setting is_home = 1.0 for A (home) and 0.0 for B (away)
            feat_a = get_features_for_team(team_a, team_b, 1.0, elo_a, elo_b, stats_a, stats_b)
            feat_b = get_features_for_team(team_b, team_a, 0.0, elo_b, elo_a, stats_b, stats_a)
            
        lambda_goals_a = max(0.05, float(model.predict(feat_a)[0]))
        lambda_goals_b = max(0.05, float(model.predict(feat_b)[0]))
        
        # Get outcome classifier predictions (0: Loss, 1: Draw, 2: Win)
        prob_a = outcome_model.predict_proba(feat_a)[0]
        prob_b = outcome_model.predict_proba(feat_b)[0]
        
        # Combine symmetric predictions (A's Win is B's Loss, etc.)
        clf_win_a = float(0.5 * (prob_a[2] + prob_b[0]))
        clf_draw  = float(0.5 * (prob_a[1] + prob_b[1]))
        clf_win_b = float(0.5 * (prob_a[0] + prob_b[2]))
        
        # Blending goals model predictions with team market values
        # If one team has a significantly higher market value, we slightly adjust expected goals to counter ELO lag
        if mv_b > 0 and mv_a > 0:
            import math
            mv_ratio = mv_a / mv_b
            mv_log_diff = math.log10(mv_ratio) # e.g. if A is 10x B, log10 is 1.0; if A is 1/10x B, log10 is -1.0
            
            mean_goals = (lambda_goals_a + lambda_goals_b) / 2.0
            goals_diff = lambda_goals_a - lambda_goals_b
            
            # 0.3 log difference factor translates to about 0.3 goals difference shift for a 10x market value difference
            adjusted_diff = goals_diff + 0.3 * mv_log_diff
            
            lambda_goals_a = max(0.05, mean_goals + adjusted_diff / 2.0)
            lambda_goals_b = max(0.05, mean_goals - adjusted_diff / 2.0)
        
        # Estimate possession combining ELO and team's tactical style
        poss_style_a = stats_a['possession']
        poss_style_b = stats_b['possession']
        elo_diff = elo_a - elo_b
        poss_a = 50.0 + (poss_style_a - poss_style_b) * 0.5 + (elo_diff / 40.0) * 1.5
        poss_a = max(35.0, min(65.0, poss_a))
        poss_b = 100.0 - poss_a
        
        # Adjust expected corners using ELO difference and possession
        elo_mult_corners_a = max(0.2, 1.0 + 0.05 * (elo_diff / 400.0))
        elo_mult_corners_b = max(0.2, 1.0 - 0.05 * (elo_diff / 400.0))
        
        # Damp the possession multiplier to prevent extreme values for high-possession teams
        poss_mult_a = 1.0 + 0.4 * ((poss_a - 50.0) / 50.0)
        poss_mult_b = 1.0 + 0.4 * ((poss_b - 50.0) / 50.0)
        
        lambda_corners_a = float(roll_corners_a_for * poss_mult_a * elo_mult_corners_a)
        lambda_corners_b = float(roll_corners_b_for * poss_mult_b * elo_mult_corners_b)
        
        # Use a more realistic minimum expected corner limit (2.0 instead of 1.0)
        lambda_corners_a = max(2.0, lambda_corners_a)
        lambda_corners_b = max(2.0, lambda_corners_b)
        
        # Get referee stats or use default averages
        ref_stats_data = None
        if referee:
            ref_stats_data = get_referee_stats(referee, conn=conn)
            
        if ref_stats_data:
            ref_yellows = ref_stats_data['yellow_cards_per_match']
            ref_reds = ref_stats_data['red_cards_per_match']
        else:
            # Default averages across all referees in the database
            ref_yellows = 4.18
            ref_reds = 0.17
            
        ref_strictness = ref_yellows + 2.0 * ref_reds
        
        # Combine team baselines (30%) with referee strictness (70%)
        team_cards_sum = roll_cards_a_for + roll_cards_b_for
        expected_total_cards = 0.3 * team_cards_sum + 0.7 * ref_strictness
        
        # Calculate distribution weights based on ELO and inverse possession
        elo_mult_cards_a = max(0.2, 1.0 - 0.1 * (elo_diff / 400.0))
        elo_mult_cards_b = max(0.2, 1.0 + 0.1 * (elo_diff / 400.0))
        
        weight_a = (poss_b / 50.0) * elo_mult_cards_a
        weight_b = (poss_a / 50.0) * elo_mult_cards_b
        
        total_weight = weight_a + weight_b
        if total_weight > 0:
            share_a = weight_a / total_weight
            share_b = weight_b / total_weight
        else:
            share_a = 0.5
            share_b = 0.5
            
        lambda_cards_a = expected_total_cards * share_a
        lambda_cards_b = expected_total_cards * share_b
        lambda_cards_a = max(0.5, lambda_cards_a)
        lambda_cards_b = max(0.5, lambda_cards_b)
        
        # Altitude adjustment
        altitude = 0.0
        if city:
            altitude = get_city_altitude(city, country or "", conn=conn)
            
        if altitude > 1500:
            high_alt_teams = {'bolivia', 'ecuador', 'colombia', 'mexico'}
            acclimated_a = team_a.lower() in high_alt_teams
            acclimated_b = team_b.lower() in high_alt_teams
            
            poss_a_scale = 1.0
            poss_b_scale = 1.0
            
            if not acclimated_a:
                fatigue_a = 1.0 - 0.06 * ((altitude - 0.0) / 1000.0)
                fatigue_a = max(0.7, fatigue_a)
                lambda_goals_a *= fatigue_a
                lambda_cards_a *= (2.0 - fatigue_a)
                poss_a_scale = fatigue_a
                
            if not acclimated_b:
                fatigue_b = 1.0 - 0.06 * ((altitude - 0.0) / 1000.0)
                fatigue_b = max(0.7, fatigue_b)
                lambda_goals_b *= fatigue_b
                lambda_cards_b *= (2.0 - fatigue_b)
                poss_b_scale = fatigue_b
                
            # Re-balance possession
            if poss_a_scale != 1.0 or poss_b_scale != 1.0:
                new_poss_a = poss_a * poss_a_scale
                new_poss_b = poss_b * poss_b_scale
                total_scaled = new_poss_a + new_poss_b
                poss_a = (new_poss_a / total_scaled) * 100.0
                poss_b = 100.0 - poss_a
                
        # Referee adjustment (already integrated above)
        pass
                
    return {
        'team_a': team_a,
        'team_b': team_b,
        'elo_a': elo_a,
        'elo_b': elo_b,
        'elo_a_raw': elo_a_raw,
        'elo_b_raw': elo_b_raw,
        'possession_a': poss_a,
        'possession_b': poss_b,
        'exp_goals_a': lambda_goals_a,
        'exp_goals_b': lambda_goals_b,
        'exp_corners_a': lambda_corners_a,
        'exp_corners_b': lambda_corners_b,
        'exp_cards_a': lambda_cards_a,
        'exp_cards_b': lambda_cards_b,
        'altitude_meters': altitude,
        'referee_stats': ref_stats_data,
        'squad_value_a': mv_a,
        'squad_value_b': mv_b,
        'absences_a': abs_a,
        'absences_b': abs_b,
        'clf_win_a': clf_win_a,
        'clf_draw': clf_draw,
        'clf_win_b': clf_win_b
    }

def simulate_penalty_shootout(p_a: float, p_b: float, rng: np.random.Generator) -> str:
    """Simulates a penalty shootout round-by-round and returns the winner ('A' or 'B')."""
    score_a = 0
    score_b = 0
    
    # 5 initial rounds
    for r in range(5):
        # A shoots
        if rng.random() < p_a:
            score_a += 1
        rem_b = 5 - r
        if score_a > score_b + rem_b:
            return 'A'
        if score_b > score_a + (rem_b - 1):
            return 'B'
            
        # B shoots
        if rng.random() < p_b:
            score_b += 1
        rem_a = 4 - r
        if score_b > score_a + rem_a:
            return 'B'
        if score_a > score_b + rem_a:
            return 'A'
            
    # Sudden death
    while score_a == score_b:
        shot_a = rng.random() < p_a
        shot_b = rng.random() < p_b
        if shot_a and not shot_b:
            return 'A'
        elif shot_b and not shot_a:
            return 'B'
            
    return 'A' if score_a > score_b else 'B'

def run_monte_carlo_simulation(
    team_a: str, team_b: str, neutral: bool = True, n_sims: int = 10000,
    missing_players_a: int = 0, missing_players_b: int = 0, knockout: bool = False,
    city: str = None, country: str = None, referee: str = None
) -> Dict[str, Any]:
    """Runs a Monte Carlo simulation of the match and returns all predicted outcomes and probabilities.
    Vectorized extra-time calculations for high performance.
    """
    exp = predict_match_expectations(
        team_a, team_b, neutral, missing_players_a, missing_players_b,
        city=city, country=country, referee=referee
    )
    
    rng = np.random.default_rng(42)
    
    # Apply Dixon-Coles correlation adjustment for low-scoring matches (0-0, 1-0, 0-1, 1-1)
    lambda_a = exp['exp_goals_a']
    lambda_b = exp['exp_goals_b']
    
    # Precompute factorials up to 10
    import math
    factorials = np.array([math.factorial(i) for i in range(11)])
    
    # Compute Poisson PMFs
    pmf_a = (lambda_a**np.arange(11) * np.exp(-lambda_a)) / factorials
    pmf_b = (lambda_b**np.arange(11) * np.exp(-lambda_b)) / factorials
    
    # Outer product for joint independent probability matrix
    joint_prob = np.outer(pmf_a, pmf_b)
    
    # Dixon-Coles parameters: dynamic rho based on expected goals to model draw correlation
    rho = -0.22 * np.exp(-(lambda_a + lambda_b) / 2.0)
    tau = np.ones((11, 11))
    tau[0, 0] = max(0.0, 1.0 - lambda_a * lambda_b * rho)
    tau[1, 0] = max(0.0, 1.0 + lambda_b * rho)
    tau[0, 1] = max(0.0, 1.0 + lambda_a * rho)
    tau[1, 1] = max(0.0, 1.0 - rho)
    
    # Apply correlation adjustments and normalize
    adjusted_prob = joint_prob * tau
    adjusted_prob /= np.sum(adjusted_prob)
    
    # Sample outcomes from the joint distribution
    flat_probs = adjusted_prob.flatten()
    sampled_indices = rng.choice(121, size=n_sims, p=flat_probs)
    
    sim_goals_a = sampled_indices // 11
    sim_goals_b = sampled_indices % 11
    
    sim_corners_a = rng.poisson(exp['exp_corners_a'], n_sims)
    sim_corners_b = rng.poisson(exp['exp_corners_b'], n_sims)
    
    sim_cards_a = rng.poisson(exp['exp_cards_a'], n_sims)
    sim_cards_b = rng.poisson(exp['exp_cards_b'], n_sims)
    
    # 1. Match outcome probabilities (90 minutes) from Poisson
    mc_win_a = float(np.sum(sim_goals_a > sim_goals_b) / n_sims)
    mc_win_b = float(np.sum(sim_goals_b > sim_goals_a) / n_sims)
    mc_draw = float(np.sum(sim_goals_a == sim_goals_b) / n_sims)
    
    # Blended probabilities (50% Poisson Monte Carlo, 50% XGBoost Multiclass Classifier)
    clf_win_a = exp['clf_win_a']
    clf_draw = exp['clf_draw']
    clf_win_b = exp['clf_win_b']
    
    final_win_a = 0.5 * mc_win_a + 0.5 * clf_win_a
    final_draw = 0.5 * mc_draw + 0.5 * clf_draw
    final_win_b = 0.5 * mc_win_b + 0.5 * clf_win_b
    
    # Normalize to prevent precision issues
    total_prob = final_win_a + final_draw + final_win_b
    if total_prob > 0:
        final_win_a /= total_prob
        final_draw /= total_prob
        final_win_b /= total_prob
        
    # Calculate scale factors to match blended probabilities
    scale_a = final_win_a / mc_win_a if mc_win_a > 0 else 0.0
    scale_draw = final_draw / mc_draw if mc_draw > 0 else 0.0
    scale_b = final_win_b / mc_win_b if mc_win_b > 0 else 0.0
    
    # Calculate exact score probabilities
    scores = {}
    for ga, gb in zip(sim_goals_a, sim_goals_b):
        score = f"{ga}-{gb}"
        if ga > gb:
            w = scale_a
        elif ga == gb:
            w = scale_draw
        else:
            w = scale_b
        scores[score] = scores.get(score, 0.0) + w
        
    # Normalize exact score probabilities
    sum_scores = sum(scores.values())
    if sum_scores > 0:
        for k in scores:
            scores[k] /= sum_scores
            
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_scores = [{"score": k, "probability": float(v)} for k, v in sorted_scores[:5]]
    
    # Calculate calibrated expected goals based on exact score probabilities
    calibrated_exp_goals_a = 0.0
    calibrated_exp_goals_b = 0.0
    for score, prob in scores.items():
        ga, gb = map(int, score.split('-'))
        calibrated_exp_goals_a += ga * prob
        calibrated_exp_goals_b += gb * prob
        
    # 2. Knockout simulation (prorrogue & penalty shootout)
    qualify_a = final_win_a
    qualify_b = final_win_b
    
    prob_et_a = 0.0
    prob_et_b = 0.0
    prob_pen_a = 0.0
    prob_pen_b = 0.0
    
    # In a knockout, we use the draws to simulate extra time and penalties
    draw_90 = int(np.sum(sim_goals_a == sim_goals_b))
    if knockout and draw_90 > 0:
        elo_diff = exp['elo_a'] - exp['elo_b']
        p_conv_a = max(0.65, min(0.85, 0.75 + 0.05 * (elo_diff / 400.0)))
        p_conv_b = max(0.65, min(0.85, 0.75 - 0.05 * (elo_diff / 400.0)))
        
        # Vectorized Extra Time goals sampling
        sim_et_goals_a = rng.poisson(calibrated_exp_goals_a / 3.0, draw_90)
        sim_et_goals_b = rng.poisson(calibrated_exp_goals_b / 3.0, draw_90)
        
        et_wins_a_mask = sim_et_goals_a > sim_et_goals_b
        et_wins_b_mask = sim_et_goals_b > sim_et_goals_a
        et_draws_mask = sim_et_goals_a == sim_et_goals_b
        
        et_wins_a_count = int(np.sum(et_wins_a_mask))
        et_wins_b_count = int(np.sum(et_wins_b_mask))
        pen_draws_count = int(np.sum(et_draws_mask))
        
        pen_wins_a = 0
        pen_wins_b = 0
        if pen_draws_count > 0:
            for _ in range(pen_draws_count):
                winner = simulate_penalty_shootout(p_conv_a, p_conv_b, rng)
                if winner == 'A':
                    pen_wins_a += 1
                else:
                    pen_wins_b += 1
                    
        prob_et_a = final_draw * (et_wins_a_count / draw_90)
        prob_et_b = final_draw * (et_wins_b_count / draw_90)
        prob_pen_a = final_draw * (pen_wins_a / draw_90)
        prob_pen_b = final_draw * (pen_wins_b / draw_90)
        
        qualify_a += prob_et_a + prob_pen_a
        qualify_b += prob_et_b + prob_pen_b
        
    # 4. Corners stats
    total_corners = sim_corners_a + sim_corners_b
    avg_corners_a = float(np.mean(sim_corners_a))
    avg_corners_b = float(np.mean(sim_corners_b))
    avg_corners_total = float(np.mean(total_corners))
    
    prob_corners_over = {
        "over_7_5": float(np.sum(total_corners > 7.5) / n_sims),
        "over_8_5": float(np.sum(total_corners > 8.5) / n_sims),
        "over_9_5": float(np.sum(total_corners > 9.5) / n_sims),
        "over_10_5": float(np.sum(total_corners > 10.5) / n_sims)
    }
    
    # 5. Cards stats
    total_cards = sim_cards_a + sim_cards_b
    avg_cards_a = float(np.mean(sim_cards_a))
    avg_cards_b = float(np.mean(sim_cards_b))
    avg_cards_total = float(np.mean(total_cards))
    
    prob_cards_over = {
        "over_2_5": float(np.sum(total_cards > 2.5) / n_sims),
        "over_3_5": float(np.sum(total_cards > 3.5) / n_sims),
        "over_4_5": float(np.sum(total_cards > 4.5) / n_sims),
        "over_5_5": float(np.sum(total_cards > 5.5) / n_sims)
    }
    
    # 6. First goal probabilities
    sum_rates = calibrated_exp_goals_a + calibrated_exp_goals_b
    prob_no_goals = scores.get('0-0', 0.0)
    
    if sum_rates > 0:
        prob_first_goal_a = float((1.0 - prob_no_goals) * (calibrated_exp_goals_a / sum_rates))
        prob_first_goal_b = float((1.0 - prob_no_goals) * (calibrated_exp_goals_b / sum_rates))
    else:
        prob_first_goal_a = 0.0
        prob_first_goal_b = 0.0
        
    report = {
        "match_info": {
            "team_a": team_a,
            "team_b": team_b,
            "elo_a_raw": exp['elo_a_raw'],
            "elo_b_raw": exp['elo_b_raw'],
            "elo_a_adjusted": exp['elo_a'],
            "elo_b_adjusted": exp['elo_b'],
            "neutral": neutral,
            "missing_players_a": missing_players_a,
            "missing_players_b": missing_players_b,
            "knockout": knockout,
            "city": city,
            "country": country,
            "altitude_meters": exp['altitude_meters'],
            "referee": referee,
            "referee_stats": exp['referee_stats'],
            "squad_value_a": exp['squad_value_a'],
            "squad_value_b": exp['squad_value_b'],
            "absences_a": exp['absences_a'],
            "absences_b": exp['absences_b']
        },
        "predictions": {
            "outcomes": {
                "win_a": final_win_a,
                "win_b": final_win_b,
                "draw": final_draw
            },
            "goals": {
                "expected_a": calibrated_exp_goals_a,
                "expected_b": calibrated_exp_goals_b,
                "expected_total": calibrated_exp_goals_a + calibrated_exp_goals_b,
                "top_exact_scores": top_scores,
                "all_exact_scores": scores
            },
            "corners": {
                "expected_a": avg_corners_a,
                "expected_b": avg_corners_b,
                "expected_total": avg_corners_total,
                "possession_a": exp['possession_a'],
                "possession_b": exp['possession_b'],
                "over_under": prob_corners_over
            },
            "cards": {
                "expected_a": avg_cards_a,
                "expected_b": avg_cards_b,
                "expected_total": avg_cards_total,
                "over_under": prob_cards_over
            },
            "events": {
                "first_goal_team_a": prob_first_goal_a,
                "first_goal_team_b": prob_first_goal_b,
                "no_goals": prob_no_goals
            }
        }
    }
    
    if knockout:
        report["predictions"]["knockout_details"] = {
            "qualify_a": qualify_a,
            "qualify_b": qualify_b,
            "won_in_90m_a": final_win_a,
            "won_in_90m_b": final_win_b,
            "won_in_et_a": prob_et_a,
            "won_in_et_b": prob_et_b,
            "won_in_pen_a": prob_pen_a,
            "won_in_pen_b": prob_pen_b
        }
        
    return report

if __name__ == "__main__":
    try:
        res = run_monte_carlo_simulation("Argentina", "France", neutral=True, knockout=True)
        import pprint
        pprint.pprint(res)
    except Exception as e:
        print(f"Cannot run simulation test: {e}")
