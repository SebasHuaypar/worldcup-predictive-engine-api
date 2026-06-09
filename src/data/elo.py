import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.data.database import load_all_matches, save_elo_ratings

def get_k_factor(tournament: str) -> float:
    """Returns the K-factor (match weight) based on the importance of the tournament."""
    t_lower = tournament.lower()
    if "world cup" in t_lower:
        if "qualification" in t_lower:
            return 50.0
        return 60.0
    elif any(comp in t_lower for comp in ["uefa euro", "copa américa", "african cup of nations", "asian cup", "gold cup", "nations cup"]):
        if "qualification" in t_lower:
            return 40.0
        return 50.0
    elif "nations league" in t_lower:
        return 35.0
    elif "friendly" in t_lower:
        return 20.0
    return 30.0

def get_goal_diff_multiplier(goal_diff: int) -> float:
    """Returns the goal difference multiplier to adjust ELO changes for blowouts."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    elif gd == 3:
        return 1.75
    else:
        return 1.75 + (gd - 3) / 8.0

def calculate_expected_outcome(elo_a: float, elo_b: float, neutral: bool) -> Tuple[float, float]:
    """Calculates the expected score (win probability) for both teams.
    Includes home advantage of 100 ELO points if the match is not neutral.
    """
    home_adv = 100.0 if not neutral else 0.0
    
    # Expected outcome for home team
    dr_home = elo_a - elo_b + home_adv
    we_home = 1.0 / (10.0 ** (-dr_home / 400.0) + 1.0)
    
    # Expected outcome for away team
    dr_away = elo_b - elo_a - home_adv
    we_away = 1.0 / (10.0 ** (-dr_away / 400.0) + 1.0)
    
    return we_home, we_away

def update_elo_ratings() -> int:
    """Reads all matches, computes ELO ratings chronologically, and saves history to database."""
    matches_df = load_all_matches()
    # Filter out matches without scores (fixtures or cancelled matches)
    matches_df = matches_df.dropna(subset=['home_score', 'away_score'])
    if matches_df.empty:
        print("No matches found in database to calculate ELO.")
        return 0
        
    print(f"Calculating ELO ratings for {len(matches_df)} matches...")
    
    # Dictionary to keep track of running ELO for each team
    running_elo: Dict[str, float] = {}
    elo_history: List[Dict[str, any]] = []
    
    for _, match in matches_df.iterrows():
        date = match['date']
        home = match['home_team']
        away = match['away_team']
        home_score = match['home_score']
        away_score = match['away_score']
        tournament = match['tournament']
        neutral = bool(match['neutral'])
        
        # Initialize team ELO if they don't exist
        if home not in running_elo:
            running_elo[home] = 1500.0
        if away not in running_elo:
            running_elo[away] = 1500.0
            
        elo_home_before = running_elo[home]
        elo_away_before = running_elo[away]
        
        # Determine actual outcome: 1 = win, 0.5 = draw, 0 = loss
        if home_score > away_score:
            w_home, w_away = 1.0, 0.0
        elif home_score < away_score:
            w_home, w_away = 0.0, 1.0
        else:
            w_home, w_away = 0.5, 0.5
            
        # K-factor & Goal difference multiplier
        k = get_k_factor(tournament)
        gd_mult = get_goal_diff_multiplier(int(home_score - away_score))
        
        # Expected scores
        we_home, we_away = calculate_expected_outcome(elo_home_before, elo_away_before, neutral)
        
        # Calculate new ELOs
        elo_home_new = elo_home_before + k * gd_mult * (w_home - we_home)
        elo_away_new = elo_away_before + k * gd_mult * (w_away - we_away)
        
        # Update running ratings
        running_elo[home] = elo_home_new
        running_elo[away] = elo_away_new
        
        # Append to history (save ELO AFTER the match to prevent desynchronization)
        elo_history.append({'team': home, 'date': date, 'elo': elo_home_new})
        elo_history.append({'team': away, 'date': date, 'elo': elo_away_new})
        
    # Also save the final ELO for all teams at the "end of time" (or current date)
    # to make query for future dates work.
    for team, elo in running_elo.items():
        elo_history.append({'team': team, 'date': '9999-12-31', 'elo': elo})
        
    save_elo_ratings(elo_history)
    print(f"Calculated and saved {len(elo_history)} ELO records.")
    return len(elo_history)

if __name__ == "__main__":
    # Test script directly
    from src.data.database import init_db
    init_db()
    update_elo_ratings()
