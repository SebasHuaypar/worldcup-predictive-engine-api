import sys
import os
import pprint

# Add the root directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simulator import run_monte_carlo_simulation
from src.data.database import get_connection

def test_prediction_engine():
    """Validates the predictive engine calculations and database integrity."""
    print("Starting predictive engine tests...")
    
    # 1. Database check
    print("\n[Test 1] Checking database connection and data counts...")
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as cnt FROM matches")
        matches_count = cursor.fetchone()['cnt']
        print(f" - Matches in database: {matches_count}")
        
        cursor.execute("SELECT COUNT(*) as cnt FROM squad_stats")
        stats_count = cursor.fetchone()['cnt']
        print(f" - Tournament squad stats in database: {stats_count}")
        
        cursor.execute("SELECT COUNT(*) as cnt FROM team_elo")
        elo_count = cursor.fetchone()['cnt']
        print(f" - ELO records in database: {elo_count}")
        
        conn.close()
        
        assert matches_count > 0, "No matches loaded in database."
        assert stats_count > 0, "No squad stats loaded in database."
        assert elo_count > 0, "No ELO records loaded in database."
        print(" -> Test 1 PASSED!")
    except Exception as e:
        print(f" -> Test 1 FAILED: {e}")
        return False
        
    # 2. Simulation check
    print("\n[Test 2] Running simulation between Argentina and France...")
    try:
        res = run_monte_carlo_simulation("Argentina", "France", neutral=True, n_sims=10000)
        
        print("\nSimulation Results Preview:")
        pprint.pprint(res['predictions']['outcomes'])
        print(f"Expected Goals: Argentina {res['predictions']['goals']['expected_a']:.2f} - {res['predictions']['goals']['expected_b']:.2f} France")
        print(f"Expected Corners: {res['predictions']['corners']['expected_total']:.2f}")
        print(f"Expected Cards: {res['predictions']['cards']['expected_total']:.2f}")
        
        # Validation checks
        outcomes = res['predictions']['outcomes']
        sum_probs = outcomes['win_a'] + outcomes['win_b'] + outcomes['draw']
        
        print(f"\nSum of outcomes: {sum_probs:.4f}")
        assert abs(sum_probs - 1.0) < 0.01, "Simulation outcome probabilities do not sum to 100%!"
        
        goals = res['predictions']['goals']
        assert goals['expected_a'] >= 0, "Expected goals cannot be negative!"
        assert goals['expected_b'] >= 0, "Expected goals cannot be negative!"
        
        corners = res['predictions']['corners']
        assert corners['expected_total'] >= 0, "Expected corners cannot be negative!"
        
        cards = res['predictions']['cards']
        assert cards['expected_total'] >= 0, "Expected cards cannot be negative!"
        
        print("\n -> Test 2 PASSED!")
        
        # 3. Injuries / Missing players check
        print("\n[Test 3] Running simulation with 3 missing players for Argentina...")
        res_missing = run_monte_carlo_simulation("Argentina", "France", neutral=True, n_sims=10000, missing_players_a=3)
        elo_a_raw = res_missing['match_info']['elo_a_raw']
        elo_a_adj = res_missing['match_info']['elo_a_adjusted']
        absences_count = len([ab for ab in res_missing['match_info']['absences_a'] if ab.get('severity') == 'star_player'])
        total_missing = 3 + absences_count
        print(f" - Raw ELO: {elo_a_raw:.2f}, Adjusted ELO ({total_missing} missing): {elo_a_adj:.2f}")
        assert abs(elo_a_adj - elo_a_raw * (1.0 - 0.025 * total_missing)) < 0.01, "ELO adjustment for missing players is mathematically incorrect!"
        
        # Check that Argentina's win probability dropped compared to Test 2
        p_win_a_raw = res['predictions']['outcomes']['win_a']
        p_win_a_missing = res_missing['predictions']['outcomes']['win_a']
        print(f" - Argentina Win Prob (Normal): {p_win_a_raw:.4f}, (with 3 missing): {p_win_a_missing:.4f}")
        assert p_win_a_missing < p_win_a_raw, "Argentina's win probability did not drop when key players are missing!"
        print(" -> Test 3 PASSED!")
        
        # 4. Knockout Mode check
        print("\n[Test 4] Running knockout stage simulation (Argentina vs France)...")
        res_ko = run_monte_carlo_simulation("Argentina", "France", neutral=True, n_sims=10000, knockout=True)
        assert 'knockout_details' in res_ko['predictions'], "knockout_details not present in response when knockout=True!"
        
        ko = res_ko['predictions']['knockout_details']
        sum_qualify = ko['qualify_a'] + ko['qualify_b']
        print(f" - Qualify probs: Argentina {ko['qualify_a']:.4f} vs {ko['qualify_b']:.4f} France")
        print(f" - Sum of qualify probs: {sum_qualify:.4f}")
        assert abs(sum_qualify - 1.0) < 0.01, "Qualify probabilities do not sum to 100%!"
        
        print(" -> Test 4 PASSED!")
        
        # 5. Referee Adjustment check
        print("\n[Test 5] Running simulation with Jesús Valenzuela (high card referee)...")
        res_normal_ref = run_monte_carlo_simulation("Argentina", "France", neutral=True, n_sims=10000, referee=None)
        res_high_ref_accent = run_monte_carlo_simulation("Argentina", "France", neutral=True, n_sims=10000, referee="Jesús Valenzuela")
        res_high_ref_no_accent = run_monte_carlo_simulation("Argentina", "France", neutral=True, n_sims=10000, referee="Jesus Valenzuela")
        
        cards_normal = res_normal_ref['predictions']['cards']['expected_total']
        cards_high_accent = res_high_ref_accent['predictions']['cards']['expected_total']
        cards_high_no_accent = res_high_ref_no_accent['predictions']['cards']['expected_total']
        
        print(f" - Expected Cards (Normal Referee): {cards_normal:.2f}, (Jesús Valenzuela): {cards_high_accent:.2f}, (Jesus Valenzuela): {cards_high_no_accent:.2f}")
        assert cards_high_accent > cards_normal * 1.1, "Referee cards scaling is not taking effect!"
        assert abs(cards_high_accent - cards_high_no_accent) < 0.01, "Referee diacritic normalization is not taking effect!"
        print(" -> Test 5 PASSED!")
        
        # 6. Altitude Adjustment check
        print("\n[Test 6] Running simulation at high altitude (La Paz & Bogotá) for non-acclimated visitors...")
        res_sea_level = run_monte_carlo_simulation("Bolivia", "France", neutral=False, n_sims=10000, city="Paris", country="France")
        res_high_alt_lapaz = run_monte_carlo_simulation("Bolivia", "France", neutral=False, n_sims=10000, city="La Paz", country="Bolivia")
        res_high_alt_bogota_accent = run_monte_carlo_simulation("Colombia", "France", neutral=False, n_sims=10000, city="Bogotá", country="Colombia")
        res_high_alt_bogota_no_accent = run_monte_carlo_simulation("Colombia", "France", neutral=False, n_sims=10000, city="Bogota", country="Colombia")
        
        goals_b_sea = res_sea_level['predictions']['goals']['expected_b']
        goals_b_alt_lapaz = res_high_alt_lapaz['predictions']['goals']['expected_b']
        
        alt_bogota_accent = res_high_alt_bogota_accent['match_info']['altitude_meters']
        alt_bogota_no_accent = res_high_alt_bogota_no_accent['match_info']['altitude_meters']
        
        print(f" - France Goals (Paris): {goals_b_sea:.2f}, France Goals (La Paz): {goals_b_alt_lapaz:.2f}")
        print(f" - Altitude Bogotá (accent): {alt_bogota_accent}m, (no accent): {alt_bogota_no_accent}m")
        
        assert goals_b_alt_lapaz < goals_b_sea, "Altitude goal fatigue for non-acclimated visiting team is not taking effect!"
        assert alt_bogota_accent == 2625.0, "Bogotá (accent) altitude check failed!"
        assert alt_bogota_no_accent == 2625.0, "Bogota (no accent) altitude check failed!"
        print(" -> Test 6 PASSED!")
        
        print(" -> All Tests PASSED!")
        return True
    except Exception as e:
        print(f" -> Test FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_prediction_engine()
    sys.exit(0 if success else 1)
