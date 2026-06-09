import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
from contextlib import closing

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "database.sqlite")

import unicodedata

# Central dictionary to ensure team names match across all sources
TEAM_NAME_MAPPING = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Macedonia": "North Macedonia",
    "Korea, South": "South Korea",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "China PR": "China",
    "Ireland": "Republic of Ireland",
    "Cape Verde Islands": "Cape Verde",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina"
}

def normalize_text(text: str) -> str:
    # Remove accents and tildes to avoid text comparison issues
    if not isinstance(text, str):
        return text
    normalized = unicodedata.normalize('NFD', text)
    return "".join(c for c in normalized if unicodedata.category(c) != 'Mn').strip()


def get_connection():
    # Connect to the DB (using WAL mode so it doesn't lock up)
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Enable WAL mode so we can read and write at the same time without issues
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Create all DB tables if they don't exist yet
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            
            # Main match table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    tournament TEXT,
                    city TEXT,
                    country TEXT,
                    neutral INTEGER,
                    UNIQUE(date, home_team, away_team)
                )
            """)
            
            # Detailed stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS match_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    home_corners INTEGER,
                    away_corners INTEGER,
                    home_yellow_cards INTEGER,
                    away_yellow_cards INTEGER,
                    home_red_cards INTEGER,
                    away_red_cards INTEGER,
                    home_shots INTEGER,
                    away_shots INTEGER,
                    home_shots_on_target INTEGER,
                    away_shots_on_target INTEGER,
                    home_possession REAL,
                    away_possession REAL,
                    home_xg REAL,
                    away_xg REAL,
                    UNIQUE(date, home_team, away_team)
                )
            """)
            
            # Team averages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS squad_stats (
                    team TEXT,
                    tournament TEXT,
                    year INTEGER,
                    mp INTEGER,
                    possession REAL,
                    shots_per_match REAL,
                    sot_per_match REAL,
                    corners_per_match REAL,
                    yellow_cards_per_match REAL,
                    red_cards_per_match REAL,
                    xg_per_match REAL,
                    PRIMARY KEY (team, tournament, year)
                )
            """)
            
            # ELO ratings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS team_elo (
                    team TEXT,
                    date TEXT,
                    elo REAL,
                    PRIMARY KEY (team, date)
                )
            """)
            
            # Market values table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS squad_values (
                    team TEXT PRIMARY KEY,
                    market_value_euros REAL,
                    last_updated TEXT
                )
            """)
            
            # Referees table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS referee_stats (
                    referee TEXT PRIMARY KEY,
                    matches_played INTEGER,
                    yellow_cards_per_match REAL,
                    red_cards_per_match REAL,
                    fouls_per_match REAL
                )
            """)
            
            # City altitudes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS city_altitudes (
                    city TEXT,
                    country TEXT,
                    altitude_meters REAL,
                    PRIMARY KEY (city, country)
                )
            """)
            
            # Injuries and absences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS squad_absences (
                    team TEXT,
                    player_name TEXT,
                    severity TEXT,
                    reason TEXT,
                    date_recorded TEXT,
                    PRIMARY KEY (team, player_name)
                )
            """)
            
    print("Database initialized successfully.")

def save_matches_from_dataframe(df: pd.DataFrame):
    # Batch save all matches so it's super fast
    records = []
    for _, row in df.iterrows():
        records.append((
            row['date'], row['home_team'], row['away_team'], 
            int(row['home_score']) if not pd.isna(row['home_score']) else None,
            int(row['away_score']) if not pd.isna(row['away_score']) else None,
            row['tournament'], row['city'], row['country'],
            1 if row['neutral'] else 0
        ))
        
    inserted = 0
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            try:
                cursor.executemany("""
                    INSERT OR IGNORE INTO matches 
                    (date, home_team, away_team, home_score, away_score, tournament, city, country, neutral)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                inserted = cursor.rowcount
            except Exception as e:
                print(f"Error during batch match insert: {e}")
                
    print(f"Inserted {inserted} new matches into database.")

def save_match_stats_from_dataframe(df: pd.DataFrame):
    # Save the detailed match stats
    records = []
    for _, row in df.iterrows():
        records.append((
            row['date'], row['home_team'], row['away_team'],
            int(row['home_corners']) if not pd.isna(row['home_corners']) else None,
            int(row['away_corners']) if not pd.isna(row['away_corners']) else None,
            int(row['home_yellow_cards']) if not pd.isna(row['home_yellow_cards']) else None,
            int(row['away_yellow_cards']) if not pd.isna(row['away_yellow_cards']) else None,
            int(row['home_red_cards']) if not pd.isna(row['home_red_cards']) else None,
            int(row['away_red_cards']) if not pd.isna(row['away_red_cards']) else None,
            int(row['home_shots']) if not pd.isna(row['home_shots']) else None,
            int(row['away_shots']) if not pd.isna(row['away_shots']) else None,
            int(row['home_shots_on_target']) if not pd.isna(row['home_shots_on_target']) else None,
            int(row['away_shots_on_target']) if not pd.isna(row['away_shots_on_target']) else None,
            float(row['home_possession']) if not pd.isna(row['home_possession']) else None,
            float(row['away_possession']) if not pd.isna(row['away_possession']) else None,
            float(row['home_xg']) if not pd.isna(row['home_xg']) else None,
            float(row['away_xg']) if not pd.isna(row['away_xg']) else None
        ))
        
    inserted = 0
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            try:
                cursor.executemany("""
                    INSERT OR REPLACE INTO match_stats 
                    (date, home_team, away_team, home_corners, away_corners, 
                     home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards,
                     home_shots, away_shots, home_shots_on_target, away_shots_on_target,
                     home_possession, away_possession, home_xg, away_xg)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                inserted = cursor.rowcount
            except Exception as e:
                print(f"Error during batch match stats insert: {e}")
                
    print(f"Inserted/updated {inserted} match stats records.")

def save_elo_ratings(elo_list: List[Dict[str, Any]]):
    # Update ELO ratings in the DB
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM team_elo")
            cursor.executemany("""
                INSERT OR REPLACE INTO team_elo (team, date, elo)
                VALUES (:team, :date, :elo)
            """, elo_list)

def get_latest_elo(team: str, date: str, conn: Optional[sqlite3.Connection] = None) -> float:
    # Find a team's ELO on a specific date (defaulting to 1500 if missing)
    def _execute(c):
        cursor = c.cursor()
        cursor.execute("""
            SELECT elo FROM team_elo 
            WHERE team = ? AND date <= ? 
            ORDER BY date DESC LIMIT 1
        """, (team, date))
        row = cursor.fetchone()
        return row['elo'] if row else 1500.0

    if conn is not None:
        return _execute(conn)
    with closing(get_connection()) as conn_new:
        return _execute(conn_new)

def load_all_matches() -> pd.DataFrame:
    # Load all matches into a DataFrame
    with closing(get_connection()) as conn:
        df = pd.read_sql_query("SELECT * FROM matches ORDER BY date ASC", conn)
        return df

def load_all_match_stats() -> pd.DataFrame:
    # Load all match stats into a DataFrame
    with closing(get_connection()) as conn:
        df = pd.read_sql_query("SELECT * FROM match_stats ORDER BY date ASC", conn)
        return df

def save_squad_stats_from_dataframe(df: pd.DataFrame):
    # Save the team averages to the DB
    records = []
    for _, row in df.iterrows():
        team_name = row['team']
        team_name = TEAM_NAME_MAPPING.get(team_name, team_name)
            
        records.append((
            team_name, row['tournament'], int(row['year']), int(row['mp']),
            float(row['possession']), float(row['shots_per_match']),
            float(row['sot_per_match']), float(row['corners_per_match']),
            float(row['yellow_cards_per_match']), float(row['red_cards_per_match']),
            float(row['xg_per_match']) if not pd.isna(row['xg_per_match']) else None
        ))
        
    inserted = 0
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            try:
                cursor.executemany("""
                    INSERT OR REPLACE INTO squad_stats 
                    (team, tournament, year, mp, possession, shots_per_match, 
                     sot_per_match, corners_per_match, yellow_cards_per_match, 
                     red_cards_per_match, xg_per_match)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)
                inserted = cursor.rowcount
            except Exception as e:
                print(f"Error during batch squad stats insert: {e}")
                
    print(f"Inserted/updated {inserted} squad stats records.")

def load_all_squad_stats() -> pd.DataFrame:
    # Load all team stats into a DataFrame
    with closing(get_connection()) as conn:
        df = pd.read_sql_query("SELECT * FROM squad_stats", conn)
        return df

def save_squad_values(values_list: List[Dict[str, Any]]):
    # Save the squad market values
    records = []
    for val in values_list:
        team_name = val['team']
        team_name = TEAM_NAME_MAPPING.get(team_name, team_name)
        records.append((team_name, float(val['market_value_euros']), val['last_updated']))
        
    inserted = 0
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            try:
                cursor.executemany("""
                    INSERT OR REPLACE INTO squad_values (team, market_value_euros, last_updated)
                    VALUES (?, ?, ?)
                """, records)
                inserted = cursor.rowcount
            except Exception as e:
                print(f"Error saving squad values: {e}")
                
    print(f"Inserted/updated {inserted} squad value records.")

def get_squad_value(team: str, conn: Optional[sqlite3.Connection] = None) -> float:
    # Get a team's market value (defaulting to 50 million if we can't find it)
    def _execute(c):
        cursor = c.cursor()
        cursor.execute("SELECT market_value_euros FROM squad_values WHERE team = ? COLLATE NOCASE", (team,))
        row = cursor.fetchone()
        return float(row['market_value_euros']) if row else 50000000.0

    if conn is not None:
        return _execute(conn)
    with closing(get_connection()) as conn_new:
        return _execute(conn_new)

def save_referee_stats(ref_list: List[Dict[str, Any]]):
    # Save the referee stats
    records = [
        (
            ref['referee'], int(ref['matches_played']),
            float(ref['yellow_cards_per_match']), float(ref['red_cards_per_match']),
            float(ref['fouls_per_match'])
        )
        for ref in ref_list
    ]
    
    inserted = 0
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            try:
                cursor.execute("DELETE FROM referee_stats")
                cursor.executemany("""
                    INSERT OR REPLACE INTO referee_stats 
                    (referee, matches_played, yellow_cards_per_match, red_cards_per_match, fouls_per_match)
                    VALUES (?, ?, ?, ?, ?)
                """, records)
                inserted = cursor.rowcount
            except Exception as e:
                print(f"Error saving referee stats: {e}")
                
    print(f"Inserted/updated {inserted} referee records.")

def get_referee_stats(referee: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    # Look up a referee's stats by name
    ref_norm = normalize_text(referee).lower()
    def _execute(c):
        cursor = c.cursor()
        cursor.execute("SELECT * FROM referee_stats")
        rows = cursor.fetchall()
        for r in rows:
            if normalize_text(r['referee']).lower() == ref_norm:
                return {
                    'referee': r['referee'],
                    'matches_played': r['matches_played'],
                    'yellow_cards_per_match': r['yellow_cards_per_match'],
                    'red_cards_per_match': r['red_cards_per_match'],
                    'fouls_per_match': r['fouls_per_match']
                }
        return None

    if conn is not None:
        return _execute(conn)
    with closing(get_connection()) as conn_new:
        return _execute(conn_new)

def load_all_referees() -> List[Dict[str, Any]]:
    # Load all referees into a list
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM referee_stats ORDER BY referee ASC")
        rows = cursor.fetchall()
        return [
            {
                'referee': r['referee'],
                'matches_played': r['matches_played'],
                'yellow_cards_per_match': r['yellow_cards_per_match'],
                'red_cards_per_match': r['red_cards_per_match'],
                'fouls_per_match': r['fouls_per_match']
            }
            for r in rows
        ]

def save_city_altitudes(altitudes: List[Dict[str, Any]]):
    # Save the city altitudes
    records = [
        (alt['city'], alt['country'], float(alt['altitude_meters']))
        for alt in altitudes
    ]
    
    inserted = 0
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            try:
                cursor.executemany("""
                    INSERT OR REPLACE INTO city_altitudes (city, country, altitude_meters)
                    VALUES (?, ?, ?)
                """, records)
                inserted = cursor.rowcount
            except Exception as e:
                print(f"Error saving city altitudes: {e}")
                
    print(f"Inserted/updated {inserted} city altitude records.")

def get_city_altitude(city: str, country: str, conn: Optional[sqlite3.Connection] = None) -> float:
    # Look up a city's altitude (returns 0 if we can't find it)
    city_norm = normalize_text(city).lower()
    country_norm = normalize_text(country).lower() if country else ""
    
    def _execute(c):
        cursor = c.cursor()
        cursor.execute("SELECT city, country, altitude_meters FROM city_altitudes")
        rows = cursor.fetchall()
        
        # First, try matching both city and country
        for r in rows:
            r_city_norm = normalize_text(r['city']).lower()
            r_country_norm = normalize_text(r['country']).lower()
            if r_city_norm == city_norm and r_country_norm == country_norm:
                return float(r['altitude_meters'])
                
        # If not, just settling for a city match is fine
        for r in rows:
            r_city_norm = normalize_text(r['city']).lower()
            if r_city_norm == city_norm:
                return float(r['altitude_meters'])
                
        return 0.0

    if conn is not None:
        return _execute(conn)
    with closing(get_connection()) as conn_new:
        return _execute(conn_new)

def save_squad_absences(absences: List[Dict[str, Any]]):
    # Save the list of injured or suspended players
    records = []
    for ab in absences:
        team_name = ab['team']
        team_name = TEAM_NAME_MAPPING.get(team_name, team_name)
        records.append((
            team_name, ab['player_name'], ab['severity'], ab['reason'], ab['date_recorded']
        ))
        
    inserted = 0
    with closing(get_connection()) as conn:
        with conn:
            cursor = conn.cursor()
            try:
                # Clear out old injuries for these teams to avoid ghost data
                teams_to_clear = list(set([r[0] for r in records]))
                for team in teams_to_clear:
                    cursor.execute("DELETE FROM squad_absences WHERE team = ?", (team,))
                    
                cursor.executemany("""
                    INSERT OR REPLACE INTO squad_absences (team, player_name, severity, reason, date_recorded)
                    VALUES (?, ?, ?, ?, ?)
                """, records)
                inserted = cursor.rowcount
            except Exception as e:
                print(f"Error saving squad absences: {e}")
                
    print(f"Inserted/updated {inserted} squad absence records.")

def get_squad_absences(team: str, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    # Look up which players are missing for a team today
    def _execute(c):
        cursor = c.cursor()
        cursor.execute("""
            SELECT * FROM squad_absences WHERE team = ? COLLATE NOCASE
        """, (team,))
        rows = cursor.fetchall()
        return [
            {
                'team': r['team'],
                'player_name': r['player_name'],
                'severity': r['severity'],
                'reason': r['reason'],
                'date_recorded': r['date_recorded']
            }
            for r in rows
        ]

    if conn is not None:
        return _execute(conn)
    with closing(get_connection()) as conn_new:
        return _execute(conn_new)

if __name__ == "__main__":
    init_db()
