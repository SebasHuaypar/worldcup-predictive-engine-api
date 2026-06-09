import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from src.data.database import (
    init_db, save_matches_from_dataframe, save_squad_stats_from_dataframe, 
    save_squad_values, save_squad_absences, save_referee_stats, save_city_altitudes, 
    TEAM_NAME_MAPPING
)
from src.data.elo import update_elo_ratings

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_squad_name(name: str) -> str:
    # Clean up the country name since it sometimes has the flag code in front
    if not isinstance(name, str):
        return name
    name = re.sub(r'^[a-z]{2}\s+', '', name)
    name = name.strip()
    
    return TEAM_NAME_MAPPING.get(name, name)

def download_historical_matches():
    # Download the historical match data and save it to the database
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    print(f"Downloading historical match data from {url}...")
    try:
        df = pd.read_csv(url)
        print(f"Downloaded {len(df)} matches.")
        
        # Filter from 1970 onwards so ELO calculation doesn't take forever (good balance)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '1970-01-01']
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        save_matches_from_dataframe(df)
    except Exception as e:
        print(f"Error downloading historical matches: {e}")

def parse_fbref_table(soup: BeautifulSoup, table_id: str) -> pd.DataFrame:
    # Parse the specific FBref table we need using pandas
    table = soup.find("table", {"id": table_id})
    if not table:
        print(f"Table with ID '{table_id}' not found.")
        return pd.DataFrame()
    
    df = pd.read_html(StringIO(str(table)))[0]
    
    # Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if not col[1].startswith('Unnamed') else col[1] for col in df.columns]
    
    # Standardize the team column
    squad_col = None
    for col in df.columns:
        if 'squad' in col.lower() or 'team' in col.lower():
            squad_col = col
            break
            
    if squad_col:
        df = df.rename(columns={squad_col: 'team'})
        df['team'] = df['team'].apply(clean_squad_name)
    
    return df

def scrape_fbref_tournament(url: str, tournament_name: str, year: int) -> pd.DataFrame:
    # Extract team stats for a given tournament on FBref
    print(f"Scraping {tournament_name} ({year}) from {url}...")
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
            return pd.DataFrame()
            
        # FBref comments out almost all tables, so we force uncomment them
        uncommented_html = response.text.replace("<!--", "").replace("-->", "")
        soup = BeautifulSoup(uncommented_html, "html.parser")
        
        # Read the standard table
        df_std = parse_fbref_table(soup, "stats_squads_standard_for")
        if df_std.empty:
            return pd.DataFrame()
            
        # Keep only the key metrics we actually care about
        std_cols = {}
        for col in df_std.columns:
            col_lower = col.lower()
            if col_lower == 'team':
                std_cols[col] = 'team'
            elif col_lower == 'mp' or col_lower.endswith('_mp'):
                std_cols[col] = 'mp'
            elif 'poss' in col_lower:
                std_cols[col] = 'possession'
            elif 'crdy' in col_lower:
                std_cols[col] = 'yellow_cards'
            elif 'crdr' in col_lower:
                std_cols[col] = 'red_cards'
            elif 'xg' in col_lower and 'npxg' not in col_lower and 'xg+xag' not in col_lower:
                std_cols[col] = 'xg'
                
        df_std_filtered = df_std[list(std_cols.keys())].rename(columns=std_cols)
        # Group by team to blow away the totals row
        df_std_filtered = df_std_filtered[df_std_filtered['team'] != 'Squad']
        df_std_filtered = df_std_filtered.dropna(subset=['mp'])
        
        # Read the shooting table
        df_shoot = parse_fbref_table(soup, "stats_squads_shooting_for")
        shoot_cols = {'team': 'team'}
        if not df_shoot.empty:
            for col in df_shoot.columns:
                col_lower = col.lower()
                if 'sh' in col_lower and 'sh/90' not in col_lower and 'sot%' not in col_lower and 'g/sh' not in col_lower:
                    shoot_cols[col] = 'shots'
                elif 'sot' in col_lower and 'sot/90' not in col_lower and 'sot%' not in col_lower:
                    shoot_cols[col] = 'sot'
            df_shoot_filtered = df_shoot[list(shoot_cols.keys())].rename(columns=shoot_cols)
            df_shoot_filtered = df_shoot_filtered[df_shoot_filtered['team'] != 'Squad']
        else:
            df_shoot_filtered = pd.DataFrame(columns=['team', 'shots', 'sot'])
            
        # Read the passing table to get corners
        df_pass_types = parse_fbref_table(soup, "stats_squads_passing_types_for")
        pass_cols = {'team': 'team'}
        if not df_pass_types.empty:
            for col in df_pass_types.columns:
                col_lower = col.lower()
                if 'ck' in col_lower or 'corners' in col_lower:
                    pass_cols[col] = 'corners'
            df_pass_types_filtered = df_pass_types[list(pass_cols.keys())].rename(columns=pass_cols)
            df_pass_types_filtered = df_pass_types_filtered[df_pass_types_filtered['team'] != 'Squad']
        else:
            df_pass_types_filtered = pd.DataFrame(columns=['team', 'corners'])
            
        # Merge everything into a single DataFrame
        merged = df_std_filtered.merge(df_shoot_filtered, on='team', how='left')
        merged = merged.merge(df_pass_types_filtered, on='team', how='left')
        
        # Fill missing data so the model doesn't complain
        merged['possession'] = pd.to_numeric(merged['possession'], errors='coerce').fillna(50.0)
        merged['mp'] = pd.to_numeric(merged['mp'], errors='coerce').fillna(1).astype(int)
        merged['yellow_cards'] = pd.to_numeric(merged['yellow_cards'], errors='coerce').fillna(0)
        merged['red_cards'] = pd.to_numeric(merged['red_cards'], errors='coerce').fillna(0)
        merged['xg'] = pd.to_numeric(merged['xg'], errors='coerce').fillna(0.0)
        merged['shots'] = pd.to_numeric(merged['shots'], errors='coerce').fillna(0)
        merged['sot'] = pd.to_numeric(merged['sot'], errors='coerce').fillna(0)
        merged['corners'] = pd.to_numeric(merged['corners'], errors='coerce').fillna(0)
        
        # Calculate per-match averages
        merged['shots_per_match'] = merged['shots'] / merged['mp']
        merged['sot_per_match'] = merged['sot'] / merged['mp']
        merged['corners_per_match'] = merged['corners'] / merged['mp']
        merged['yellow_cards_per_match'] = merged['yellow_cards'] / merged['mp']
        merged['red_cards_per_match'] = merged['red_cards'] / merged['mp']
        merged['xg_per_match'] = merged['xg'] / merged['mp']
        
        merged['tournament'] = tournament_name
        merged['year'] = year
        
        final_df = merged[[
            'team', 'tournament', 'year', 'mp', 'possession',
            'shots_per_match', 'sot_per_match', 'corners_per_match',
            'yellow_cards_per_match', 'red_cards_per_match', 'xg_per_match'
        ]]
        
        return final_df
        
    except Exception as e:
        print(f"Error scraping {tournament_name} ({year}): {e}")
        return pd.DataFrame()

def scrape_fbref_tournaments():
    # Scrape the main international tournaments from FBref (with fallback if blocked)
    tournaments = [
        {
            "url": "https://fbref.com/en/comps/1/2022/2022-World-Cup-Stats",
            "name": "FIFA World Cup",
            "year": 2022
        },
        {
            "url": "https://fbref.com/en/comps/1/2018/stats/2018-World-Cup-Stats",
            "name": "FIFA World Cup",
            "year": 2018
        },
        {
            "url": "https://fbref.com/en/comps/676/stats/European-Championship-Stats",
            "name": "UEFA Euro",
            "year": 2024
        },
        {
            "url": "https://fbref.com/en/comps/676/2020/stats/2020-European-Championship-Stats",
            "name": "UEFA Euro",
            "year": 2020
        },
        {
            "url": "https://fbref.com/en/comps/685/stats/Copa-America-Stats",
            "name": "Copa América",
            "year": 2024
        },
        {
            "url": "https://fbref.com/en/comps/685/2021/stats/2021-Copa-America-Stats",
            "name": "Copa América",
            "year": 2021
        }
    ]
    
    all_squads_stats = []
    scraped_any = False
    
    for t in tournaments:
        # Pause a bit so FBref doesn't ban us for too many requests
        time.sleep(2.0)
        df = scrape_fbref_tournament(t['url'], t['name'], t['year'])
        if not df.empty:
            all_squads_stats.append(df)
            print(f"Scraped {len(df)} teams for {t['name']} ({t['year']}).")
            scraped_any = True
            
    if scraped_any:
        combined_df = pd.concat(all_squads_stats, ignore_index=True)
        save_squad_stats_from_dataframe(combined_df)
    else:
        import os
        print("\n[WARNING] All FBref scraping attempts failed (likely due to 403 Forbidden).")
        print("Loading local fallback squad statistics from data/raw/squad_stats_fallback.csv...")
        try:
            fallback_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "raw", "squad_stats_fallback.csv"
            )
            fallback_df = pd.read_csv(fallback_path)
            save_squad_stats_from_dataframe(fallback_df)
            print("Loaded local fallback squad statistics successfully.")
        except Exception as e:
            print(f"Error loading fallback squad stats: {e}")

def scrape_squad_values():
    # Extract market value from Transfermarkt (or use local fallback if blocked)
    print("Scraping squad values from Transfermarkt...")
    url = "https://www.transfermarkt.com/spieler-statistik/wertvollstenationalmannschaften/marktwertetop"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"class": "items"})
            if table:
                tbody = table.find("tbody")
                rows = tbody.find_all("tr", recursive=False) if tbody else []
                values_list = []
                for row in rows:
                    cols = row.find_all("td", recursive=False)
                    if len(cols) >= 4:
                        # Extract the team name and convert its value to euros
                        team_col = cols[1].find("a")
                        team_name = team_col.text.strip() if team_col else cols[1].text.strip()
                        
                        val_col = cols[3].find("a")
                        val_text = val_col.text.strip() if val_col else cols[3].text.strip()
                        
                        val_euros = 50000000.0
                        if "bn" in val_text:
                            val_euros = float(val_text.replace("€", "").replace("bn", "").strip()) * 1e9
                        elif "m" in val_text:
                            val_euros = float(val_text.replace("€", "").replace("m", "").strip()) * 1e6
                            
                        values_list.append({
                            "team": team_name,
                            "market_value_euros": val_euros,
                            "last_updated": time.strftime("%Y-%m-%d")
                        })
                if values_list:
                    save_squad_values(values_list)
                    print(f"Successfully scraped and saved {len(values_list)} squad values.")
                    return
        print(f"Failed to scrape Transfermarkt: status code {response.status_code}")
    except Exception as e:
        print(f"Error scraping Transfermarkt: {e}")
        
    # If it fails, load the backup data from the local file
    import os
    print("Loading local fallback squad values from data/raw/squad_values_fallback.csv...")
    try:
        fallback_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "raw", "squad_values_fallback.csv"
        )
        fallback_df = pd.read_csv(fallback_path)
        values_list = fallback_df.to_dict(orient="records")
        save_squad_values(values_list)
        print("Loaded local fallback squad values successfully.")
    except Exception as fe:
        print(f"Error loading fallback squad values: {fe}")

def scrape_squad_absences():
    # Extract player injuries from Transfermarkt
    print("Scraping player absences from Transfermarkt...")
    url = "https://www.transfermarkt.com/spieler-statistik/verletztespieler/statistik/yt0"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"class": "items"})
            if table:
                tbody = table.find("tbody")
                rows = tbody.find_all("tr", recursive=False) if tbody else []
                absences_list = []
                for row in rows:
                    cols = row.find_all("td", recursive=False)
                    if len(cols) >= 6:
                        player_name = cols[0].text.strip()
                        team_name = cols[2].text.strip()
                        reason = cols[3].text.strip()
                        
                        severity = "star_player"
                        
                        absences_list.append({
                            "team": team_name,
                            "player_name": player_name,
                            "severity": severity,
                            "reason": reason,
                            "date_recorded": time.strftime("%Y-%m-%d")
                        })
                if absences_list:
                    save_squad_absences(absences_list)
                    print(f"Successfully scraped and saved {len(absences_list)} squad absences.")
                    return
        print(f"Failed to scrape Transfermarkt absences: status code {response.status_code}")
    except Exception as e:
        print(f"Error scraping Transfermarkt absences: {e}")
        
    # If it fails, load the backup data from the local file
    import os
    print("Loading local fallback squad absences from data/raw/squad_absences_fallback.csv...")
    try:
        fallback_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "raw", "squad_absences_fallback.csv"
        )
        fallback_df = pd.read_csv(fallback_path)
        absences_list = fallback_df.to_dict(orient="records")
        save_squad_absences(absences_list)
        print("Loaded local fallback squad absences successfully.")
    except Exception as fe:
        print(f"Error loading fallback squad absences: {fe}")

def load_static_reference_data():
    # Load fixed data like referee stats and city altitudes
    print("Loading static reference data (referee stats and city altitudes)...")
    import os
    
    try:
        ref_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "raw", "referee_stats_fallback.csv"
        )
        if os.path.exists(ref_path):
            df_ref = pd.read_csv(ref_path)
            ref_list = df_ref.to_dict(orient="records")
            save_referee_stats(ref_list)
            print("Loaded referee stats successfully.")
        else:
            print("[WARNING] referee_stats_fallback.csv not found.")
    except Exception as e:
        print(f"Error loading referee stats: {e}")
        
    try:
        alt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "raw", "city_altitudes_fallback.csv"
        )
        if os.path.exists(alt_path):
            df_alt = pd.read_csv(alt_path)
            alt_list = df_alt.to_dict(orient="records")
            save_city_altitudes(alt_list)
            print("Loaded city altitudes successfully.")
        else:
            print("[WARNING] city_altitudes_fallback.csv not found.")
    except Exception as e:
        print(f"Error loading city altitudes: {e}")

def cleanup_usa_stats():
    # Delete duplicate 'USA' records to keep only 'United States'
    from src.data.database import get_connection
    from contextlib import closing
    print("Cleaning up stale 'USA' records from squad_stats...")
    try:
        with closing(get_connection()) as conn:
            with conn:
                conn.execute("DELETE FROM squad_stats WHERE team = 'USA'")
        print("Cleaned up 'USA' stats successfully.")
    except Exception as e:
        print(f"Error cleaning up 'USA' stats: {e}")

def run_scraper_pipeline():
    # Run the entire data fetching process
    print("Starting data scraper pipeline...")
    init_db()
    
    load_static_reference_data()
    download_historical_matches()
    scrape_fbref_tournaments()
    scrape_squad_values()
    scrape_squad_absences()
    update_elo_ratings()
    cleanup_usa_stats()
    
    print("Data scraper pipeline finished successfully.")

if __name__ == "__main__":
    run_scraper_pipeline()
