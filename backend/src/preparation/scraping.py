import json
import os
from typing import Dict, List
import pandas as pd
import requests
from tqdm import tqdm
import time
import yaml
from pathlib import Path

# Load configuration from config.yaml
config_path = Path(__file__).parent.parent / 'config.yaml'
with open(config_path, 'r') as f:
    # Load only the scraping section
    config = yaml.safe_load(f)['scraping']

# Configuration constants
BASE_URL = config['base_url']
LEAGUES = config['leagues']
SEASONS = list(range(config['seasons']['start'], config['seasons']['end'] + 1))
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

def get_shots_from_match_pages(league_name: str, year: int) -> List[Dict]:
    """
    Retrieve individual shot data for a specific league and season via the
    Understat JSON API.

    Understat exposes two relevant endpoints:
      - GET /main/getLeagueData/{league}/{season}  -> match IDs in `dates`
      - GET /main/getMatchData/{match_id}          -> shot data in `shots`

    :param league_name: Understat league identifier (e.g. "EPL", "Bundesliga").
    :param year: Season start year (e.g. 2022 for the 2022/23 season).
    :return: List of shot dictionaries, each augmented with `league` and `season`.
    """
    print(f"Getting shot data for {league_name} {year}...")

    headers = {
        'User-Agent': USER_AGENT,
        'X-Requested-With': 'XMLHttpRequest',
    }

    # Fetch all match IDs for the league/season via the JSON API.
    league_data_url = f"{BASE_URL}/main/getLeagueData/{league_name}/{year}"
    # BASE_URL is https://understat.com, so the full path becomes
    # https://understat.com/main/getLeagueData/{league}/{season}
    try:
        res = requests.get(league_data_url, headers=headers)
        res.raise_for_status()
        league_data = res.json()
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch league data: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Could not parse league data response: {e}")
        return []

    dates = league_data.get('dates', [])
    match_ids = [entry['id'] for entry in dates if 'id' in entry]

    if not match_ids:
        print("Could not find match IDs")
        return []

    print(f"Found {len(match_ids)} matches")

    all_shots = []
    for match_id in tqdm(match_ids, desc="Scraping matches"):
        match_data_url = f"{BASE_URL}/main/getMatchData/{match_id}"
        try:
            res = requests.get(match_data_url, headers=headers)
            res.raise_for_status()
            match_data = res.json()
            time.sleep(0.5)  # Be respectful to the server
        except requests.exceptions.RequestException as e:
            print(f"Could not fetch match {match_id}: {e}")
            continue
        except json.JSONDecodeError as e:
            print(f"Could not parse shot data for match {match_id}: {e}")
            continue

        shots_by_team = match_data.get('shots', {})
        for shot_list in shots_by_team.values():
            for shot in shot_list:
                shot['league'] = league_name
                shot['season'] = year
                all_shots.append(shot)

    return all_shots

def main():
    """
    Scrape shot data from Understat and save it into separate CSV files by league and season.
    Skip if the file already exists.
    """
    # Set up the base output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.normpath(os.path.join(script_dir, "../../..")))
    base_output_dir = os.path.join(project_root, "backend", "data", "raw")
    
    total = len(LEAGUES) * len(SEASONS)
    with tqdm(total=total, desc="Scraping Understat Data") as pbar:
        for league in LEAGUES:
            league_dir = os.path.join(base_output_dir, league)
            os.makedirs(league_dir, exist_ok=True)
            for season in SEASONS:
                output_path = os.path.join(league_dir, f"shots_{season}.csv")
                if os.path.exists(output_path):
                    print(f"Data already exists for {league} {season}. Skipping...")
                else:
                    print(f"Scraping {league} {season}...")
                    shots = get_shots_from_match_pages(league, season)
                    if shots:
                        df = pd.DataFrame(shots)
                        # Convert numeric columns
                        for col in ['X', 'Y', 'xG']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Save to CSV
                        df.to_csv(output_path, index=False)
                        print(f"Saved {len(df)} shots to {output_path}")
                    else:
                        print(f"No data found for {league} {season}")
                pbar.update(1)

if __name__ == "__main__":
    main()