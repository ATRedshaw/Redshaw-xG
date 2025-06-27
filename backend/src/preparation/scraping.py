import json
import os
import re
from typing import Dict, List
import pandas as pd
import requests
from bs4 import BeautifulSoup
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
    Get individual shot data from match pages for a specific league and season.
    """
    print(f"Getting shot data from individual match pages for {league_name} {year}...")
    
    # Fetch the league page to find match IDs
    league_url = f"{BASE_URL}/{league_name}/{year}"
    headers = {
        'User-Agent': USER_AGENT
    }
    
    try:
        res = requests.get(league_url, headers=headers)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch league page: {e}")
        return []
    
    soup = BeautifulSoup(res.content, "lxml")
    scripts = soup.find_all("script")
    
    # Extract match IDs from datesData
    match_ids = []
    for script in scripts:
        if script.string and "datesData" in script.string:
            match = re.search(r"datesData\s*=\s*JSON\.parse\('(.+?)'\)", script.string)
            if match:
                try:
                    json_data = match.group(1)
                    decoded_data = bytes(json_data, "utf-8").decode("unicode_escape")
                    matches_data = json.loads(decoded_data)
                    for match_info in matches_data:
                        if 'id' in match_info:
                            match_ids.append(match_info['id'])
                    print(f"Found {len(match_ids)} matches")
                    break
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Could not parse matches data: {e}")
    
    if not match_ids:
        print("Could not find match IDs")
        return []
    
    all_shots = []
    for match_id in tqdm(match_ids, desc="Scraping matches"):
        match_url = f"https://understat.com/match/{match_id}"
        
        try:
            res = requests.get(match_url, headers=headers)
            res.raise_for_status()
            time.sleep(0.5)  # Be respectful to the server
        except requests.exceptions.RequestException as e:
            print(f"Could not fetch match {match_id}: {e}")
            continue
        
        match_soup = BeautifulSoup(res.content, "lxml")
        match_scripts = match_soup.find_all("script")
        
        # Extract shots data from match page
        for script in match_scripts:
            if script.string and "shotsData" in script.string:
                match = re.search(r"shotsData\s*=\s*JSON\.parse\('(.+?)'\)", script.string)
                if match:
                    try:
                        json_data = match.group(1)
                        decoded_data = bytes(json_data, "utf-8").decode("unicode_escape")
                        shots_data = json.loads(decoded_data)
                        for shot_list in shots_data.values():
                            for shot in shot_list:
                                shot['match_id'] = match_id
                                shot['league'] = league_name
                                shot['season'] = year
                                all_shots.append(shot)
                        break
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Could not parse shots data for match {match_id}: {e}")
    
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