import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import yaml
from tqdm import tqdm

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_raw_data_files(data_dir: Path, leagues: List[str], years: List[int]) -> List[Path]:
    """Get list of all raw data files to process"""
    files = []
    for league in leagues:
        league_dir = data_dir / 'raw' / league
        if not league_dir.exists():
            print(f"Warning: League directory not found: {league_dir}")
            continue
            
        for year in years:
            # Look for files like shots_YYYY.csv
            pattern = f"shots_{year}.csv"
            year_files = list(league_dir.glob(pattern))
            if not year_files:
                print(f"Warning: No files found for {league} {year} with pattern {pattern}")
            files.extend(year_files)
    return files

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the shot data"""
    # Select and rename columns
    df = df[['X', 'Y', 'situation', 'shotType', 'result']].copy()
    
    # Convert data types
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    
    # Convert categorical columns to string
    for col in ['situation', 'shotType', 'result']:
        df[col] = df[col].astype(str)
    
    return df

def main():
    # Load configuration
    config = load_config()
    prep_config = config.get('cleansing', {})
    
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    output_dir = data_dir / 'cleansed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of files to process
    leagues = prep_config.get('leagues', ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1'])
    years = list(range(prep_config.get('start_year', 2014), 
                       prep_config.get('end_year', 2024) + 1))
    
    files = get_raw_data_files(data_dir, leagues, years)
    
    if not files:
        print("No files found to process. Exiting.")
        return
    
    print(f"Found {len(files)} files to process...")
    
    # Process all files
    all_shots = []
    for file_path in tqdm(files, desc="Processing files"):
        try:
            df = pd.read_csv(file_path)
            df = clean_data(df)
            all_shots.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_shots:
        print("No data was processed. Exiting.")
        return
    
    # Combine all data
    combined = pd.concat(all_shots, ignore_index=True)
    
    # Save combined data
    output_path = output_dir / 'all_shots_cleansed.csv'
    combined.to_csv(output_path, index=False)
    print(f"\nSaved cleansed data to {output_path}")
    print(f"Total shots processed: {len(combined):,}")
    
    # Display summary
    print("\nData summary:")
    print(combined.dtypes)
    print("\nValue counts for categorical columns:")
    for col in ['situation', 'shotType', 'result']:
        print(f"\n{col}:")
        print(combined[col].value_counts().head())

if __name__ == "__main__":
    main()