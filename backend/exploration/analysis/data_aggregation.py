import pandas as pd

def create_player_match_stats(df_with_xg):
    """
    Convert the detailed df_with_xg (shot-level data) into a summary DataFrame
    where each row represents a player's aggregated statistics for a single match.
    """
    player_match_df = df_with_xg.groupby(['match_id', 'player_id', 'h_a', 'date']).agg(
        total_xg_basic_match=('xG_basic_model', 'sum'),
        total_xg_situation_match=('xG_situation_model', 'sum'),
        total_xg_shottype_match=('xG_shottype_model', 'sum'),
        total_xg_advanced_match=('xG_advanced_model', 'sum'),
        total_goals_match=('result', lambda x: (x == 'Goal').sum())
    ).reset_index()

    player_match_df = player_match_df.sort_values(by=['player_id', 'date']).reset_index(drop=True)
    return player_match_df

def create_lagged_features(player_match_df, past_window_size=5, future_window_size=5):
    """
    For each player and each match, calculate their aggregated performance over a
    past_window_size and their actual goals over a future_window_size.
    """
    lagged_analysis_df = player_match_df.copy()

    # Ensure sorted for correct rolling/shifting
    lagged_analysis_df = lagged_analysis_df.sort_values(by=['player_id', 'date'])

    # Calculate Past Features
    for col in ['total_xg_basic_match', 'total_xg_situation_match', 'total_xg_shottype_match', 'total_xg_advanced_match', 'total_goals_match']:
        lagged_analysis_df[f'past_{col.replace("total_", "").replace("_match", "")}'] = \
            lagged_analysis_df.groupby('player_id')[col].rolling(
                window=past_window_size, closed='left'
            ).sum().reset_index(level=0, drop=True)

    # Calculate Future Outcome
    lagged_analysis_df['future_goals'] = \
        lagged_analysis_df.groupby('player_id')['total_goals_match'].shift(
            periods=-(future_window_size), axis=0
        ).rolling(
            window=future_window_size, closed='left'
        ).sum().reset_index(level=0, drop=True)

    # Drop rows with NaN values (where past or future windows couldn't be fully calculated)
    lagged_analysis_df = lagged_analysis_df.dropna().reset_index(drop=True)

    return lagged_analysis_df