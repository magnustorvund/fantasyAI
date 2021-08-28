"""
This python file formats the data from db.py
"""
import pandas as pd
from db import get_data, get_gameweek_history, get_season_history
from tqdm import tqdm
tqdm.pandas()

  
def create_player_df():
    """
    Create a baseline player dataframe for all players

    Args:
        player_id: The player_id of the player

    Returns:
        A dataframe containing gameweek stats for each gameweek in the current season
    """
    
    # retrieve data
    data = get_data()
    players = data[0]
    teams = data[1]
    positions = data[2]
    
    # select columns of interest from players df
    players = players[
        ['id', 'first_name', 'second_name', 'web_name', 'team',
        'element_type']
    ]

    # join team name
    players = players.merge(
        teams[['id', 'name']],
        left_on='team',
        right_on='id',
        suffixes=['_player', None]
    ).drop(['team', 'id'], axis=1)

    # join player positions
    players = players.merge(
        positions[['id', 'singular_name_short']],
        left_on='element_type',
        right_on='id'
    ).drop(['element_type', 'id'], axis=1)
    
    return players


def get_full_gw_hist(players_df: pd.DataFrame):
    """
    Get a gameweek stat history for all registred players

    Args:
        players_df (pd.Dataframe): A DataFrame containing id's for PL players

    Returns:
        A dataframe containing gameweek stats for all players in the input.
    """
    players = players_df
    
    # get gameweek histories for each player
    points = players['id_player'].progress_apply(get_gameweek_history)

    # combine results into single dataframe
    points = pd.concat(df for df in points)

    # join web_name
    points = players[['id_player', 'web_name', 'singular_name_short']].merge(
        points,
        left_on='id_player',
        right_on='element'
    )
    
    return points