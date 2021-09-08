"""
This python file formats the data from db.py
"""
import pandas as pd
from db import get_data, get_gameweek_history
from tqdm import tqdm
import os
tqdm.pandas()

# GLOBAL VARIABLES
cdPATH = os.getcwd()

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


def get_full_gw_hist(player_ids: list):
    """
    Get a gameweek stat history for all registred players

    Args:
        players_df (pd.Dataframe): A DataFrame containing id's for PL players

    Returns:
        A dataframe containing gameweek stats for all players in the input.
    """
    
    # get gameweek histories for each player
    points = player_ids.progress_apply(get_gameweek_history)

    # combine results into single dataframe
    points = pd.concat(df for df in points)
    
    return points

def transform_id_to_team(season: str):
    """
    Get PL team id and name for a specified season

    Args:
        season (str): A string specifying the season in the form of "2016-17"

    Returns:
        A dataframe containing the team id and name for the specified season
    """
    wdPATH = os.getcwd()

    with open(os.path.join(wdPATH, "..", "data_files", "master_team_list.csv")) as file:
        df = pd.read_csv(file)

    df = df.loc[df["season"] == season]
    
    df = df[["team", "team_name"]]
    
    df = df.rename(columns={'team': 'team_id'})

    return df

def create_data_for_exploration(season: str):
    """
    Get unfiltered data from a full season for data exploration

    Args:
        season (str): A string specifying the season in the form of "2020-21"

    Returns:
        A dataframe containing the team id and name for the specified season
    """
    print("create exploration data version 1.0")
    # Import data from previous seasons
    wdPATH = os.getcwd()

    with open(os.path.join(wdPATH, "..", "data_files", "2020-21", "gws", "merged_gw.csv")) as file:
        player_gw = pd.read_csv(file)
    
    player_gw = player_gw.rename(columns={'value': 'now_cost', 'total_points': 'event_points'})

    players_df = player_gw.rename(columns={'team': 'team_name'})
    
    # Fixture difficulty:
    # Get upcoming gameweek:
    with open(os.path.join(wdPATH, "..", "data_files", season, "fixtures.csv")) as file:
        fixture_df = pd.read_csv(file)
        
    season_team_df = transform_id_to_team(season) # df containing a name to each team associated with the id for the specified season
    players_df = pd.merge(players_df, season_team_df, how="left", on="team_name")
    # TO DO: Create a for loop attaching the FDR for each player for each round to the df    
    
    # 2. Separate home and away team FDR and append them into two columns
    fixture_a = fixture_df[["team_a", "team_a_difficulty", "event"]]
    fixture_a = fixture_a.rename(columns={'team_a': 'team', 'team_a_difficulty': 'fdr', 'event': 'round'})
    fixture_h = fixture_df[["team_h", "team_h_difficulty", "event"]]
    fixture_h = fixture_h.rename(columns={'team_h': 'team', 'team_h_difficulty': 'fdr', 'event': 'round'})
    fixture_difficulty = fixture_a.append(fixture_h)
    
    fixture_difficulty = fixture_difficulty.rename(columns={'team': 'team_id'})
    
    # 3. Merge the FDR to the main dataframe
    main_df = pd.merge(players_df, fixture_difficulty, how="left", on=['team_id', "round"])

    return main_df


def create_training_data(season: str):
    """ 
    MVP variables: 
    - id_player

    
    Player form:
    - ict_index
    - bonus (so far this season)
    - minutes (played so far this season)
    - points_per_game
    - form

    Player ability:
    the highest ict_index points of the previous last 3 seasons (to avoid more or less the same number compared to averages)
    if player hasn't played in PL previously then transform now_cost into a proxy ict_index number

    Fixture difficulty:
    - team (id on the team that the player currently plays for - categorical variable)
    - FD (use the team diffictuly rating for the next fixture for the team id that the player belongs to)

    """
    print("create training data version 1.0 stable")
    # Import data from previous seasons
    wdPATH = os.getcwd()

    with open(os.path.join(wdPATH, "..", "data_files", "2020-21", "gws", "merged_gw.csv")) as file:
        player_gw = pd.read_csv(file)
    
    player_gw = player_gw.rename(columns={'value': 'now_cost', 'total_points': 'event_points'})
    # Player form + ability:
    variables_to_keep = ['name', 'team', 'ict_index', 'bps', 'minutes', 'now_cost', 'event_points', 'round'] # should be able to change this in a config file
    players_df = player_gw[variables_to_keep]
    
    players_df = players_df.rename(columns={'team': 'team_name'})
    
    # Fixture difficulty:
    # Get upcoming gameweek:
    with open(os.path.join(wdPATH, "..", "data_files", season, "fixtures.csv")) as file:
        fixture_df = pd.read_csv(file)
        
    season_team_df = transform_id_to_team(season) # df containing a name to each team associated with the id for the specified season
    players_df = pd.merge(players_df, season_team_df, how="left", on="team_name")
    # TO DO: Create a for loop attaching the FDR for each player for each round to the df    
    
    # 2. Separate home and away team FDR and append them into two columns
    fixture_a = fixture_df[["team_a", "team_a_difficulty", "event"]]
    fixture_a = fixture_a.rename(columns={'team_a': 'team', 'team_a_difficulty': 'fdr', 'event': 'round'})
    fixture_h = fixture_df[["team_h", "team_h_difficulty", "event"]]
    fixture_h = fixture_h.rename(columns={'team_h': 'team', 'team_h_difficulty': 'fdr', 'event': 'round'})
    fixture_difficulty = fixture_a.append(fixture_h)
    
    fixture_difficulty = fixture_difficulty.rename(columns={'team': 'team_id'})
    
    # 3. Merge the FDR to the main dataframe
    main_df = pd.merge(players_df, fixture_difficulty, how="left", on=['team_id', "round"])
    
    # 4. Remove team and team_id as a feature to predict on to create a more generalized dataset to train on. 
    variables_to_keep = ['name','ict_index', 'bps', 'minutes', 'now_cost', 'team_id', 'fdr', 'event_points'] # should be able to change this in a config file
    main_df = main_df[variables_to_keep]

    return main_df


def create_evaluation_data():
    """ 
    MVP variables: 
    - id_player

    
    Player form:
    - ict_index
    - bonus (so far this season)
    - minutes (played so far this season)
    - points_per_game
    - form

    Player ability:
    the highest ict_index points of the previous last 3 seasons (to avoid more or less the same number compared to averages)
    if player hasn't played in PL previously then transform now_cost into a proxy ict_index number

    Fixture difficulty:
    - team (id on the team that the player currently plays for - categorical variable)
    - FDR (use the team diffictuly rating for the next fixture for the team id that the player belongs to)

    """
    print("test data version 0.1 stable")
    # Import data
    data = get_data()
    current_players = data[0]
    
    # Player form + ability:
    variables_to_keep = ['web_name', 'team', 'ict_index', 'bps', 'minutes', 'form', 'now_cost', 'event_points'] # should be able to change this in a config file
    current_players = current_players[variables_to_keep]
    
    # Fixture difficulty:
    # Get upcoming gameweek:
    fixture_df = data[3]
    # Get upcoming gameweek:
    current_gw = fixture_df.loc[fixture_df["finished"] == True]
    current_gw = current_gw.sort_values(by = ["event"], ascending=False)
    current_gw = current_gw.iloc[0][1]
    upcoming_gw = current_gw + 1
    
    # 1. Filter out the upcoming gameweek fixtures
    fixture_df = fixture_df.loc[fixture_df['event'] == upcoming_gw]
    
    # 2. Separate home and away team FDR and append them into two columns
    fixture_a = fixture_df[["team_a", "team_a_difficulty"]]
    fixture_a = fixture_a.rename(columns={'team_a': 'team', 'team_a_difficulty': 'fdr'})
    fixture_h = fixture_df[["team_h", "team_h_difficulty"]]
    fixture_h = fixture_h.rename(columns={'team_h': 'team', 'team_h_difficulty': 'fdr'})
    fixture_difficulty = fixture_a.append(fixture_h)
    
    # 3. Merge the FDR to the main dataframe
    main_df = pd.merge(current_players, fixture_difficulty, how="left", on='team')
    
    main_df = main_df.rename(columns={'web_name': 'name'})
    
    main_df['ict_index'] = main_df['ict_index'].astype('float')
    
    return main_df