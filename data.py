"""
This python file formats the data from db.py
"""
import pandas as pd
from db import get_data, get_gameweek_history
from tqdm import tqdm
import os
import copy
tqdm.pandas()

"""
    # TO DO:
       - Transform minutes to avergage minutes per match so far this season both for evaluation and create_training_data (average of up to the last 5 matches)
       - Transform evaluation data ict_index, bps, and now cost to current (filter out the rows where gw or round == current gameweek)
       - Add a feature "ict_index_change_lag1", where the change in ict_index from gw(current-2) to gw(current-1)
       - Add more time series features to catch minor changes which can signal that a better form is on its way (increase in goal creations, goals/chance ratio etc. - possibly use other data sources) 
       
"""

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

def pred_fdr(current_players_df, fixture_df, upcoming_gw):
    # 1. Filter out the upcoming gameweek fixture difficulties
    fixture_df = fixture_df.loc[fixture_df['event'] == upcoming_gw]
    
    # 2. Separate home and away team FDR and append them into two columns
    fixture_a = fixture_df[["team_a", "team_a_difficulty"]]
    fixture_a = fixture_a.rename(columns={'team_a': 'team', 'team_a_difficulty': 'fdr'})
    fixture_h = fixture_df[["team_h", "team_h_difficulty"]]
    fixture_h = fixture_h.rename(columns={'team_h': 'team', 'team_h_difficulty': 'fdr'})
    fixture_difficulty = fixture_a.append(fixture_h)
    
    # 3. Merge the FDR to the main dataframe
    main_df = pd.merge(current_players_df, fixture_difficulty, how="left", on='team')
    
    main_df = main_df.rename(columns={'web_name': 'name'})
    
    main_df['ict_index'] = main_df['ict_index'].astype('float')
    
    return main_df

def test_fdr(current_players_df, fixture_df, current_gw):
    # 1. Filter out the current gameweek fixture difficulties
    fixture_df = fixture_df.loc[fixture_df['event'] == current_gw]
    
    # 2. Separate home and away team FDR and append them into two columns
    fixture_a = fixture_df[["team_a", "team_a_difficulty"]]
    fixture_a = fixture_a.rename(columns={'team_a': 'team', 'team_a_difficulty': 'fdr'})
    fixture_h = fixture_df[["team_h", "team_h_difficulty"]]
    fixture_h = fixture_h.rename(columns={'team_h': 'team', 'team_h_difficulty': 'fdr'})
    fixture_difficulty = fixture_a.append(fixture_h)
    
    # 3. Merge the FDR to the main dataframe
    main_df = pd.merge(current_players_df, fixture_difficulty, how="left", on='team')
    
    main_df = main_df.rename(columns={'web_name': 'name'})
    
    main_df['ict_index'] = main_df['ict_index'].astype('float')
    
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


def create_evaluation_data2(session: str):
    """ 
    session (str): "test" or "pred"
    
    
    MVP variables: 
    - id_player

    
    Player form:
    - ict_index (current)
    - bps (current)
    - minutes (average minutes played per match the latest <= 5 matches) - !!!TO DO
    - fdr (for next game)
    - now_cost (current)
    
    Player ability:
    the highest ict_index points of the previous last 3 seasons (to avoid more or less the same number compared to averages)
    if player hasn't played in PL previously then transform now_cost into a proxy ict_index number

    Fixture difficulty:
    - team (id on the team that the player currently plays for - categorical variable)
    - FDR (use the team diffictuly rating for the next fixture for the team id that the player belongs to)

    returns: a prediction quality test dataframe or a prediction dataframe for predicting next week

    """
    print("test data version 0.4")
    # Import data
    db_data = get_data()
    data = db_data[0]
    current_players = data[['id']]
    current_players = tuple(current_players['id'])
    
    df = pd.DataFrame()
    
    for player in current_players:
        player_gw = get_gameweek_history(player)
        player_gw['id'] = player
        df = df.append(player_gw)
    
    df = df.rename(columns={'total_points': 'event_points', 'value': 'now_cost'})
    extra = data[['id', 'team', 'web_name']]
    df = pd.merge(df, extra, how='left', on='id')
    
    # Player form + ability:
    variables_to_keep = ['web_name', 'team', 'ict_index', 'bps', 'round', 'minutes', 'now_cost', 'event_points'] # should be able to change this in a config file
    current_players_df = df[variables_to_keep]
    
    # Fixture difficulty:
    # Get upcoming gameweek:
    fixture_df = db_data[3]
    # Get upcoming gameweek:
    current_gw = fixture_df.loc[fixture_df["finished"] == True]
    current_gw = current_gw.sort_values(by = ["event"], ascending=False)
    current_gw = current_gw.iloc[0][1]
    upcoming_gw = current_gw + 1  
    
    # Test
    if session == "pred":
        main_df = pred_fdr(current_players_df, fixture_df, upcoming_gw)
        print(f"upcoming gw to predict for is {upcoming_gw}")
        main_df['avg_minutes'] = 0
        main_df['ict_index_change'] = 0
        main_df['bps_change'] = 0

        output_df = pd.DataFrame()
        current_round_df = pd.DataFrame()
        
        for name in main_df['name']:
            input_df = copy.copy(main_df)
            input_df = input_df.loc[input_df['name'] == name]
            input_df['avg_minutes'] = input_df["minutes"].mean()
            if current_gw > 1:
                input_df['ict_index_change'] = input_df["ict_index"].diff(periods=1)
                input_df['bps_change'] = input_df["bps"].diff(periods=1)
            output_df = output_df.append(input_df)

        output_df = output_df.drop_duplicates(subset=["name"], keep="last") # keeping only the row with the latest difference stats
        
    elif session == "test":
        main_df = test_fdr(current_players_df, fixture_df, current_gw)
        print(f"current_gw to get data test for is {current_gw}")
        
        main_df['avg_minutes'] = 0
        main_df['ict_index_change'] = 0
        main_df['bps_change'] = 0

        output_df = pd.DataFrame()
        current_round_df = pd.DataFrame()
        
        for name in main_df['name']:
            current_round_df = main_df.loc[main_df["round"] == current_gw]

        current_round_df = current_round_df[["name", "event_points"]]
        previous_gw_list = list(range(1,current_gw))
        # The dataframe for testing the prediction quality (lag all features by removing current gw stats and see what features could possibly explain next weeks points):
        for name in main_df['name']:
            input_df = copy.copy(main_df)
            input_df = input_df.loc[input_df['name'] == name]
            input_df = input_df[input_df["round"].isin(previous_gw_list)] # exclude current GW
            input_df['avg_minutes'] = input_df["minutes"].mean()
            if current_gw > 2:
                input_df['ict_index_change'] = input_df["ict_index"].diff(periods=1)
                input_df['bps_change'] = input_df["bps"].diff(periods=1)
            output_df = output_df.append(input_df)

        output_df = output_df.drop_duplicates(subset=["name"], keep="last") # keeping only the row with the latest difference stats
        output_df = output_df.drop(['event_points'], axis='columns')
        output_df = output_df.merge(current_round_df, how="left", on="name")
    
    else:
        print("You must have typed wrong session name, try pred or test")
    
    return output_df


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
    variables_to_keep = ['web_name', 'team', 'ict_index', 'bps', 'minutes', 'now_cost', 'event_points'] # should be able to change this in a config file
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