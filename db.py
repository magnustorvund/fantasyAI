"""
    This python file connects to endpoints in the fantasy premier league
    and convert the json data into dataframes
"""
    
import requests, json
import pandas as pd

# base url
BASE_URL = 'https://fantasy.premierleague.com/api/'

def get_data():
    """
    Retrieve data on players, teams and positions from fantasy PL from a static endpoint

    Returns:
        Dataframe for players, teams and positions
    """
    # the specific request endpoint
    r = requests.get(BASE_URL + 'bootstrap-static/').json()
    r2 = requests.get(BASE_URL + 'fixtures/').json()
    
    # get player information
    players = pd.json_normalize(r['elements'])
    # get team information
    teams = pd.json_normalize(r['teams'])
    # get position information
    positions = pd.json_normalize(r['element_types'])
    # get fixture information
    fixture = pd.json_normalize(r2)

    return players, teams, positions, fixture

def get_gameweek_history(player_id):
    """
    Get gameweek history for a given player_id (PID) in current season

    Args:
        player_id: The player_id of the player

    Returns:
        A dataframe containing gameweek stats for each gameweek so far in the current season
    """
    # send GET request to
    # https://fantasy.premierleague.com/api/element-summary/{PID}/
    gw_r = requests.get(BASE_URL + 'element-summary/' + str(player_id) + '/').json()
    # extract 'history' data from response into dataframe
    df = pd.json_normalize(gw_r['history'])
    
    return df

def get_season_history(player_id):
    """
    Get stats from previous seasons for given player_ids (PID)

    Args:
        player_id: The player_id of the player(s)

    Returns:
        A dataframe containing stats from previous premier league seasons for the specified players
    """
    
    # send GET request to
    # https://fantasy.premierleague.com/api/element-summary/{PID}/
    df = pd.DataFrame()
    for player in player_id:
        r = requests.get(
            BASE_URL + 'element-summary/' + str(player) + '/'
            ).json()
         
        # extract 'history_past' data from response into dataframe
        history = pd.json_normalize(r['history_past'])
        history["id"] = player # adding a column with the id value from the i-th index of the list
        df = df.append(history)
        print(f"Appended season history for player_id: {player}")
    return df
