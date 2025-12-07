import pandas as pd

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath, parse_dates=['date'])

def _calculate_rolling_stats(matches, window=10):
    """
    Internal helper to calculate rolling averages for teams.
    """
    # create a long view (one row per team per match)
    cols_h = ["match_id", "date", "home_team", "shots_h", "shot_xg_sum_h", "shots_on_target_h", "home_goals", "away_goals"]
    cols_a = ["match_id", "date", "away_team", "shots_a", "shot_xg_sum_a", "shots_on_target_a", "home_goals", "away_goals"]
    
    home = matches[cols_h].rename(columns={
        "home_team": "team", "shots_h": "shots", "shot_xg_sum_h": "xg", "shots_on_target_h": "sot",
        "home_goals": "goals_for", "away_goals": "goals_against"
    })
    
    away = matches[cols_a].rename(columns={
        "away_team": "team", "shots_a": "shots", "shot_xg_sum_a": "xg", "shots_on_target_a": "sot",
        "away_goals": "goals_for", "home_goals": "goals_against"
    })
    
    home["is_home"] = True
    away["is_home"] = False

    # calculate Points
    home["points"] = home.apply(lambda x: 3 if x.goals_for > x.goals_against else (1 if x.goals_for == x.goals_against else 0), axis=1)
    away["points"] = away.apply(lambda x: 3 if x.goals_for > x.goals_against else (1 if x.goals_for == x.goals_against else 0), axis=1)

    team_stats = pd.concat([home, away]).sort_values(["team", "date"])
    team_stats = team_stats.reset_index(drop=True)
    
    # calculate Rest Days
    team_stats["rest_days"] = team_stats.groupby("team")["date"].diff().dt.days
    team_stats["rest_days"] = team_stats["rest_days"].fillna(7)

    # calculate General Rolling Averages
    features = ["shots", "xg", "sot", "points", "goals_against", "goals_for"]
    
    grouped = team_stats.groupby("team")[features]
    rolling = grouped.apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    
    if isinstance(rolling.index, pd.MultiIndex):
        rolling = rolling.droplevel(0)
    
    rolling.columns = [f"roll_{c}" for c in rolling.columns]
    
    # group by Team, Venue 
    venue_features = ["points", "goals_for", "goals_against"]
    grouped_venue = team_stats.groupby(["team", "is_home"])[venue_features]
    
    # must shift(1) to ensure we only look at PAST games
    rolling_venue = grouped_venue.apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    
    # drop the extra index levels created by groupby
    if isinstance(rolling_venue.index, pd.MultiIndex):
        rolling_venue = rolling_venue.droplevel([0, 1])
    
    rolling_venue.columns = [f"roll_venue_{c}" for c in rolling_venue.columns]

    # combine everything
    team_stats_final = pd.concat([
        team_stats[["match_id", "team", "rest_days", "is_home"]], 
        rolling, 
        rolling_venue
    ], axis=1)
    
    return team_stats_final

def _merge_team_stats(matches, team_stats, team_type):
    """
    Helper to merge rolling stats for a specific team type ('home' or 'away').
    """
    prefix = f"{team_type}_"
    is_home = True if team_type == 'home' else False
    
  
    stats = team_stats[team_stats["is_home"] == is_home]
    
  
    matches = matches.merge(
        stats, 
        left_on=["match_id", f"{team_type}_team"], 
        right_on=["match_id", "team"], 
        how="left"
    )
    
    # Rename columns dynamically
    rename_map = {
        "roll_shots": f"{prefix}roll_shots", 
        "roll_xg": f"{prefix}roll_xg", 
        "roll_sot": f"{prefix}roll_sot",
        "roll_points": f"{prefix}roll_points",
        "roll_goals_against": f"{prefix}roll_goals_conceded",
        "roll_goals_for": f"{prefix}roll_goals_scored",
        "rest_days": f"{prefix}rest_days",
        "roll_venue_points": f"{prefix}venue_points",
        "roll_venue_goals_for": f"{prefix}venue_goals_scored",
        "roll_venue_goals_against": f"{prefix}venue_goals_conceded"
    }
    matches = matches.rename(columns=rename_map)

    matches.drop(columns=["team", "is_home"], inplace=True)
    
    # Fill NaNs in Venue stats with General stats
    cols_venue = [f"{prefix}venue_points", f"{prefix}venue_goals_scored", f"{prefix}venue_goals_conceded"]
    cols_general = [f"{prefix}roll_points", f"{prefix}roll_goals_scored", f"{prefix}roll_goals_conceded"]
    
    for v_col, g_col in zip(cols_venue, cols_general):
        matches[v_col] = matches[v_col].fillna(matches[g_col])
        
    return matches

def preprocess_data(df):
    print("Preprocessing data...")
    

    agg = df.groupby(["match_id", "h_a"]).agg(
        shots=("result", "count"),
        shot_xg_sum=("xG", "sum"),
        shots_on_target=("result", lambda s: ((s == "Goal") | (s == "SavedShot") | (s == "ShotOnPost")).sum())
    ).reset_index()

    pivot = agg.pivot(index="match_id", columns="h_a", values=["shots", "shot_xg_sum", "shots_on_target"])
    pivot.columns = ["_".join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    match_static = df.sort_values("date").groupby("match_id").first().reset_index()[
        ["match_id", "date", "home_team", "away_team", "home_goals", "away_goals"]
    ]

    matches = match_static.merge(pivot, on="match_id", how="left").fillna(0)

    # add rolling features
    print("Calculating rolling averages (last 10 games)...")
    team_rolling = _calculate_rolling_stats(matches, window=10)
    
    matches = _merge_team_stats(matches, team_rolling, "home")
    matches = _merge_team_stats(matches, team_rolling, "away")

    
    def get_outcome(row):
        if row["home_goals"] > row["away_goals"]: return "home"
        elif row["home_goals"] == row["away_goals"]: return "draw"
        else: return "away"
    
    matches["outcome"] = matches.apply(get_outcome, axis=1)
    
    # drop rows where we don't have enough history yet
    matches.dropna(subset=["home_roll_xg", "away_roll_xg"], inplace=True)
   
 
    matches["rest_days_diff"] = matches["home_rest_days"] - matches["away_rest_days"]
    matches["day_code"] = matches["date"].dt.dayofweek
    matches["hour"] = matches["date"].dt.hour

    # cleanpu
    cols_to_drop = [
        "match_id", "date", 
        "shots_h", "shots_a", 
        "shot_xg_sum_h", "shot_xg_sum_a", 
        "shots_on_target_h", "shots_on_target_a",
        "home_goals", "away_goals",
        "home_roll_shots", "away_roll_shots", 
        "home_roll_sot", "away_roll_sot",     
        "home_rest_days", "away_rest_days"    
    ]
    
    matches.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return matches