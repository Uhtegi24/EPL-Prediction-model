import pandas as pd

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath, parse_dates=['date'])

def _calculate_rolling_stats(matches, window=5):
    """
    Internal helper to calculate rolling averages for teams.
    """
    # 1. Create a long view (one row per team per match)
    # We need to know the result to calculate points
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
    
    # Calculate Points for each row
    # Home team points
    home["points"] = home.apply(lambda x: 3 if x.goals_for > x.goals_against else (1 if x.goals_for == x.goals_against else 0), axis=1)
    # Away team points (logic is same because we renamed cols to goals_for/against)
    away["points"] = away.apply(lambda x: 3 if x.goals_for > x.goals_against else (1 if x.goals_for == x.goals_against else 0), axis=1)

    team_stats = pd.concat([home, away]).sort_values(["team", "date"])
    team_stats = team_stats.reset_index(drop=True)
    
    # 2. Calculate Rolling Averages
    # Added "points" to the list
    features = ["shots", "xg", "sot", "points"]
    
    grouped = team_stats.groupby("team")[features]
    rolling = grouped.apply(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    
    if isinstance(rolling.index, pd.MultiIndex):
        rolling = rolling.droplevel(0)
    
    rolling.columns = [f"roll_{c}" for c in rolling.columns]
    
    # 3. Combine back
    team_stats_final = pd.concat([team_stats[["match_id", "team"]], rolling], axis=1)
    
    return team_stats_final

def preprocess_data(df):
    print("Preprocessing data...")
    
    # --- STEP 1: Basic Aggregation (Same as before) ---
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

    # --- STEP 2: Add Rolling Features (The New Part) ---
    print("Calculating rolling averages (last 5 games)...")
    team_rolling = _calculate_rolling_stats(matches, window=5)
    
    # Merge Home Stats
    matches = matches.merge(team_rolling, left_on=["match_id", "home_team"], right_on=["match_id", "team"], how="left")
    matches = matches.rename(columns={
        "roll_shots": "home_roll_shots", 
        "roll_xg": "home_roll_xg", 
        "roll_sot": "home_roll_sot",
        "roll_points": "home_roll_points" # <--- NEW
    })
    matches.drop(columns=["team"], inplace=True)
    
    # Merge Away Stats
    matches = matches.merge(team_rolling, left_on=["match_id", "away_team"], right_on=["match_id", "team"], how="left")
    matches = matches.rename(columns={
        "roll_shots": "away_roll_shots", 
        "roll_xg": "away_roll_xg", 
        "roll_sot": "away_roll_sot",
        "roll_points": "away_roll_points" # <--- NEW
    })
    matches.drop(columns=["team"], inplace=True)

    # --- STEP 3: Create Target ---
    def get_outcome(row):
        if row["home_goals"] > row["away_goals"]: return "home"
        elif row["home_goals"] == row["away_goals"]: return "draw"
        else: return "away"
    
    matches["outcome"] = matches.apply(get_outcome, axis=1)
    
    # Drop the first few rows where rolling stats are NaN (start of season/dataset)
    matches.dropna(subset=["home_roll_shots", "away_roll_shots"], inplace=True)
    
    return matches