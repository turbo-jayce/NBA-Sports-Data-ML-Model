import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


# === STEP 1: ENHANCED DATA LOADING AND CLEANING ===

def load_season_pbp(filepath, season_tag):
    """Load and clean play-by-play data with enhanced preprocessing"""
    df = pd.read_csv(filepath)
    df['season'] = season_tag

    # Enhanced text cleaning
    df['actionType'] = df['actionType'].fillna('unknown').str.lower().str.strip()
    df['subType'] = df['subType'].fillna('unknown').str.lower().str.strip()
    df['description'] = df['description'].fillna('').str.strip()
    df['playerNameI'] = df['playerNameI'].fillna('unknown').astype(str).str.strip()

    # Convert period to numeric and handle overtime
    df['period'] = pd.to_numeric(df['period'], errors='coerce').fillna(1)
    df['is_overtime'] = (df['period'] > 4).astype(int)

    # Parse clock time to seconds remaining
    df['seconds_remaining'] = df['clock'].apply(parse_clock_to_seconds)

    return df


def parse_clock_to_seconds(clock_str):
    """Convert clock string (MM:SS) to seconds remaining in period"""
    try:
        if pd.isna(clock_str) or clock_str == '':
            return 0
        parts = str(clock_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
    except:
        return 0


print("Loading data...")
pbp_2223 = load_season_pbp("S2223/S2223_all_play-by-play.csv", "2022-23")
pbp_2324 = load_season_pbp("S2324/S2324_all_play-by-play.csv", "2023-24")
pbp_2425 = load_season_pbp("S2425/S2425_all_play-by-play.csv", "2024-25")

pbp_all = pd.concat([pbp_2223, pbp_2324, pbp_2425], ignore_index=True)


# === STEP 2: ENHANCED STAT EXTRACTION WITH MORE DETAIL ===

def extract_comprehensive_player_stats(pbp):
    """Extract comprehensive player statistics with maximum accuracy"""

    # Points - handle all scoring types with more detail
    scoring_plays = pbp[pbp['isFieldGoal'] == 1].copy()
    scoring_plays['points_scored'] = scoring_plays['shotVal'].fillna(0)

    # Separate 2PT and 3PT makes/attempts
    fg_2pt = scoring_plays[scoring_plays['shotVal'] == 2]
    fg_3pt = scoring_plays[scoring_plays['shotVal'] == 3]

    pts_2pt = fg_2pt.groupby(['game_id', 'playerNameI']).agg({
        'shotVal': ['count', 'sum'],
        'description': lambda x: sum('MAKES' in str(desc) for desc in x)
    }).reset_index()
    pts_2pt.columns = ['game_id', 'playerNameI', 'fga_2pt', 'pts_2pt_total', 'fgm_2pt']

    pts_3pt = fg_3pt.groupby(['game_id', 'playerNameI']).agg({
        'shotVal': ['count', 'sum'],
        'description': lambda x: sum('MAKES' in str(desc) for desc in x)
    }).reset_index()
    pts_3pt.columns = ['game_id', 'playerNameI', 'fga_3pt', 'pts_3pt_total', 'fgm_3pt']

    # Free throws with attempts
    ft_attempts = pbp[pbp['actionType'].str.contains('free throw', na=False)]

    ft_stats = ft_attempts.groupby(['game_id', 'playerNameI']).agg({
        'description': [
            lambda x: sum('MAKES' in str(desc) for desc in x),
            'count'
        ]
    }).reset_index()
    ft_stats.columns = ['game_id', 'playerNameI', 'ftm', 'fta']
    ft_stats['ft_points'] = ft_stats['ftm']

    # Field goal points total
    fg_points = scoring_plays.groupby(['game_id', 'playerNameI'])['points_scored'].sum().reset_index()

    # Combine all points
    all_points = fg_points.merge(ft_stats[['game_id', 'playerNameI', 'ft_points']],
                                 on=['game_id', 'playerNameI'], how='outer').fillna(0)
    all_points['points'] = all_points['points_scored'] + all_points['ft_points']
    points = all_points[['game_id', 'playerNameI', 'points']]

    # Enhanced rebounds
    reb_plays = pbp[pbp['description'].str.contains("REBOUND", na=False)].copy()

    # Player rebounds by type
    off_reb = reb_plays[reb_plays['description'].str.contains("Offensive", na=False)]
    off_reb_stats = off_reb.groupby(['game_id', 'playerNameI']).size().reset_index(name='off_rebounds')

    def_reb = reb_plays[reb_plays['description'].str.contains("Defensive", na=False)]
    def_reb_stats = def_reb.groupby(['game_id', 'playerNameI']).size().reset_index(name='def_rebounds')

    total_reb = reb_plays.groupby(['game_id', 'playerNameI']).size().reset_index(name='rebounds')

    # Enhanced assists
    assist_pattern = r'\(([\w\.\'\-\s]+)\s+(\d+)\s+AST\)'
    assist_rows = pbp[pbp['description'].str.contains(assist_pattern, na=False, regex=True)]

    assists_list = []
    for _, row in assist_rows.iterrows():
        match = pd.Series([row['description']]).str.extract(assist_pattern)
        if not match.isna().any().any():
            assists_list.append({
                'game_id': row['game_id'],
                'playerNameI': match[0].iloc[0].strip(),
                'assists': int(match[1].iloc[0]),
                'period': row['period']
            })

    if assists_list:
        assists_df = pd.DataFrame(assists_list)
        assists = assists_df.groupby(['game_id', 'playerNameI'])['assists'].sum().reset_index()
        assists_first_half = assists_df[assists_df['period'] <= 2].groupby(['game_id', 'playerNameI'])[
            'assists'].sum().reset_index()
        assists_first_half.rename(columns={'assists': 'assists_first_half'}, inplace=True)
    else:
        assists = pd.DataFrame(columns=['game_id', 'playerNameI', 'assists'])
        assists_first_half = pd.DataFrame(columns=['game_id', 'playerNameI', 'assists_first_half'])

    # Additional stats
    turnovers = pbp[pbp['actionType'].str.contains('turnover', na=False)]
    to_stats = turnovers.groupby(['game_id', 'playerNameI']).size().reset_index(name='turnovers')

    steals = pbp[pbp['actionType'].str.contains('steal', na=False)]
    steal_stats = steals.groupby(['game_id', 'playerNameI']).size().reset_index(name='steals')

    blocks = pbp[pbp['actionType'].str.contains('block', na=False)]
    block_stats = blocks.groupby(['game_id', 'playerNameI']).size().reset_index(name='blocks')

    fouls = pbp[pbp['actionType'].str.contains('foul', na=False)]
    foul_stats = fouls.groupby(['game_id', 'playerNameI']).size().reset_index(name='fouls')

    # Merge all stats
    stats = points.copy()
    for df_to_merge in [total_reb, off_reb_stats, def_reb_stats, assists, assists_first_half,
                        pts_2pt, pts_3pt, ft_stats, to_stats, steal_stats,
                        block_stats, foul_stats]:
        stats = stats.merge(df_to_merge, on=['game_id', 'playerNameI'], how='outer')

    return stats.fillna(0)


print("Extracting comprehensive player statistics...")
player_stats = extract_comprehensive_player_stats(pbp_all)


# === STEP 3: IMPROVED MINUTES WITH STARTER/BENCH DETECTION ===

def estimate_minutes_and_role(pbp):
    """Enhanced minutes estimation with role detection"""

    # Method 1: Unique time periods
    pbp['time_key'] = pbp['period'].astype(str) + '_' + pbp['clock'].astype(str)
    unique_times = pbp.groupby(['game_id', 'playerNameI'])['time_key'].nunique().reset_index(name='unique_times')

    # Method 2: Game time span
    pbp['game_seconds'] = (pbp['period'] - 1) * 12 * 60 + (12 * 60 - pbp['seconds_remaining'])

    time_span = pbp.groupby(['game_id', 'playerNameI'])['game_seconds'].agg(['min', 'max']).reset_index()
    time_span['time_span_minutes'] = (time_span['max'] - time_span['min']) / 60
    time_span = time_span[['game_id', 'playerNameI', 'time_span_minutes']]

    # Method 3: Early game presence (starter indicator)
    early_game = pbp[(pbp['period'] == 1) & (pbp['seconds_remaining'] > 10 * 60)]
    early_presence = early_game.groupby(['game_id', 'playerNameI']).size().reset_index(name='early_events')
    early_presence['likely_starter'] = (early_presence['early_events'] > 2).astype(int)

    # Method 4: Late game presence (closer indicator)
    late_game = pbp[(pbp['period'] >= 4) & (pbp['seconds_remaining'] < 5 * 60)]
    late_presence = late_game.groupby(['game_id', 'playerNameI']).size().reset_index(name='late_events')
    late_presence['likely_closer'] = (late_presence['late_events'] > 3).astype(int)

    # Event density
    event_counts = pbp.groupby(['game_id', 'playerNameI']).size().reset_index(name='total_events')

    # Combine all methods
    minutes_data = unique_times.merge(time_span, on=['game_id', 'playerNameI'], how='outer')
    minutes_data = minutes_data.merge(event_counts, on=['game_id', 'playerNameI'], how='outer')
    minutes_data = minutes_data.merge(early_presence[['game_id', 'playerNameI', 'likely_starter']],
                                      on=['game_id', 'playerNameI'], how='outer')
    minutes_data = minutes_data.merge(late_presence[['game_id', 'playerNameI', 'likely_closer']],
                                      on=['game_id', 'playerNameI'], how='outer')
    minutes_data = minutes_data.fillna(0)

    # Improved minutes calculation with role weighting
    starter_bonus = minutes_data['likely_starter'] * 8
    closer_bonus = minutes_data['likely_closer'] * 4

    minutes_data['minutes'] = np.clip(
        0.3 * (minutes_data['unique_times'] * 0.6) +
        0.3 * minutes_data['time_span_minutes'] +
        0.2 * (minutes_data['total_events'] * 0.25) +
        0.1 * starter_bonus +
        0.1 * closer_bonus,
        0, 48
    )

    return minutes_data[['game_id', 'playerNameI', 'minutes', 'likely_starter', 'likely_closer']]


print("Estimating minutes and roles...")
minutes_data = estimate_minutes_and_role(pbp_all)
player_stats = player_stats.merge(minutes_data, on=['game_id', 'playerNameI'], how='left')
player_stats[['minutes', 'likely_starter', 'likely_closer']] = player_stats[
    ['minutes', 'likely_starter', 'likely_closer']].fillna(0)

# === STEP 4: ADVANCED FEATURE ENGINEERING (NO OVERLAPS) ===

print("Advanced feature engineering...")

# Add team and opponent context
team_info = pbp_all[['game_id', 'playerNameI', 'teamTricode']].dropna().drop_duplicates()
home_away = pbp_all[['game_id', 'home', 'away']].drop_duplicates()

player_stats = player_stats.merge(team_info, on=['game_id', 'playerNameI'], how='left')
player_stats = player_stats.merge(home_away, on='game_id', how='left')

# Home/away and opponent
player_stats['is_home'] = (player_stats['teamTricode'] == player_stats['home']).astype(int)
player_stats['opponent'] = np.where(player_stats['is_home'], player_stats['away'], player_stats['home'])

# Season and game sequencing
player_stats['season'] = player_stats['game_id'].str[:5]
player_stats = player_stats.sort_values(['playerNameI', 'season', 'game_id'])
player_stats['season_game_num'] = player_stats.groupby(['season', 'playerNameI']).cumcount() + 1
player_stats['career_game_num'] = player_stats.groupby('playerNameI').cumcount() + 1

# CALCULATED EFFICIENCY METRICS (not averages)
player_stats['true_shooting_pct'] = np.where(
    (player_stats['fga_2pt'] + player_stats['fga_3pt'] + 0.44 * player_stats['fta']) > 0,
    player_stats['points'] / (2 * (player_stats['fga_2pt'] + player_stats['fga_3pt'] + 0.44 * player_stats['fta'])),
    0
)

player_stats['assist_to_turnover'] = np.where(
    player_stats['turnovers'] > 0,
    player_stats['assists'] / player_stats['turnovers'],
    player_stats['assists'] * 2  # If no turnovers, give bonus
)

player_stats['usage_proxy'] = (player_stats['fga_2pt'] + player_stats['fga_3pt'] +
                               player_stats['fta'] + player_stats['turnovers']) / np.maximum(player_stats['minutes'], 1)

player_stats['rebound_rate'] = player_stats['rebounds'] / np.maximum(player_stats['minutes'], 1)
player_stats['assist_rate'] = player_stats['assists'] / np.maximum(player_stats['minutes'], 1)
player_stats['scoring_efficiency'] = player_stats['points'] / np.maximum(
    player_stats['fga_2pt'] + player_stats['fga_3pt'] + player_stats['fta'], 1)


# ROLLING FEATURES - Comprehensive but non-overlapping
def add_comprehensive_rolling_features(df, stats_cols, windows=[3, 5, 7, 10]):
    """Add rolling features ensuring no duplicates"""
    for window in windows:
        for stat in stats_cols:
            # Rolling mean
            col_name = f'{stat}_avg_{window}g'
            if col_name not in df.columns:  # Prevent duplicates
                df[col_name] = df.groupby('playerNameI')[stat].rolling(
                    window=window, min_periods=max(1, window // 3)
                ).mean().shift(1).reset_index(0, drop=True).fillna(0)

            # Rolling std (consistency) - only for key stats
            if stat in ['points', 'assists', 'rebounds', 'minutes'] and window in [5, 10]:
                std_col_name = f'{stat}_std_{window}g'
                if std_col_name not in df.columns:
                    df[std_col_name] = df.groupby('playerNameI')[stat].rolling(
                        window=window, min_periods=max(2, window // 3)
                    ).std().shift(1).reset_index(0, drop=True).fillna(0)

            # Rolling max (ceiling) - only for shorter windows and key stats
            if window <= 5 and stat in ['points', 'rebounds', 'assists']:
                max_col_name = f'{stat}_max_{window}g'
                if max_col_name not in df.columns:
                    df[max_col_name] = df.groupby('playerNameI')[stat].rolling(
                        window=window, min_periods=1
                    ).max().shift(1).reset_index(0, drop=True).fillna(0)

    return df


# Core stats for rolling features
core_stats = ['points', 'rebounds', 'assists', 'minutes', 'turnovers', 'steals', 'blocks', 'fouls']
player_stats = add_comprehensive_rolling_features(player_stats, core_stats, windows=[3, 5, 7, 10])

# Efficiency stats rolling (shorter windows only to avoid overlap)
efficiency_stats = ['true_shooting_pct', 'assist_to_turnover', 'usage_proxy', 'rebound_rate', 'assist_rate']
player_stats = add_comprehensive_rolling_features(player_stats, efficiency_stats, windows=[5, 10])

# PERFORMANCE TRENDS AND MOMENTUM
for stat in ['points', 'assists', 'rebounds']:
    # Linear trend
    trend_col = f'{stat}_trend_5g'
    if trend_col not in player_stats.columns:
        player_stats[trend_col] = (
            player_stats.groupby('playerNameI')[stat]
            .rolling(window=5, min_periods=3)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0)
            .shift(1).reset_index(0, drop=True).fillna(0)
        )

    # Recent vs long-term performance ratio
    momentum_col = f'{stat}_momentum'
    recent_col = f'{stat}_avg_3g'
    longterm_col = f'{stat}_avg_10g'
    if momentum_col not in player_stats.columns and recent_col in player_stats.columns and longterm_col in player_stats.columns:
        player_stats[momentum_col] = np.where(
            player_stats[longterm_col] > 0.1,  # Avoid division by very small numbers
            player_stats[recent_col] / player_stats[longterm_col],
            1
        )

# REST AND GAME CONTEXT
player_stats['games_since_last'] = player_stats.groupby('playerNameI')['career_game_num'].diff().fillna(1)
player_stats['is_back_to_back'] = (player_stats['games_since_last'] == 1).astype(int)
player_stats['is_rested'] = (player_stats['games_since_last'] > 2).astype(int)

# Game context features
player_stats['season_progress'] = player_stats['season_game_num'] / 82
player_stats['is_season_start'] = (player_stats['season_game_num'] <= 10).astype(int)
player_stats['is_season_end'] = (player_stats['season_game_num'] >= 72).astype(int)

# OPPONENT ANALYSIS (avoiding duplicates)
# Team offensive stats (what opponent typically scores)
team_offense_stats = player_stats.groupby(['season', 'teamTricode']).agg({
    'points': 'mean',
    'assists': 'mean',
    'rebounds': 'mean',
    'true_shooting_pct': 'mean'
}).reset_index()
team_offense_stats.columns = ['season', 'opponent', 'opponent_off_points_avg',
                              'opponent_off_assists_avg', 'opponent_off_rebounds_avg', 'opponent_off_ts_avg']

# Team defensive stats (what opponent typically allows)
team_defense_stats = player_stats.groupby(['season', 'opponent']).agg({
    'points': 'mean',
    'assists': 'mean',
    'rebounds': 'mean'
}).reset_index()
team_defense_stats.columns = ['season', 'teamTricode', 'opponent_def_points_allowed',
                              'opponent_def_assists_allowed', 'opponent_def_rebounds_allowed']

player_stats = player_stats.merge(team_offense_stats, on=['season', 'opponent'], how='left')
player_stats = player_stats.merge(team_defense_stats, on=['season', 'teamTricode'], how='left')

# Fill NAs for opponent stats
opponent_stat_cols = [col for col in player_stats.columns if col.startswith('opponent_')]
for col in opponent_stat_cols:
    player_stats[col] = player_stats[col].fillna(player_stats[col].mean())

# PLAYER CLUSTERING
print("Creating player clusters...")
cluster_features = ['points_avg_10g', 'rebounds_avg_10g', 'assists_avg_10g',
                    'minutes_avg_10g', 'true_shooting_pct_avg_10g', 'usage_proxy_avg_10g']

# Only cluster players with enough games
cluster_data = player_stats[player_stats['career_game_num'] >= 10][['playerNameI'] + cluster_features].copy()
cluster_data = cluster_data.groupby('playerNameI')[cluster_features].mean().reset_index()
cluster_data = cluster_data.fillna(0)

# Standardize for clustering
scaler = StandardScaler()
cluster_features_scaled = scaler.fit_transform(cluster_data[cluster_features])

# K-means clustering
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
cluster_data['player_cluster'] = kmeans.fit_predict(cluster_features_scaled)

# Map clusters back to main data
cluster_map = cluster_data.set_index('playerNameI')['player_cluster'].to_dict()
player_stats['player_cluster'] = player_stats['playerNameI'].map(cluster_map).fillna(0).astype(int)

# POSITION ENCODING
player_season_stats = player_stats.groupby(['playerNameI', 'season']).agg({
    'points': 'mean', 'rebounds': 'mean', 'assists': 'mean',
    'blocks': 'mean', 'steals': 'mean', 'usage_proxy': 'mean'
}).reset_index()


def advanced_position_classification(row):
    if row['assists'] > 6 and row['steals'] > 1:
        return 'point_guard'
    elif row['assists'] > 4 and row['points'] > 15:
        return 'combo_guard'
    elif row['points'] > 18 and row['rebounds'] < 6:
        return 'wing_scorer'
    elif row['rebounds'] > 9 and row['blocks'] > 1:
        return 'center'
    elif row['rebounds'] > 7 and row['points'] > 12:
        return 'power_forward'
    elif row['points'] > 12 and row['rebounds'] > 4:
        return 'forward'
    else:
        return 'role_player'


player_season_stats['position_detailed'] = player_season_stats.apply(advanced_position_classification, axis=1)
position_map = player_season_stats.set_index(['playerNameI', 'season'])['position_detailed'].to_dict()
player_stats['position_detailed'] = player_stats.apply(
    lambda row: position_map.get((row['playerNameI'], row['season']), 'role_player'), axis=1
)

# Encode categorical variables
le_pos = LabelEncoder()
player_stats['position_encoded'] = le_pos.fit_transform(player_stats['position_detailed'])

le_opp = LabelEncoder()
player_stats['opponent_encoded'] = le_opp.fit_transform(player_stats['opponent'].fillna('UNK'))

# === STEP 5: OPTIMIZED MODEL TRAINING WITH CAREFUL FEATURE SELECTION ===

print("Preparing optimized models with no feature overlap...")

# DEFINE DISTINCT FEATURE SETS (NO OVERLAPS)

# Base contextual features
base_features = [
    'is_home', 'season_game_num', 'career_game_num', 'minutes',
    'is_back_to_back', 'is_rested', 'likely_starter', 'likely_closer',
    'position_encoded', 'opponent_encoded', 'player_cluster',
    'season_progress', 'is_season_start', 'is_season_end'
]

# Rolling average features (get all available)
rolling_avg_features = [col for col in player_stats.columns if '_avg_' in col and col.endswith('g')]

# Rolling variability features
rolling_std_features = [col for col in player_stats.columns if '_std_' in col and col.endswith('g')]

# Rolling ceiling features
rolling_max_features = [col for col in player_stats.columns if '_max_' in col and col.endswith('g')]

# Trend and momentum features
trend_features = [col for col in player_stats.columns if '_trend_' in col or '_momentum' in col]

# Opponent context features
opponent_features = [col for col in player_stats.columns if col.startswith('opponent_')]

# Current game efficiency features (not rolling averages)
current_efficiency_features = ['true_shooting_pct', 'assist_to_turnover', 'usage_proxy',
                               'rebound_rate', 'assist_rate', 'scoring_efficiency']

# Raw stat features (for some models)
raw_stat_features = ['off_rebounds', 'def_rebounds', 'fga_2pt', 'fga_3pt', 'ftm', 'fta',
                     'turnovers', 'steals', 'blocks', 'fouls', 'assists_first_half']


# FUNCTION TO CHECK FOR DUPLICATES
def check_feature_duplicates(feature_list):
    """Check for duplicate features and print them"""
    seen = set()
    duplicates = set()
    for item in feature_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    if duplicates:
        print(f"WARNING: Duplicate features found: {duplicates}")
        return list(seen)  # Return deduplicated list
    return feature_list


# Target-specific feature optimization with duplicate checking
target_configs = {
    'points': {
        'features': check_feature_duplicates(
            base_features + rolling_avg_features + rolling_std_features[:3] +
            trend_features + opponent_features + current_efficiency_features +
            ['fga_2pt', 'fga_3pt', 'ftm']
        ),
        'params': {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 10,
                   'num_leaves': 100, 'min_child_samples': 20, 'subsample': 0.8}
    },
    'assists': {
        'features': check_feature_duplicates(
            base_features + rolling_avg_features + rolling_std_features[:4] +
            trend_features + opponent_features + current_efficiency_features +
            ['assists_first_half', 'turnovers', 'steals']
        ),
        'params': {'n_estimators': 600, 'learning_rate': 0.025, 'max_depth': 8,
                   'num_leaves': 80, 'min_child_samples': 15, 'subsample': 0.85}
    },
    'rebounds': {
        'features': check_feature_duplicates(
            base_features + rolling_avg_features + rolling_std_features[:4] +
            trend_features + opponent_features + current_efficiency_features +
            ['off_rebounds', 'def_rebounds', 'blocks', 'fouls']
        ),
        'params': {'n_estimators': 450, 'learning_rate': 0.035, 'max_depth': 9,
                   'num_leaves': 90, 'min_child_samples': 18, 'subsample': 0.8}
    }
}

# Enhanced training with feature selection and validation
results = {}

for target, config in target_configs.items():
    print(f"\n🚀 Training optimized model for {target.upper()}")

    # Prepare data with available features only
    available_features = [col for col in config['features'] if col in player_stats.columns]
    df = player_stats.dropna(subset=available_features + [target]).copy()

    # Filter out players with very few games for stability
    df = df[df['career_game_num'] >= 3]

    print(f"   📊 Using {len(available_features)} features on {len(df):,} samples")

    X = df[available_features]
    y = df[target]

    # Fill any remaining NAs
    X = X.fillna(0)
    y = y.fillna(0)

    # Final check for duplicate columns
    if len(X.columns) != len(set(X.columns)):
        duplicate_cols = [col for col in X.columns if list(X.columns).count(col) > 1]
        print(f"   ⚠️ Removing duplicate columns: {set(duplicate_cols)}")
        X = X.loc[:, ~X.columns.duplicated()]

    # Time series cross-validation
    tss = TimeSeriesSplit(n_splits=4)
    mae_scores = []
    r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model with optimized parameters
        model = lgb.LGBMRegressor(
            random_state=42,
            verbose=-1,
            **config['params']
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # Final model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.12, random_state=42, shuffle=False
    )

    final_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **config['params'])
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    final_mae = mean_absolute_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns.tolist(),
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    results[target] = {
        'model': final_model,
        'mae': final_mae,
        'r2': final_r2,
        'rmse': final_rmse,
        'cv_mae_mean': np.mean(mae_scores),
        'cv_r2_mean': np.mean(r2_scores),
        'feature_importance': feature_importance,
        'features': X.columns.tolist()
    }

    print(f"   ✅ Final Results:")
    print(f"      Test R²: {final_r2:.4f} | MAE: {final_mae:.2f} | RMSE: {final_rmse:.2f}")
    print(f"      CV R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"      CV MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
    print(f"      Top 5 Features: {', '.join(feature_importance.head(5)['feature'].tolist())}")

print(f"\n🎯 Model Training Complete!")
print(f"📈 Dataset size: {len(player_stats):,} player-games")

# Summary comparison
print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
for target, result in results.items():
    print(f"   {target.upper():<8} | R²: {result['r2']:.4f} | MAE: {result['mae']:.2f} | RMSE: {result['rmse']:.2f}")

# Feature importance insights
print(f"\n🔍 KEY INSIGHTS:")
for target, result in results.items():
    top_feature = result['feature_importance'].iloc[0]
    print(
        f"   {target.upper()}: Most important feature is '{top_feature['feature']}' (importance: {top_feature['importance']:.0f})")

# Detect potential improvements
print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
for target, result in results.items():
    if result['r2'] < 0.7:
        print(f"   {target.upper()}: R² < 0.7, consider adding more contextual features or player-specific adjustments")
    elif result['r2'] < 0.85:
        print(f"   {target.upper()}: Good performance, may benefit from matchup-specific features")
    else:
        print(f"   {target.upper()}: Excellent performance!")

# Optional: Save models and feature importance
print(f"\n💾 Models ready for deployment. Uncomment save section if needed.")

# === OPTIONAL: SAVE MODELS AND ANALYSIS ===
# Uncomment to save models and analysis
import joblib
for target, result in results.items():
    joblib.dump(result['model'], f'models/nba_{target}_model_v2.pkl')
    result['feature_importance'].to_csv(f'models/nba_{target}_feature_importance_v2.csv', index=False)

# Save feature definitions for future reference
feature_definitions = {
    'base_features': base_features,
    'rolling_avg_features': rolling_avg_features,
    'rolling_std_features': rolling_std_features,
    'trend_features': trend_features,
    'opponent_features': opponent_features,
    'efficiency_features': current_efficiency_features
}

import json
with open('feature_definitions.json', 'w') as f:
    json.dump(feature_definitions, f, indent=2)

player_stats.to_csv("data/latest_player_stats.csv", index=False)
