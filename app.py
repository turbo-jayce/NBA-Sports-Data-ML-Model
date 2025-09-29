import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Load Models ===
points_model = joblib.load('models/nba_points_model_v2.pkl')
assists_model = joblib.load('models/nba_assists_model_v2.pkl')
rebounds_model = joblib.load('models/nba_rebounds_model_v2.pkl')

# === Load Latest Player Data ===
player_stats = pd.read_csv("data/latest_player_stats.csv")

# Get most recent game per player
latest_stats = player_stats.sort_values("game_id").groupby("playerNameI").tail(1)
available_players = sorted(latest_stats["playerNameI"].unique())

# === Streamlit GUI ===
st.set_page_config(page_title="NBA Lineup Predictor", layout="wide")
st.title("🏀 NBA Lineup Predictor")
st.markdown("Select 5 players to predict their next-game **points**, **assists**, and **rebounds**.")

# === Select Starting 5 ===
selected_players = st.multiselect(
    "Search & select your starting five:",
    options=available_players,
    max_selections=5
)

# === Prediction Button ===
if st.button("Predict") and len(selected_players) == 5:
    st.subheader("🔮 Predictions")

    results = []

    for player in selected_players:
        player_row = latest_stats[latest_stats['playerNameI'] == player].copy()
        if player_row.empty:
            continue


        # Get model features
        def extract_features(row, model_features):
            return row[model_features].fillna(0).values.reshape(1, -1)


        # Predict for each stat
        features_points = extract_features(player_row, points_model.feature_name_)
        features_assists = extract_features(player_row, assists_model.feature_name_)
        features_rebounds = extract_features(player_row, rebounds_model.feature_name_)

        pred_points = points_model.predict(features_points)[0]
        pred_assists = assists_model.predict(features_assists)[0]
        pred_rebounds = rebounds_model.predict(features_rebounds)[0]

        results.append({
            'Player': player,
            'Predicted Points': round(pred_points, 1),
            'Predicted Assists': round(pred_assists, 1),
            'Predicted Rebounds': round(pred_rebounds, 1)
        })

    # Show predictions in table
    result_df = pd.DataFrame(results)
    st.table(result_df)

elif len(selected_players) != 5:
    st.warning("Please select exactly **5 players** to run the prediction.")
