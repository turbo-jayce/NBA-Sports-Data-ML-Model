import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pickle

# === Load latest player stats ===
player_stats = pd.read_csv("data/latest_player_stats.csv")

# Define feature lists (must match training features)
points_features = ['age', 'minutes', 'fieldGoalsAttempted', 'threePointersAttempted']
assists_features = ['age', 'minutes', 'passes', 'touches']
rebounds_features = ['age', 'height', 'minutes', 'contestedRebounds']

# ✅ Correct: structured model+features dictionary
results = {
    'points': {
        'model': pickle.load(open("models/nba_points_model_v2.pkl", "rb")),
        'features': points_features
    },
    'assists': {
        'model': pickle.load(open("models/nba_assists_model_v2.pkl", "rb")),
        'features': assists_features
    },
    'rebounds': {
        'model': pickle.load(open("models/nba_rebounds_model_v2.pkl", "rb")),
        'features': rebounds_features
    }
}




# === Load player stats ===
try:
    player_stats = pd.read_csv("data/latest_player_stats.csv")
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'data/latest_player_stats.csv'. Make sure the file exists.")

# === Load trained models ===
results = {}

model_paths = {
    "points": "models/nba_points_model_v2.pkl",
    "assists": "models/nba_assists_model_v2.pkl",
    "rebounds": "models/nba_rebounds_model_v2.pkl"
}

for stat, path in model_paths.items():
    try:
        with open(path, "rb") as f:
            results[stat] = pickle.load(f)
    except FileNotFoundError:
        print(f"⚠️ Could not find model file: {path}")
        results[stat] = {}

# === Get unique players ===
available_players = sorted(player_stats['playerNameI'].dropna().unique().tolist())

# === GUI App ===
class NBAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏀 NBA Stat Predictor")
        self.root.geometry("600x400")

        # Instructions
        label = tk.Label(root, text="Select 5 Players for Prediction", font=('Helvetica', 14))
        label.pack(pady=10)

        # Dropdown frame
        self.dropdown_frame = tk.Frame(root)
        self.dropdown_frame.pack(pady=10)

        self.player_vars = []
        for i in range(5):
            var = tk.StringVar()
            combo = ttk.Combobox(self.dropdown_frame, textvariable=var, values=available_players, width=40)
            combo.grid(row=i, column=0, padx=5, pady=5)
            self.player_vars.append(var)

        # Predict button
        predict_btn = tk.Button(root, text="📊 Predict Stats", command=self.predict)
        predict_btn.pack(pady=20)

        # Output area
        self.output_text = tk.Text(root, height=10, width=70)
        self.output_text.pack(pady=10)

    def predict(self):
        selected_players = [var.get() for var in self.player_vars if var.get()]
        self.output_text.delete(1.0, tk.END)

        if len(selected_players) != 5:
            messagebox.showwarning("Input Error", "Please select exactly 5 players.")
            return

        results_summary = []

        for player in selected_players:
            latest_row = player_stats[player_stats['playerNameI'] == player].sort_values(
                'career_game_num', ascending=False
            ).head(1)

            if latest_row.empty:
                results_summary.append(f"{player}: ❌ No data available\n")
                continue

            output = [f"\n🔹 {player}"]
            for stat in ['points', 'assists', 'rebounds']:
                model = results.get(stat, {}).get('model', None)
                features = results.get(stat, {}).get('features', [])
                if model and all(f in latest_row.columns for f in features):
                    X = latest_row[features].fillna(0)
                    pred = model.predict(X)[0]
                    output.append(f"  {stat.title()}: {pred:.1f}")
                else:
                    output.append(f"  {stat.title()}: ❓ (model or features missing)")
            results_summary.append("\n".join(output))

        self.output_text.insert(tk.END, "\n".join(results_summary))


# === Run the GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = NBAApp(root)
    root.mainloop()
