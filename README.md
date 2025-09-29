# NBA-Sports-Data-ML-Model

Work in progress model that is supposed to be used to predict future player performance. Using lightgbm and sklearn. Currently model has issue due to overfitting, need to fix via bigger data set to learn on.

Current Output Of Model

🚀 Training optimized model for POINTS
   📊 Using 81 features on 90,841 samples
   ✅ Final Results:
      Test R²: 0.9998 | MAE: 0.04 | RMSE: 0.16
      CV R²: 0.9995 ± 0.0004
      CV MAE: 0.07 ± 0.03
      Top 5 Features: fga_2pt, fga_3pt, minutes, usage_proxy, true_shooting_pct

🚀 Training optimized model for ASSISTS
   📊 Using 82 features on 90,841 samples
   ✅ Final Results:
      Test R²: 0.9970 | MAE: 0.03 | RMSE: 0.67
      CV R²: 0.9789 ± 0.0325
      CV MAE: 0.13 ± 0.17
      Top 5 Features: assist_rate, assist_to_turnover, turnovers, assists_avg_10g, assists_first_half

🚀 Training optimized model for REBOUNDS
   📊 Using 83 features on 90,841 samples
   ✅ Final Results:
      Test R²: 0.9996 | MAE: 0.02 | RMSE: 0.05
      CV R²: 0.9992 ± 0.0005
      CV MAE: 0.02 ± 0.01
      Top 5 Features: rebound_rate, minutes, rebounds_avg_7g, rebounds_avg_3g, rebounds_trend_5g

🎯 Model Training Complete!
📈 Dataset size: 93,305 player-games

📊 FINAL PERFORMANCE SUMMARY:
   POINTS   | R²: 0.9998 | MAE: 0.04 | RMSE: 0.16
   ASSISTS  | R²: 0.9970 | MAE: 0.03 | RMSE: 0.67
   REBOUNDS | R²: 0.9996 | MAE: 0.02 | RMSE: 0.05

🔍 KEY INSIGHTS:
   POINTS: Most important feature is 'fga_2pt' (importance: 9512)
   ASSISTS: Most important feature is 'assist_rate' (importance: 3465)
   REBOUNDS: Most important feature is 'rebound_rate' (importance: 9903)

💡 IMPROVEMENT SUGGESTIONS:
   POINTS: Excellent performance!
   ASSISTS: Excellent performance!
   REBOUNDS: Excellent performance!

💾 Models ready for deployment. Uncomment save section if needed.


