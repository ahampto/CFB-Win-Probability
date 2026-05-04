# 🏈 College Football Win Probability Engine

**A production-grade, play-by-play win probability model for FBS college football — built on 1.72 million plays, 8 seasons of data, a 24-feature engineered dataset, dual machine learning models, and a live Streamlit application.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.x-E25A1C?logo=apachespark&logoColor=white)](https://spark.apache.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-brightgreen)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Key Results](#key-results)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Feature Engineering](#feature-engineering)
- [Statistical Validation](#statistical-validation)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Streamlit Application](#streamlit-application)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Technical Stack](#technical-stack)
- [Author](#author)

---

## Overview

Win probability estimation is one of the most analytically demanding challenges in sports data science. Unlike binary outcome prediction, a win probability model must assign a **continuous, well-calibrated probability to every play of every game** — reflecting the live game state as it evolves from kickoff to the final whistle.

This project addresses that challenge end-to-end:

- **1,718,250** raw play events collected via API across 8 FBS seasons
- A **4-stage preprocessing pipeline** handling clock normalization, field position correction, timeout imputation, and perspective alignment
- A **24-feature production dataset** distilled from 362 raw columns through domain-informed engineering
- **Two production models** — Regularized Logistic Regression and XGBoost — achieving **ROC-AUC of 0.924+** and **83%+ accuracy**
- **Statistical validation** via statsmodels Logit (Pseudo R² = 0.501, inter_score_x_time z = 129.3)
- A **Streamlit web application** delivering broadcast-quality, real-time win probability charts for any game in the dataset

The central finding: **domain-informed feature engineering closed the performance gap between a linear model and a 500-tree gradient boosted ensemble.** Feature quality beats model complexity.

---

## Live Demo

> 🚀 *Streamlit app link — add your deployed URL here*

The application allows users to:
- Select any game from the 8-season dataset
- Step through plays one at a time or scrub to any moment
- Watch win probability update in real time after each snap
- Automatically identify the **Play of the Game** (largest win probability swing)
- Switch between XGBoost and Logistic Regression models and compare metrics live

![App Screenshot](assets/app_screenshot.png)

---

## Key Results

| Metric | Logistic Regression | XGBoost |
|--------|-------------------|---------|
| **ROC-AUC** | 0.9245 | 0.9263 |
| **Accuracy** | 83.34% | 83.51% |
| **Brier Score** | 0.1121 | 0.1113 |
| **Log Loss** | 0.3437 | 0.3488 |

> Evaluated on ~318,000 held-out plays (GroupShuffleSplit by game_id — the model never trains and evaluates on plays from the same game)

**Statsmodels Logit Validation (1,590,878 training plays):**
- Pseudo R² = **0.501**
- `inter_score_x_time`: z = **129.3**, p = **0.000** ← most significant predictor
- `pos_score_diff`: z = 71.6, p < 0.001
- `inter_spread_x_time`: z = -85.4, p < 0.001
- Engineered features represent **54% of columns** but carry **~60% of total predictive signal** (efficiency ratio: **1.38×**)

---

## Project Architecture

```
Raw API Data (362 cols, 1.72M rows)
        │
        ▼
┌─────────────────────────────┐
│   Stage 1: Temporal Fix     │  adj_TimeSecsRem — continuous 3600→0
│   Stage 2: Field Position   │  Red zone distance cap + timeout ffill
│   Stage 3: Perspective Align│  pos_team_elo, def_pos_team_elo, pos_team_spread
│   Stage 4: PySpark Pipeline │  Rolling features, lag(1), skinny-down to 24 cols
└─────────────────────────────┘
        │
        ▼
  24-Feature Production Dataset
        │
        ├──────────────────────────┐
        ▼                          ▼
Logistic Regression           XGBoost GBDT
(L2, C=0.1, SAGA)         (depth=3, lr=0.01, n=500)
        │                          │
        └──────────┬───────────────┘
                   ▼
         Streamlit Application
    (perspective flip + WP swing + caching)
```

---

## Dataset

- **Source:** [collegefootballdata.com](https://collegefootballdata.com) API via the R package [`cfbfastR`](https://cfbfastr.sportsdataverse.org/)
- **Seasons:** 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025 *(2020 excluded — COVID-19 pandemic)*
- **Scale:** 1,718,250 rows × 362 features
- **Coverage:** All FBS programs — regular season, conference championships, bowl games, postseason
- **Target variable:** `pos_team_win` — binary win indicator for the team currently in possession

### Leaky Features Removed

Of the 362 raw columns, **114 were removed** as post-snap data leakage (yards gained, play EPA, play outcome, end-of-play field position). The model only sees information available **before the ball is snapped**.

---

## Data Cleaning & Preprocessing

### Stage 1 — Clock Normalization
Raw data resets the game clock at halftime (1,800s → 0 per half). A model trained on this cannot distinguish a 2-minute drill before halftime from a 2-minute drill in the fourth quarter.

**Fix:** Created `adj_TimeSecsRem` — a continuous countdown from 3,600 seconds at kickoff to 0 at the final whistle. Periods 3 & 4 add remaining time to the 1,800 seconds of the completed first half.

### Stage 2 — Field Position & Timeout Correction
**Red Zone Distance Anomaly:** Goal-to-go plays frequently listed `distance = 10` even when the team was on the 2-yard line — a logical impossibility.
- **Fix:** When `Goal_To_Go == True`, cap `distance = yards_to_goal`

**Sparse Timeout Data:** Timeouts reset to 3 each half and can only decrease. Mean imputation is logically incorrect for monotonically non-increasing resources.
- **Fix:** Forward-fill (`ffill`) grouped by `game_id` and `half`, resetting at the start of Q3

### Stage 3 — Possession-Perspective Alignment
Raw features are tied to the home team, causing the model to learn team-specific historical patterns rather than transferable football dynamics.

**Fix:** All team features reframed relative to the possessing team:
- `pos_team_elo` / `def_pos_team_elo` — ELO for ball-carrier vs. defender
- `pos_team_spread` — Vegas spread from the possessing team's perspective

> The model learns *"the value of being a 7-point favorite"* — not *"Alabama usually wins."*

### Stage 4 — PySpark Distributed Pipeline
Standard Pandas operations caused repeated OOM failures on 1.6M rows with window functions.

**Fix:** PySpark for all heavy transformations:
1. Distributed chronological sort (game → period → clock → play number)
2. Grouped window functions for rolling EPA and yards (`lag(1)` applied)
3. Forward-fill timeout imputation partitioned by `game_id` and `half`
4. **Skinny-down:** 362 columns → 24-feature dense matrix
5. `.toPandas()` handoff → scikit-learn modeling pipeline

### Additional Noise Fixes
| Issue | Fix |
|-------|-----|
| ~0.5% duplicate/skipped play sequences | Secondary sort by period + TimeSecsRem |
| Rows with 'BREAK'/'null' play text | Re-classified or dropped if `play_type` missing |
| Score differential recorded after the play | Recalculated `pos_score_diff` from start-of-play score |
| Overtime periods (Period 5+) | Filtered entirely — different rules corrupt time-urgency weights |

---

## Feature Engineering

The 24 production features were **actively constructed**, not selected from the raw data.

### Category 1 — Base Game State (11 features)
| Feature | Description |
|---------|-------------|
| `down` | Current down (1–4) |
| `distance` | Yards to first down (capped in red zone) |
| `yards_to_goal` | Yards to the end zone |
| `TimeSecsRem` | Adjusted continuous time remaining (3600→0) |
| `pos_score_diff` | Possession team score minus opponent score |
| `pos_team_timeouts_rem_before` | Timeouts for possession team |
| `def_pos_team_timeouts_rem_before` | Timeouts for defending team |
| `is_home_team` | Whether possession team is home (binary) |
| `pos_team_elo` | ELO rating of the possessing team |
| `def_pos_team_elo` | ELO rating of the defending team |
| `pos_team_spread` | Vegas spread aligned to possession team |

### Category 2 — Drive Fatigue & Rolling Momentum (5 features)
| Feature | Description |
|---------|-------------|
| `drive_yards_so_far` | Cumulative yards on current drive |
| `drive_success_rate` | % of successful plays on current drive |
| `drive_time_elapsed` | Time elapsed on current drive (fatigue proxy) |
| `pos_rolling_EPA_last_5` | Rolling EPA over previous 5 plays |
| `pos_rolling_yards_last_5` | Rolling yards gained over previous 5 plays |

> All momentum features computed with `lag(1)` — the model never sees the outcome of the play it is predicting.

### Category 3 — Interaction Terms (5 features)
| Feature | Formula | Intuition |
|---------|---------|-----------|
| `inter_score_x_time` | `pos_score_diff × time_elapsed_fraction` | Being up 7 in Q4 >> being up 7 in Q1 |
| `inter_spread_x_time` | `pos_team_spread × time_remaining_fraction` | Pre-game expectations decay as live reality takes over |
| `inter_down_x_distance` | `down × distance` | 3rd & 10 is fundamentally different from 1st & 10 |
| `inter_elo_diff` | `pos_team_elo − def_pos_team_elo` | Net team strength from the ball-carrier's perspective |
| `inter_timeout_diff` | `pos_timeouts − def_timeouts` | Late-game clock management leverage |

### Category 4 — Polynomial Terms (3 features)
| Feature | Intuition |
|---------|-----------|
| `poly_score_diff_sq` | A 14-pt lead is **exponentially** safer than a 7-pt lead, not 2× safer |
| `poly_TimeSecsRem_sq` | Accelerating urgency in final minutes — 3 min vs 1 min >> 10 min vs 8 min |
| `poly_yards_to_goal_sq` | Non-linear red zone leverage — scoring probability curves sharply inside the 10 |

### Feature Signal Validation

![Feature Correlation Chart](assets/feature_correlation.png)

Engineered features (orange) dominate the correlation ranking despite being *derived* from raw features (blue):
- **54%** of feature columns are engineered
- **~60%** of total predictive power comes from engineered features
- **Efficiency ratio: 1.38×** signal per column vs. raw features

---

## Statistical Validation

A full `statsmodels` Logit regression was run on 1,590,878 training plays to validate feature significance beyond predictive accuracy alone.

![p-value Table](assets/pvalue_table.png)

### Key findings

**Highly significant (p < 0.001):**
- `inter_score_x_time` — z = **129.3** ← single most significant predictor
- `pos_score_diff` — z = 71.6
- `inter_spread_x_time` — z = -85.4
- `pos_team_spread` — z = -69.3
- `poly_score_diff_sq`, `poly_yards_to_goal_sq`

**NaN p-values — ELO features:**
`inter_elo_diff = pos_team_elo − def_pos_team_elo` exactly → perfect multicollinearity → singular matrix. Resolved by L2 regularization in production models.

**p = 1.000, std err ≈ 33,000 — Timeout features:**
Classic multicollinearity explosion: given both individual timeout counts, `inter_timeout_diff` is fully determined. Stabilized by L2 penalty.

**The Redundancy Paradox:**
Non-significant features like `poly_TimeSecsRem_sq` and `inter_down_x_distance` are rendered *statistically redundant* by the dominant interaction terms — not because they lack football signal, but because that signal is already captured more efficiently by the engineered features. Removing them would degrade accuracy even as their p-values improved.

---

## Model Architecture

### Logistic Regression
- **Penalty:** L2 (Ridge), `C = 0.1`
- **Solver:** SAGA — designed for datasets exceeding 1M rows
- **Tuning:** GridSearchCV, 3-fold cross-validation on grouped training set
- **Key design:** Interaction and polynomial features pre-encode the non-linear relationships that logistic regression cannot discover independently. The model learns weights, not physics.
- **Strength:** Fully interpretable coefficients, smooth probability curves, stable under multicollinearity via regularization

### XGBoost
- **Method:** `tree_method='hist'` — histogram-based for fast training at scale
- **Hyperparameters:** 500 estimators, `learning_rate=0.01`, `max_depth=3`, subsampling 0.8
- **Key design:** `max_depth=3` is deliberate — shallow trees prevent game-specific overfitting and produce the smooth, monotonic WP curves required for broadcast-quality output
- **Tuning:** RandomizedSearchCV, 3-fold cross-validation on grouped training set
- **Feature importance (gain):** `poly_score_diff_sq` = 41.9%, `pos_score_diff` = 31.8%, `inter_spread_x_time` = 8.5%, `inter_score_x_time` = 5.1%

### Why Both Models Converge

When a linear model matches a gradient boosted ensemble, it is because feature engineering has **linearized the non-linear structure of the problem**. By constructing `inter_score_x_time` and `poly_score_diff_sq` explicitly, the pipeline gave the logistic regression pre-computed signals encoding the physics of football. The model doesn't need to discover these relationships — they're embedded in the features.

> **Feature quality beats model complexity.**

![Calibration and Feature Importance](assets/calibration_dashboard.png)

---

## Training Pipeline

```
1. GroupShuffleSplit (game_id) → 80% train / 20% test
        ↓
2. StandardScaler fit on training set only (required for L2 regularization)
        ↓
3. Cross-validation (GridSearchCV / RandomizedSearchCV)
   - LogReg: search over C values
   - XGBoost: search over depth, learning rate, n_estimators, subsampling
        ↓
4. Best hyperparameters selected on validation log loss
        ↓
5. Full retrain on all 8 seasons (train + test recombined)
        ↓
6. Export: model.pkl + scaler.pkl → Streamlit app
```

---

## Streamlit Application

### Architecture

**Perspective-Flip Logic (the continuous graph secret)**

The model predicts possession-team win probability. On a turnover, the raw output would jump discontinuously (e.g., 80% → 20%). The app resolves this by anchoring all output to the home team:

```python
if pos_team == home_team:
    display_wp = model_wp
else:
    display_wp = 1 - model_wp
```

Turnovers appear as smooth directional movements — matching the ESPN broadcast format viewers expect.

**Automatic Play of the Game**

```python
wp_delta = wp_series.diff()  # delta between consecutive plays
play_of_game = wp_delta[winning_team_perspective].idxmax()
```

The single most pivotal moment is auto-detected from the win probability sequence with no manual analysis required.

**Optimized Inference**

```python
@st.cache_resource
def load_models():
    model = joblib.load("xgb_production.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
```

Both models and the fitted scaler are cached at startup. Full 200-play game inference and chart render completes in **< 1.5 seconds**.

### App Features
- 🏈 Select any game from the 8-season dataset
- ▶️ Play/Pause/Scrub controls for replay
- 📊 Live score, quarter, time, possession, down & distance, field position
- 📈 Continuous home-team win probability chart
- ⭐ Auto-highlighted Play of the Game
- 🔀 Switch between XGBoost and Logistic Regression with live metric comparison

---

## Repository Structure

```
cfb-win-probability/
│
├── data/
│   ├── raw/                    # Raw API pulls (not included — see Data Collection)
│   └── processed/              # 24-feature production dataset
│
├── notebooks/
│   ├── 01_data_collection.R    # cfbfastR API collection
│   ├── 02_cleaning_pipeline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_statistical_validation.ipynb
│   ├── 05_logistic_regression.ipynb
│   └── 06_xgboost.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── clock_normalization.py
│   │   ├── field_position.py
│   │   ├── timeout_imputation.py
│   │   └── perspective_alignment.py
│   ├── features/
│   │   ├── base_features.py
│   │   ├── momentum_features.py
│   │   ├── interaction_terms.py
│   │   └── polynomial_terms.py
│   ├── models/
│   │   ├── logistic_regression.py
│   │   └── xgboost_model.py
│   └── pipeline/
│       └── spark_pipeline.py
│
├── app/
│   ├── streamlit_app.py        # Main application
│   ├── model_inference.py      # Perspective-flip + WP swing logic
│   └── assets/
│
├── models/
│   ├── xgb_production.pkl      # Serialized XGBoost production model
│   ├── logreg_production.pkl   # Serialized Logistic Regression model
│   └── scaler.pkl              # Fitted StandardScaler
│
├── assets/                     # README images
│   ├── app_screenshot.png
│   ├── feature_correlation.png
│   ├── calibration_dashboard.png
│   └── pvalue_table.png
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Java 8+ (for PySpark)
- R + cfbfastR (for data collection only)

### Clone & Install

```bash
git clone https://github.com/alvinhampton/cfb-win-probability.git
cd cfb-win-probability

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=1.7
pyspark>=3.4
statsmodels>=0.14
streamlit>=1.28
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
```

---

## Usage

### Run the Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

### Run the Feature Engineering Pipeline

```python
from src.pipeline.spark_pipeline import build_production_dataset

# Build 24-feature matrix from raw play-by-play data
df = build_production_dataset(
    raw_data_path="data/raw/pbp_2017_2025.parquet",
    output_path="data/processed/production_features.parquet"
)
```

### Run Model Training

```python
from src.models.xgboost_model import train_xgboost
from src.models.logistic_regression import train_logreg

# Train and evaluate both models
xgb_model, xgb_metrics = train_xgboost("data/processed/production_features.parquet")
log_model, log_metrics = train_logreg("data/processed/production_features.parquet")

print(xgb_metrics)
# {'roc_auc': 0.9263, 'accuracy': 0.8351, 'brier': 0.1113, 'log_loss': 0.3488}
```

### Get Win Probability for a Single Play State

```python
import joblib
import pandas as pd

model = joblib.load("models/xgb_production.pkl")
scaler = joblib.load("models/scaler.pkl")

play = pd.DataFrame([{
    "down": 3, "distance": 8, "yards_to_goal": 45,
    "TimeSecsRem": 420, "pos_score_diff": -3,
    "pos_team_timeouts_rem_before": 2,
    "def_pos_team_timeouts_rem_before": 1,
    "is_home_team": 1,
    "pos_team_elo": 1550, "def_pos_team_elo": 1480,
    "pos_team_spread": 3.5,
    # ... interaction and polynomial terms
}])

win_prob = model.predict_proba(scaler.transform(play))[0][1]
print(f"Win probability: {win_prob:.1%}")  # e.g., Win probability: 38.4%
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Data Collection | R, `cfbfastR`, collegefootballdata.com API |
| Data Processing | PySpark 3.x, Pandas |
| Feature Engineering | NumPy, custom PySpark window functions |
| Statistical Validation | statsmodels |
| Modeling | scikit-learn, XGBoost |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV |
| Model Serialization | joblib |
| Web Application | Streamlit |
| Visualization | Matplotlib, Seaborn, Plotly |
| Environment | Python 3.10+, Java 8+ |

---

## Author

**Alvin Hampton**

*Data Science | Machine Learning | Sports Analytics*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/alvinhampton)

---

*Built for INFO 516 — Spring 2026*
