import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# APP STARTUP 

@st.cache_resource
def load_models():
    xgb_model = joblib.load('Models/cfb_xgboost_prod.pkl')
    log_model = joblib.load('Models/cfb_logreg_prod.pkl')
    scaler = joblib.load('Models/cfb_prod_scaler.pkl')
    return xgb_model, log_model, scaler

@st.cache_data
def load_data():
    ml_df = pd.read_parquet('Data/cfb_ml_features_prod.parquet')
    raw_df = pd.read_parquet('Data/cfb_raw_context_prod.parquet')
    return ml_df, raw_df


#CORE LOGIC

@st.cache_data
def generate_win_probability_dashboard(game_id, ml_df, context_df, _model, _scaler, model_type):
    game_ml = ml_df[ml_df['game_id'] == game_id].sort_values(by=['TimeSecsRem', 'down'], ascending=[False, True]).copy()
    game_raw = context_df[context_df['game_id'] == game_id].copy()
    
    X_predict = game_ml.drop(columns=['pos_team_win', 'game_id'])
    
    if model_type == 'Logistic Regression':
        X_predict_scaled = _scaler.transform(X_predict)
        game_ml['wp'] = _model.predict_proba(X_predict_scaled)[:, 1]
    else:
        game_ml['wp'] = _model.predict_proba(X_predict)[:, 1]
    
    game_ml['home_wp'] = np.where(game_ml['is_home_team'] == 1, game_ml['wp'], 1 - game_ml['wp'])
    game_ml['home_wp_next_play'] = game_ml['home_wp'].shift(-1)
    game_ml['home_wp_swing'] = game_ml['home_wp_next_play'] - game_ml['home_wp']
    game_ml['abs_wp_swing'] = abs(game_ml['home_wp_swing'])
    
    final_ui_df = pd.merge(
        game_ml, game_raw,
        left_on=['game_id', 'TimeSecsRem', 'down'],
        right_on=['game_id', 'adj_TimeSecsRem', 'down'],
        how='left', suffixes=('', '_raw')
    ).drop_duplicates(subset=['TimeSecsRem', 'down'])
    
    final_ui_df['home_score'] = np.where(final_ui_df['pos_team'] == final_ui_df['home_team'], final_ui_df['pos_team_score'], final_ui_df['def_pos_team_score'])
    final_ui_df['away_score'] = np.where(final_ui_df['pos_team'] == final_ui_df['away_team'], final_ui_df['pos_team_score'], final_ui_df['def_pos_team_score'])
    
    final_ui_df['seconds_elapsed'] = 3600 - final_ui_df['TimeSecsRem']
    final_ui_df['wp_percent'] = (final_ui_df['home_wp'] * 100).round(1).astype(str) + '%'
    
    def format_quarter_time(secs):
        if secs <= 0: return "0:00"
        q_secs = secs % 900
        if q_secs == 0: q_secs = 900 
        return f"{int(q_secs//60)}:{int(q_secs%60):02d}"
        
    final_ui_df['time_str'] = final_ui_df['TimeSecsRem'].apply(format_quarter_time)
    
    return final_ui_df.sort_values(by=['seconds_elapsed']).reset_index(drop=True)


# FOOTBALL FIELD DRAWING FUNCTION

def draw_football_field(row):
    fig = go.Figure()
    
    # Green Field & Endzones
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=53.3, fillcolor="#2E8B57", line_color="white")
    fig.add_shape(type="rect", x0=-10, y0=0, x1=0, y1=53.3, fillcolor="#1E5128", line_color="white")
    fig.add_shape(type="rect", x0=100, y0=0, x1=110, y1=53.3, fillcolor="#1E5128", line_color="white")
    
    # Yard Lines and Numbers
    yard_numbers = [10, 20, 30, 40, 50, 40, 30, 20, 10]
    for i, x in enumerate(range(10, 100, 10)):
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=53.3, line=dict(color="white", width=2))
        fig.add_annotation(x=x, y=5, text=str(yard_numbers[i]), showarrow=False, font=dict(color='white', size=16))
        fig.add_annotation(x=x, y=48.3, text=str(yard_numbers[i]), showarrow=False, font=dict(color='white', size=16))
    
    # Directional & Dynamic Football Logic
    is_home_offense = (row['pos_team'] == row['home_team'])
    ball_x = 100 - row['yards_to_goal'] if is_home_offense else row['yards_to_goal']
    play_type = str(row['play_type']).lower()
    
    if 'kickoff' in play_type or 'punt' in play_type:
        ball_icon = "🦵🏈 〰️" if is_home_offense else "〰️ 🏈🦵"
    elif 'pass' in play_type:
        ball_icon = "🏈 ⇢" if is_home_offense else "⇠ 🏈"
    elif 'rush' in play_type or 'run' in play_type:
        # Runner with explicit directional arrows
        ball_icon = "🏃🏈 ⇢" if is_home_offense else "⇠ 🏈🏃"
    else:
        ball_icon = "🏈⇢" if is_home_offense else "⇠🏈"
        
    # Draw Ball
    fig.add_trace(go.Scatter(
        x=[ball_x], y=[26.6], mode="markers", 
        marker=dict(color="saddlebrown", size=18, symbol="diamond", line=dict(color="white", width=2)),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_annotation(x=ball_x, y=32, text=ball_icon, showarrow=False, font=dict(size=25))
    
    # --- TOUCHDOWN BANNER LOGIC ---
    play_text_lower = str(row['play_text']).lower()
    play_type_lower = str(row['play_type']).lower()
    if 'touchdown' in play_text_lower or ' td ' in play_text_lower or 'touchdown' in play_type_lower:
        fig.add_annotation(
            x=50, y=26.6, text="TOUCHDOWN!", showarrow=False, 
            font=dict(family="Arial Black", size=50, color="gold"),
            opacity=0.9, bordercolor="black", borderwidth=2, borderpad=4, bgcolor="rgba(0,0,0,0.6)"
        )
    
    fig.update_xaxes(range=[-15, 115], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-5, 60], showgrid=False, zeroline=False, visible=False)
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig


# STREAMLIT FRONTEND

st.set_page_config(page_title="CFB Win Probability", layout="wide")
st.title("🏈 CFB Win Probability & Game Replay")

# State Management
if 'play_idx' not in st.session_state:
    st.session_state.play_idx = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_game_id' not in st.session_state:
    st.session_state.current_game_id = None

xgb_model, log_model, scaler = load_models()
ml_data, raw_data = load_data()

#  SIDEBAR 
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio("Select AI Model:", ["XGBoost", "Logistic Regression"])

# Playback Speed (1 is slow, 5 is fast)
speed_level = st.sidebar.slider("Playback Speed (1=Slow, 5=Fast):", min_value=1, max_value=5, value=3, step=1)
speed_mapping = {1: 1.5, 2: 1.0, 3: 0.5, 4: 0.25, 5: 0.1}
play_speed = speed_mapping[speed_level]

# Sound Toggle
enable_sound = st.sidebar.checkbox("🔊 Enable Sound Effects", value=False)

# MODEL METRICS DISPLAY 
st.sidebar.markdown("### 📊 Model Comparison")

#  Model Comparison
st.sidebar.markdown(
    """
    | Metric | LogReg | XGBoost |
    |---|---|---|
    | **Accuracy** | 83.30% | <span style="color:green; font-weight:bold;">83.33%</span> |
    | **ROC-AUC** | 0.9242 | <span style="color:green; font-weight:bold;">0.9245</span> |
    | **Brier Score** | <span style="color:green; font-weight:bold;">0.1123</span> | 0.1126 |
    | **Log Loss** | <span style="color:green; font-weight:bold;">0.3444</span> | 0.3527 |
    """, unsafe_allow_html=True
)

# Metrics Explanation 
with st.sidebar.expander("📚 What do these metrics mean?"):
    st.markdown("""
    - <span style='color:#00BFFF; font-weight:bold;'>Accuracy:</span> The basic win/loss prediction rate.<br>
    *Example: If the model predicts Alabama will win at kickoff, do they actually end up winning the game?*
    
    - <span style='color:#00BFFF; font-weight:bold;'>ROC-AUC:</span> Measures how well the model separates certain wins from certain losses (1.0 is perfect).<br>
    *Example: It successfully distinguishes a 28-point lead in the 4th quarter (high WP) from a tie game in the 1st quarter (50/50 WP).*
    
    - <span style='color:#00BFFF; font-weight:bold;'>Brier Score:</span> Measures the exactness of the probability (Lower is better).<br>
    *Example: If the model says a team down by 4 with 2 minutes left has a 20% chance to win, does that exact scenario actually result in a win exactly 2 times out of 10 over the course of a season?*
    
    - <span style='color:#00BFFF; font-weight:bold;'>Log Loss:</span> Penalizes the model heavily if it is highly confident but wrong (Lower is better).<br>
    *Example: If the model is 99% sure Ohio State will win, but they throw a pick-six and end up losing, Log Loss heavily penalizes that overconfidence.*
    """, unsafe_allow_html=True)

st.sidebar.header("🔍 Find a Game")
all_teams = sorted(list(set(raw_data['home_team'].dropna().unique()) | set(raw_data['away_team'].dropna().unique())))
selected_team = st.sidebar.selectbox("Filter by Team:", ["All Teams"] + all_teams)

if selected_team != "All Teams":
    filtered_games = raw_data[(raw_data['home_team'] == selected_team) | (raw_data['away_team'] == selected_team)]
else:
    filtered_games = raw_data

game_mapping = {row['game_id']: f"{row['away_team']} @ {row['home_team']}" for _, row in filtered_games.drop_duplicates(subset=['game_id']).iterrows()}
selected_game_id = st.sidebar.selectbox("Select Matchup:", list(game_mapping.keys()), format_func=lambda x: game_mapping[x])

if selected_game_id:
    # --- BUG FIX: Reset play index when a new game is selected ---
    if st.session_state.current_game_id != selected_game_id:
        st.session_state.current_game_id = selected_game_id
        st.session_state.play_idx = 0
        st.session_state.is_playing = False
        
    active_model = xgb_model if model_choice == "XGBoost" else log_model
    df = generate_win_probability_dashboard(selected_game_id, ml_data, raw_data, active_model, scaler, model_choice)
    
    home_team = df['home_team'].iloc[0]
    away_team = df['away_team'].iloc[0]
    st.markdown("---")
    
    base_wp_fig = go.Figure()
    base_wp_fig.add_trace(go.Scatter(x=df['seconds_elapsed'], y=df['home_wp'], mode='lines', line=dict(color='black', width=3), name="Win Prob"))
    base_wp_fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    for q in [900, 1800, 2700]:
        base_wp_fig.add_vline(x=q, line_dash="dot", line_color="gray", opacity=0.5)
    base_wp_fig.update_layout(yaxis_title=f"{home_team} WP ({model_choice})", xaxis_title="Seconds Elapsed", yaxis_range=[0, 1], xaxis_range=[0, 3600], height=400, margin=dict(l=0, r=0, t=30, b=0))
    
    with st.expander("⏱️ Jump to Specific Time / Quarter"):
        q_col, m_col, s_col, btn_col = st.columns([1, 1, 1, 2])
        target_q = q_col.selectbox("Quarter", [1, 2, 3, 4], index=0)
        target_m = m_col.number_input("Minute", min_value=0, max_value=15, value=15)
        target_s = s_col.number_input("Second", min_value=0, max_value=59, value=0)
        st.markdown("<br>", unsafe_allow_html=True)
        if btn_col.button("Jump to Time"):
            target_time_secs = ((4 - target_q) * 900) + (target_m * 60) + target_s
            closest_idx = (df['TimeSecsRem'] - target_time_secs).abs().idxmin()
            st.session_state.play_idx = int(closest_idx)
            st.session_state.is_playing = False
            st.rerun()

    col1, col2, col3 = st.columns([1, 1, 4])
    if col1.button("▶ Play"):
        st.session_state.is_playing = True
        st.rerun()
    if col2.button("⏸ Pause"):
        st.session_state.is_playing = False
        st.rerun()
        
    new_idx = col3.slider("Scrub through plays:", 0, len(df)-1, st.session_state.play_idx)
    if new_idx != st.session_state.play_idx:
        st.session_state.play_idx = new_idx
        st.session_state.is_playing = False
    
    ui_placeholder = st.empty()
    
    def render_play(idx):
        row = df.iloc[idx]
        with ui_placeholder.container():
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric(f"{away_team} (Away)", int(row['away_score']))
            sc2.metric(f"{home_team} (Home)", int(row['home_score']))
            sc3.metric("Quarter", int(row['period']))
            sc4.metric("Time", row['time_str'])
            sc5.metric(f"{home_team} WP", row['wp_percent'])
            
            st.markdown(f"**Possession:** {row['pos_team']} | **Down & Distance:** {int(row['down'])} & {int(row['distance'])} | **Yards to Goal:** {int(row['yards_to_goal'])}")
            st.plotly_chart(draw_football_field(row), use_container_width=True, key=f"field_{idx}")
            
            # Play Type & Touchdown Logic
            play_text_lower = str(row['play_text']).lower()
            play_type_lower = str(row['play_type']).lower()
            is_touchdown = 'touchdown' in play_text_lower or ' td ' in play_text_lower or 'touchdown' in play_type_lower
            
            if is_touchdown:
                st.success(f"🚨 **TOUCHDOWN!** {row['play_text']}")
            else:
                st.info(f"🗣️ **Play Type [{row['play_type']}]:** {row['play_text']}")
                
            # --- AUDIO PLAYBACK LOGIC ---
            if enable_sound:
                audio_file = 'touchdown.mp3' if is_touchdown else 'soccer sounds info 516.mp3'
                if os.path.exists(audio_file):
                    st.audio(audio_file, autoplay=True)
                # else: (silently fail if they didn't upload the file yet to avoid spamming the UI)
            
            fig_wp = go.Figure(base_wp_fig)
            fig_wp.add_trace(go.Scatter(x=[row['seconds_elapsed']], y=[row['home_wp']], mode='markers', marker=dict(color='red', size=15), showlegend=False))
            st.plotly_chart(fig_wp, use_container_width=True, key=f"wp_{idx}")

    # Main Loop
    if st.session_state.is_playing:
        while st.session_state.play_idx < len(df) - 1:
            render_play(st.session_state.play_idx)
            time.sleep(play_speed)
            st.session_state.play_idx += 1
        st.session_state.is_playing = False
    else:
        render_play(st.session_state.play_idx)

    # TOP PLAYS SECTION 
    st.markdown("### 🔥 Top 3 Game-Changing Plays")
    top_plays = df.sort_values(by='abs_wp_swing', ascending=False).head(3)
    
    for i, (_, row) in enumerate(top_plays.iterrows(), 1):
        swing_percent = round(float(row['home_wp_swing']) * 100, 1)
        swing_color = "green" if swing_percent > 0 else "red"
        swing_sign = "+" if swing_percent > 0 else ""
        
        with st.container():
            st.markdown(f"**{i}. Q{int(row['period'])} | {row['time_str']} | {row['pos_team']} ball**")
            st.markdown(f"> *{row['play_text']}*")
            st.markdown(f"**Win Probability Swing:** <span style='color:{swing_color}; font-weight:bold;'>{swing_sign}{swing_percent}%</span>", unsafe_allow_html=True)
            st.markdown("---")
