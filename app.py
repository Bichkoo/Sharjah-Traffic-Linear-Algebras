import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import time

# ==========================================
# PAGE CONFIGURATION & AESTHETICS
# ==========================================
st.set_page_config(
    page_title="Sharjah Commute Optimizer",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, premium look
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    [data-testid="stMetric"] { 
        background-color: var(--secondary-background-color); 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2); 
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING & MATH ENGINE (MATH 101)
# ==========================================
@st.cache_data
def load_and_train_model():
    """
    Loads the dataset and computes the Least Squares beta vector
    using pure linear algebra: beta = (X^T X)^-1 X^T y
    """
    try:
        df = pd.read_csv("sharjah_congestion.csv")
    except FileNotFoundError:
        # Fallback dummy data if CSV isn't in the same folder yet
        st.error("Error: 'sharjah_congestion.csv' not found. Please ensure it's in the same directory.")
        st.stop()

    # Melt the dataframe into a long format: [Hour, Day, Congestion]
    melted = df.melt(id_vars='Hour', var_name='Day', value_name='Congestion')
    
    # Section 4.3: Interaction Terms for Workday Variance
    # W = 1 for Monday-Friday, W = 0 for Saturday-Sunday
    melted['W'] = melted['Day'].apply(lambda x: 1 if x in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] else 0)
        
    # Section 4.1 & 4.2: Cyclical Time Mapping & Fourier Harmonics
    melted['sin24'] = np.sin(2 * np.pi * melted['Hour'] / 24)
    melted['cos24'] = np.cos(2 * np.pi * melted['Hour'] / 24)
    melted['sin12'] = np.sin(4 * np.pi * melted['Hour'] / 24)
    melted['cos12'] = np.cos(4 * np.pi * melted['Hour'] / 24)
    melted['sin8'] = np.sin(6 * np.pi * melted['Hour'] / 24) # 8-Hour Harmonic
    melted['cos8'] = np.cos(6 * np.pi * melted['Hour'] / 24)
    
    # Section 4.3: Multiplying baseline sine and cosine features by W
    melted['W_sin24'] = melted['W'] * melted['sin24']
    melted['W_cos24'] = melted['W'] * melted['cos24']
    
    # Construct Design Matrix X and Target Vector y
    feature_cols = ['W', 'sin24', 'cos24', 'sin12', 'cos12', 'sin8', 'cos8', 'W_sin24', 'W_cos24']
    X = np.column_stack((np.ones(len(melted)), melted[feature_cols].values)) # Adding Intercept
    y = melted['Congestion'].values
    
    # Ordinary Least Squares (OLS) via pure matrix algebra
    X_T = X.T
    beta = np.linalg.inv(X_T @ X) @ X_T @ y
    
    return df, beta

# Load data and train
raw_df, beta_vector = load_and_train_model()

def predict_congestion(day, hour_float):
    """Predicts congestion percentage at a continuous time using the trained beta vector."""
    w_val = 1 if day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] else 0
    
    sin24 = np.sin(2 * np.pi * hour_float / 24)
    cos24 = np.cos(2 * np.pi * hour_float / 24)
    sin12 = np.sin(4 * np.pi * hour_float / 24)
    cos12 = np.cos(4 * np.pi * hour_float / 24)
    sin8 = np.sin(6 * np.pi * hour_float / 24)
    cos8 = np.cos(6 * np.pi * hour_float / 24)
    
    x_vec = [
        1,       # Intercept
        w_val,   # Workday Indicator
        sin24, cos24,
        sin12, cos12,
        sin8, cos8,
        w_val * sin24, # Interaction terms active only on workdays
        w_val * cos24
    ]
    
    prediction = np.dot(x_vec, beta_vector)
    return max(0, prediction) # Congestion cannot be logically negative


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def float_to_time_str(h_float):
    """Converts a float hour (e.g. 7.5) to a formatted string (07:30 AM)."""
    h_float = max(0, h_float) # Prevent negative time, allow wrapping past midnight
    hours = int(h_float)
    minutes = int(round((h_float - hours) * 60))
    if minutes == 60:
        hours += 1
        minutes = 0
    t = time(hour=hours % 24, minute=minutes)
    return t.strftime("%I:%M %p")

def time_to_float(t):
    """Converts a datetime.time object to a float."""
    return t.hour + t.minute / 60.0


# ==========================================
# USER INTERFACE (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204094.png", width=60) # Placeholder aesthetic icon
    st.title("Commute Parameters")
    st.markdown("Set your route details below to calculate the mathematical optimum.")
    
    selected_day = st.selectbox("Day of the Week", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    target_time_input = st.time_input("Target Arrival Time", time(9, 0)) # Default 9:00 AM
    target_time_float = time_to_float(target_time_input)
    
    distance_km = st.number_input("Route Distance (km)", min_value=1.0, max_value=200.0, value=40.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### 🧮 Model Constraints")
    st.latex(r"T_{base} = d \text{ (mins)}")
    st.latex(r"D(t) = T_{base} + \left(\frac{C(t)}{100} \times 1.5 \times T_{base}\right)")
    st.latex(r"t_{dep} + \frac{D(t)}{60} \le t_{target}")


# ==========================================
# OPTIMIZATION ALGORITHM
# ==========================================
# We will simulate departure times throughout the day (every 3 minutes = 0.05 hours)
time_steps = np.arange(0, 24, 0.05)
results = []

T_base = distance_km # Assuming 60km/h baseline speed

for t_dep in time_steps:
    c = predict_congestion(selected_day, t_dep)
    
    # Calculate drive time in minutes
    drive_time_mins = T_base + (c / 100.0) * 1.5 * T_base
    
    # Calculate arrival time in hours
    t_arr = t_dep + (drive_time_mins / 60.0)
    
    if t_arr <= target_time_float:
        results.append({
            't_dep': t_dep,
            't_arr': t_arr,
            'drive_time': drive_time_mins,
            'congestion': c
        })

# ==========================================
# MAIN DASHBOARD AREA
# ==========================================
st.title("Sharjah Traffic Prescriptive Model")
st.markdown("This dashboard maps predictive traffic percentages to schedule the **mathematically optimal departure time**, bypassing black-box libraries using pure Least Squares Approximation.")

if not results:
    st.error("⚠️ Impossible to reach the destination by the target time on the same day. Try an earlier departure or later arrival time.")
else:
    # Find the optimal departure (Minimizes drive time)
    results_df = pd.DataFrame(results)
    optimal = results_df.loc[results_df['drive_time'].idxmin()]
    
    # Find the latest possible departure (Maximizes departure time while satisfying constraint)
    latest = results_df.loc[results_df['t_dep'].idxmax()]

    # Metric Cards
    st.markdown("### 🎯 Recommended: Latest Safe Departure")
    st.markdown("The maximum boundary constraint: the absolute **latest time you can leave** to satisfy your target arrival deadline.")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Latest Departure", float_to_time_str(latest['t_dep']), "Deadline to leave")
    with col2:
        st.metric("Estimated Arrival", float_to_time_str(latest['t_arr']))
    with col3:
        st.metric("Total Drive Time", f"{int(latest['drive_time'])} mins", f"+{int(latest['drive_time'] - T_base)} mins delay", delta_color="inverse")
    with col4:
        st.metric("Congestion at Departure", f"{latest['congestion']:.1f}%")

    # ==========================================
    # VISUALIZATION (PLOTLY)
    # ==========================================
    st.markdown("### 📈 Continuous Traffic Curve & Feasible Region")
    
    # Generate continuous curve for the plot
    plot_x = np.arange(0, 24, 0.1)
    plot_y = [predict_congestion(selected_day, x) for x in plot_x]
    
    fig = go.Figure()
    
    # Add predicted congestion curve
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y,
        mode='lines',
        name='Predicted Congestion',
        line=dict(color='royalblue', width=3),
        fill='tozeroy',
        fillcolor='rgba(65, 105, 225, 0.1)'
    ))
    
    # Add raw data points for verification
    raw_day_data = raw_df[['Hour', selected_day]]
    fig.add_trace(go.Scatter(
        x=raw_day_data['Hour'], y=raw_day_data[selected_day],
        mode='markers',
        name='Raw Observations',
        marker=dict(color='rgba(255, 99, 71, 0.6)', size=8)
    ))

    # Highlight Latest Safe Departure
    fig.add_trace(go.Scatter(
        x=[latest['t_dep']], y=[latest['congestion']],
        mode='markers+text',
        name='Latest Safe Departure',
        marker=dict(color='red', size=14, symbol='star'),
        text=["Latest Safe Departure"],
        textposition="top center"
    ))

    # Add Target Arrival Time Boundary
    fig.add_vline(x=target_time_float, line_width=2, line_dash="dash", line_color="red", 
                  annotation_text="Deadline", annotation_position="top left")

    # Format Axes
    fig.update_layout(
        xaxis=dict(
            title="Time of Day (Hours)",
            tickvals=list(range(0, 25, 2)),
            ticktext=[float_to_time_str(h) for h in range(0, 25, 2)]
        ),
        yaxis=dict(title="Congestion (%)", range=[0, max(100, max(plot_y)+10)]),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MATH DOCUMENTATION EXPANDER
# ==========================================
with st.expander("📚 View Mathematical Reasoning (MATH 101 Documentation)"):
    st.markdown("""
    ### 1. The Overdetermined System
    When mapping traffic congestion percentages ($y$) against time ($X$), we have hundreds of hourly observations but only a few temporal variables. This creates a system of linear equations:
    
    $$X\\beta = y$$
    
    Because $X$ is a tall, rectangular matrix, the system is **inconsistent**. To find the line of best fit, we multiply both sides by $X^T$ to solve the normal equations:
    
    $$\\beta = (X^T X)^{-1} X^T y$$
    
    ### 2. Feature Engineering
    We engineered a design matrix using:
    * **Categorical Interaction Terms:** One-hot encoding for days of the week.
    * **Fourier Harmonics:** Cyclical mappings using $\\sin(\\frac{2\\pi t}{24})$ and $\\cos(\\frac{2\\pi t}{24})$ to model the continuous 24-hour cycle of a day.
    
    ### 3. Constrained Optimization
    The objective is to find the departure time $t_{dep}$ that minimizes drive time $D(t_{dep})$, subject to the constraint that final arrival time must be less than or equal to $t_{target}$.
    
    $$t_{dep} + \\frac{D(t_{dep})}{60} \\le t_{target}$$
    """)
