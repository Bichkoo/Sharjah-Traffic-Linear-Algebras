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
    
    # Section 3: Ordinary Least Squares (OLS) via Normal Equations
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
