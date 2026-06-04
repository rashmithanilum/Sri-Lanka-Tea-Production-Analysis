import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json
import base64
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------
# 1. General Setup & Tech Stack
# ---------------------------------------------------------
st.set_page_config(page_title="Sri Lankan Tea Production", layout="wide", page_icon="🍵")

@st.cache_data
def load_data():
    df = pd.read_excel("sorted.xlsx")
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

@st.cache_resource
def load_models_metadata():
    total_model = joblib.load("total_sarimax.joblib")
    share_model = joblib.load("share_catboost.joblib")
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    return total_model, share_model, metadata

try:
    data = load_data()
    total_model, share_model, metadata = load_models_metadata()
except Exception as e:
    st.error(f"Error loading files. Ensure csv, joblib, and json files exist. Details: {e}")
    st.stop()

# ---------------------------------------------------------
# Dynamic Watermark Background
# ---------------------------------------------------------
def set_watermark_overlay(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            .stApp::after {{
                content: "";
                background-image: url("data:image/jpg;base64,{b64}");
                background-size: 400px;
                background-repeat: repeat;
                background-position: top left;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                opacity: 0.08;
                pointer-events: none;
                z-index: 0;
            }}
            
            /* Premium Metric Hover Animations */
            [data-testid="stMetric"] {{
                background-color: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 10px;
                transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}

            [data-testid="stMetric"]:hover {{
                transform: translateY(-5px) scale(1.02);
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
                background-color: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(133, 193, 233, 0.4);
                cursor: default;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

set_watermark_overlay("watermark_transparent.jpg")

# ---------------------------------------------------------
# Classification Data Prep & Training (Page 3)
# ---------------------------------------------------------
# Classification logic replaced by Time-Series ADF module.

# ---------------------------------------------------------
# Sidebar Navigation Bar
# ---------------------------------------------------------
st.sidebar.title("🍵 Navigation")
page = st.sidebar.radio(
    "Select a Page:",
    [
        "Data Exploration & EDA",
        "Market Forecast",
        "Monte Carlo Scenario",
        "Limitations & Assumptions"
    ]
)
st.sidebar.markdown("---")

if page == "Data Exploration & EDA":
    # ---------------------------------------------------------
    # PAGE 1: Data Exploration & EDA
    # ---------------------------------------------------------
    st.title("Prediction of Tea Production in Sri Lanka")

    # Sidebar Data Filters specific to Page 1
    st.sidebar.header("Global Data Filters")
    min_year = int(data['Year'].min())
    max_year = int(data['Year'].max())
    selected_years = st.sidebar.slider("Select Year Range:", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    elevations = data['Elevation'].unique().tolist()
    tea_types = data['Tea Type'].unique().tolist()
    selected_elevations = st.sidebar.pills("Select Elevation:", elevations, default=elevations, selection_mode="multi")
    selected_tea_types = st.sidebar.pills("Select Tea Type:", tea_types, default=tea_types, selection_mode="multi")

    # Filtering data logic
    filtered_data = data[
        (data['Year'] >= selected_years[0]) &
        (data['Year'] <= selected_years[1]) &
        (data['Elevation'].isin(selected_elevations)) &
        (data['Tea Type'].isin(selected_tea_types))
    ]

    st.header("Data Exploration")
    st.markdown("Below is a snapshot of the raw data based on your current filter selection.")
    
    # 1-based indexing for the display
    display_df = filtered_data.copy()
    display_df.index = np.arange(1, len(display_df) + 1)
    st.dataframe(display_df, use_container_width=True)

    st.markdown("""
    ---
    
    ### 🎯 Objectives
    Our primary thesis is to statistically determine which climatic factors (Rainfall, Humidity, Temperature) act as the most significant drivers of yield and whether these impacts heavily vary across distinct elevations.
    
    * **Temporal Patterns:** Examine how tea production fluctuates across months and years to formally identify seasonality and long-term macro-trends.
    * **Climatic Influence:** Analyze the isolated effect of rainfall, temperature, and humidity on tea production to identify key driving climatic constraints.
    * **Elevation Effects:** Study how tea production organically differs across altitude levels and how elevation ultimately modifies climate-production relationships.
    * **Tea Type Wise Production:** Evaluate systematic differences in production among granular tea types and their individualized responses to historical climatic conditions.
    
    *Ultimate Objective:* Integrate temporal, climatic, elevation, and tea-type insights to map overall architecture relevant for accurately forecasting Sri Lanka's global tea production.
    """)

    st.markdown("---")
    
    if not filtered_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. General Seasonality & Trend")
            # Objective 1: Temporal Patterns
            temporal_data = filtered_data.groupby('date', as_index=False)['Tea Production'].sum()
            temporal_data = temporal_data.sort_values('date')
            temporal_data['12M MA'] = temporal_data['Tea Production'].rolling(window=12, min_periods=1).mean()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=temporal_data['date'], y=temporal_data['Tea Production'], mode='lines', name='Monthly Volume', line=dict(color='#85C1E9', width=1.5)))
            fig1.add_trace(go.Scatter(x=temporal_data['date'], y=temporal_data['12M MA'], mode='lines', name='12-Month Trend Line', line=dict(color='#C0392B', width=3)))
            fig1.update_layout(title="Tea Production Over Time + Long-term Trend", xaxis_title="Timeline Interval", yaxis_title="Sum Production (kg)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig1, use_container_width=True)
            
            st.info("💡 **Temporal Patterns:** Plot reveals that overall tea production has experienced a general downward trend over the observed period. The 12-month rolling mean provides a smoother representation of this steady decay alongside characteristic peak yields consistently appearing around March to June.")

        with col2:
            st.subheader("2. Yield vs Climate Scatter")
            # Objective 2: Climatic Influence
            climate_var = st.selectbox("Select Exogenous Metric:", ["Air Tempurature", "Humidity", "Rain fall"])
            
            fig2 = px.scatter(filtered_data, x=climate_var, y='Tea Production', color='Elevation', trendline="ols", opacity=0.5,
                              title=f"Objective 2: {climate_var} Isolation Cluster",
                              labels={climate_var: climate_var, 'Tea Production': 'Volume (kg)'})
            st.plotly_chart(fig2, use_container_width=True)
            
            if climate_var == "Air Tempurature":
                st.info("💡 **Climatic Influence:** Air temperature computationally exhibits the strongest positive continuous influence on tea production, making it a critical driver determining yield variability.")
            elif climate_var == "Humidity":
                st.info("💡 **Climatic Influence:** Humidity maintains a weak but structurally positive linear association. Increases taper off at higher boundary points heavily suggesting biological saturation.")
            else:
                st.info("💡 **Climatic Influence:** Rainfall definitively registers a negligible direct linear threshold with total crop scale, operating loosely on its own dimension irrespective of sheer precipitation.")
                
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("3. Production Disparities by Altitude")
            # Objective 3: Elevation Effects
            fig3 = px.box(filtered_data, x='Elevation', y='Tea Production', color='Elevation',
                          title="Volumetric Density Spread Evaluation",
                          labels={'Tea Production': 'Recorded Dispersion (kg)'})
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info("💡 **Elevation Effects:** Low elevation geographically forces the highest median capacity while also reflecting enormous variance. High elevation harvests drastically less net material, but its production floor remains uniquely and exceptionally stable.")
            
        with col4:
            st.subheader("4. Industry Architecture by Tea Type")
            # Objective 4: Tea Type Wise Production
            tea_agg = filtered_data.groupby('Tea Type', as_index=False)['Tea Production'].sum().sort_values(by='Tea Production', ascending=False)
            fig4 = px.bar(tea_agg, x='Tea Type', y='Tea Production', color='Tea Type', text_auto='.2s',
                          title="Global Composition Hierarchy",
                          labels={'Tea Production': 'Aggregate Yield Ceiling (kg)'})
            fig4.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(fig4, use_container_width=True)
            
            st.info("💡 **Total Production by Tea Type:** Industry volume is heavily monopolized symmetrically. `Total Black Tea` and `Orthodox` inherently drive the macroscopic volume baseline; the remaining variations act as relatively small niche market fractions.")
            
    else:
        st.warning("No data rows dynamically intersect with the selected multi-filter requirements. Adjust the sidebar.")


elif page == "Market Forecast":
    # ---------------------------------------------------------
    # PAGE 2: Modelling Component (Market Forecast)
    # ---------------------------------------------------------
    st.title("Market Forecast Simulator")
    st.markdown("Project global Sri Lankan tea industry capacities simulating specific future timeline shocks against multi-variate statistical regressors.")
    
    with st.expander("ℹ️ About Our Forecasting Model"):
        st.markdown("""
        Our approach utilizes a robust **two-stage forecasting pipeline**:
        1. **Total Market Sizing (SARIMAX):** A Seasonal Auto-Regressive Integrated Moving Average model with eXogenous variables. This time-series model calculates the total aggregate volume of tea production based on historical seasonality, trends, and user-provided climate inputs.
        2. **Market Share Distribution (CatBoost):** An advanced gradient boosting regressor that maps complex, non-linear relationships. It accurately distributes the SARIMAX baseline volume across specific Elevations and Tea Types based on their historical lag patterns and sensitivities to climatic forces.
        """)

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        st.subheader("Specify Simulation Inputs")
        
        # Calculate dataset boundaries for sliders
        rain_min = float(np.percentile(data['Rain fall'].dropna(), 5))
        rain_max = float(np.percentile(data['Rain fall'].dropna(), 95))
        rain_mean = float(data['Rain fall'].mean())
    
        hum_min = float(np.percentile(data['Humidity'].dropna(), 5))
        hum_max = float(np.percentile(data['Humidity'].dropna(), 95))
        hum_mean = float(data['Humidity'].mean())
    
        temp_min = float(np.percentile(data['Air Tempurature'].dropna(), 5))
        temp_max = float(np.percentile(data['Air Tempurature'].dropna(), 95))
        temp_mean = float(data['Air Tempurature'].mean())
        
        # State Initialization for perfect symmetrical two-way bindings
        if 'r_sli' not in st.session_state: st.session_state.r_sli = rain_mean
        if 'r_num' not in st.session_state: st.session_state.r_num = rain_mean
        
        if 'h_sli' not in st.session_state: st.session_state.h_sli = hum_mean
        if 'h_num' not in st.session_state: st.session_state.h_num = hum_mean
        
        if 't_sli' not in st.session_state: st.session_state.t_sli = temp_mean
        if 't_num' not in st.session_state: st.session_state.t_num = temp_mean

        # Exact Synchronized Update Callbacks
        def sync_r_sli(): st.session_state.r_num = st.session_state.r_sli
        def sync_r_num(): st.session_state.r_sli = st.session_state.r_num
        def sync_h_sli(): st.session_state.h_num = st.session_state.h_sli
        def sync_h_num(): st.session_state.h_sli = st.session_state.h_num
        def sync_t_sli(): st.session_state.t_num = st.session_state.t_sli
        def sync_t_num(): st.session_state.t_sli = st.session_state.t_num

        # Binded Interactive Rows (No hardcoded value= params)
        cr1, cr2 = st.columns([3, 1])
        cr1.slider("Rainfall Scenario (mm)", min_value=rain_min, max_value=rain_max, key="r_sli", on_change=sync_r_sli)
        cr2.number_input("Type (mm)", min_value=rain_min, max_value=rain_max, key="r_num", on_change=sync_r_num)

        ch1, ch2 = st.columns([3, 1])
        ch1.slider("Humidity Limit (%)", min_value=hum_min, max_value=hum_max, key="h_sli", on_change=sync_h_sli)
        ch2.number_input("Type (%)", min_value=hum_min, max_value=hum_max, key="h_num", on_change=sync_h_num)

        ct1, ct2 = st.columns([3, 1])
        ct1.slider("Air Temp Baseline (°C)", min_value=temp_min, max_value=temp_max, key="t_sli", on_change=sync_t_sli)
        ct2.number_input("Type (°C)", min_value=temp_min, max_value=temp_max, key="t_num", on_change=sync_t_num)

        input_rain = st.session_state.r_sli
        input_hum = st.session_state.h_sli
        input_temp = st.session_state.t_sli
        
        last_date = data['date'].max()
        next_month_date = (last_date.replace(day=1) + pd.Timedelta(days=32)).replace(day=1)
        
        st.markdown("---")
        st.markdown("Target Forecast Date:")
        
        month_labels = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        selected_month_label = st.selectbox("Target Month", month_labels, index=next_month_date.month - 1)
        forecast_month = month_labels.index(selected_month_label) + 1
        
        forecast_year = st.number_input("Target Year", min_value=last_date.year, value=next_month_date.year)

        trigger_forecast = st.button("Initialize Forecast Run", type="primary", use_container_width=True)

    with col_f2:
        if trigger_forecast:
            try:
                # 1. Total Market Size via SARIMAX
                exog_future = pd.DataFrame([{
                    'Rain fall': input_rain,
                    'Humidity': input_hum,
                    'Air Tempurature': input_temp
                }])
                
                total_pred = float(total_model.forecast(steps=1, exog=exog_future).iloc[0])
                st.metric("Total Market Predicted Trajectory (kg)", f"{total_pred:,.2f}")
                
                # 2. Market Share Split using CatBoost
                all_combos = metadata['data']['all_combos']
                structural_zero_combos = metadata['data']['structural_zero_combos']
                
                month_sin = np.sin(2 * np.pi * forecast_month / 12)
                month_cos = np.cos(2 * np.pi * forecast_month / 12)
                
                # Historical function dynamic binding
                def get_historical_share_lags(dataset, elevation, tea_type):
                    combo_df = dataset[(dataset['Elevation'] == elevation) & (dataset['Tea Type'] == tea_type)]
                    combo_df = combo_df.sort_values(by='date')
                    total_per_month = dataset.groupby('date')['Tea Production'].sum()
                    
                    combo_df = combo_df.set_index('date')
                    combo_df['Month Total'] = total_per_month
                    combo_df['share'] = combo_df['Tea Production'] / combo_df['Month Total'].replace(0, 1)
                    
                    shares_history = combo_df['share'].tail(12).values
                    if len(shares_history) < 12:
                        shares_history = np.pad(shares_history, (12 - len(shares_history), 0), 'constant')
                        
                    share_roll3 = np.mean(shares_history[-3:])
                    share_roll6 = np.mean(shares_history[-6:])
                    return shares_history[::-1], share_roll3, share_roll6
                
                rows = []
                for combo in all_combos:
                    elevation = combo['Elevation']
                    tea_type = combo['Tea Type']
                    
                    shares_lags, roll3, roll6 = get_historical_share_lags(data, elevation, tea_type)
                    lag_dict = {f"share_lag_{i+1}": shares_lags[i] for i in range(12)}
                    
                    row = {
                        'Elevation': elevation,
                        'Tea Type': tea_type,
                        'Rain fall': input_rain,
                        'Humidity': input_hum,
                        'Air Tempurature': input_temp,
                        'month_sin': month_sin,
                        'month_cos': month_cos,
                        'share_roll3': roll3,
                        'share_roll6': roll6
                    }
                    row.update(lag_dict)
                    rows.append(row)
                    
                share_features_df = pd.DataFrame(rows)
                feature_cols = metadata['models']['share_model']['feature_cols']
                share_features_df = share_features_df[feature_cols] # Column Ordering mapping
                
                raw_shares = share_model.predict(share_features_df)
                raw_shares = np.maximum(raw_shares, 0)
                
                # Enforcing Structural Zeros precisely 
                for idx, combo in enumerate(all_combos):
                    is_zero = any((combo['Elevation'] == sz['Elevation'] and combo['Tea Type'] == sz['Tea Type']) 
                                  for sz in structural_zero_combos)
                    if is_zero:
                        raw_shares[idx] = 0.0
                
                # Distribution Share Normalization Mathematics
                total_raw_share = np.sum(raw_shares)
                final_shares = raw_shares / total_raw_share if total_raw_share > 0 else np.zeros_like(raw_shares)
                
                # Merge into Absolute Values
                final_predictions = final_shares * total_pred
                
                results_df = pd.DataFrame({
                    'Matrix Dimension 1 (Elevation)': [c['Elevation'] for c in all_combos],
                    'Matrix Dimension 2 (Tea Type)': [c['Tea Type'] for c in all_combos],
                    'Normalized Confidence Bound (%)': final_shares * 100,
                    'Projected Future Value (kg)': final_predictions
                })
                
                results_df = results_df.sort_values(by='Projected Future Value (kg)', ascending=False)
                
                st.success("Targeting simulation succeeded.")
                st.dataframe(
                    results_df.style.format({
                        'Normalized Confidence Bound (%)': '{:.3f}%',
                        'Projected Future Value (kg)': '{:,.2f}'
                    }),
                    use_container_width=True
                )
                st.markdown("💡 **Interpretation Engine:** Total market bulk was aggregated utilizing standard **SARIMAX** components. "
                            "It was then allocated fractionally across the framework using a **CatBoost** hierarchy. "
                            "Critically, **Instant Tea at Low/Medium Elevations** is enforced identically down to **0 kg** corresponding strictly with our non-linear distribution rules established in the metadata framework.")
                
            except Exception as e:
                st.error(f"Inference pipeline execution error: {e}")

elif page == "Monte Carlo Scenario":
    # ---------------------------------------------------------
    # PAGE 3: Scenario-Based Monte Carlo Simulation
    # ---------------------------------------------------------
    st.title("Scenario-Based Monte Carlo Simulation")
    st.header("Individual Contribution: Probabilistic Volatility")
    
    st.markdown("""
    > *"A Monte Carlo-based scenario simulation was conducted by varying climatic inputs within realistic ranges and generating multiple production outcomes using the trained SARIMAX/CatBoost models to analyze uncertainty and climate sensitivity in tea yield."*

    Traditional analytical forecasting predicts a generic, singular "Expected Value". However, Sri Lankan agriculture is intrinsically prone to unpredictable exogenous variance. 
    This module utilizes an advanced **Monte Carlo Engine** layered natively on top of the pre-trained overarching **SARIMAX** and **CatBoost** predictive infrastructure. 
    By mathematically mutating the climatic bounds through precise Gaussian distributions 500+ times, we compute an entire probability matrix visually exposing your actual structural risk limitations.
    """)
    
    col_input, col_output = st.columns([1, 2])
    
    with col_input:
        st.subheader("Configure Reality Constraints")
        sim_elevation = st.selectbox("Select Target Elevation:", sorted(data['Elevation'].dropna().unique()), key='sim_elev')
        sim_tea_type = st.selectbox("Select Tea Variety:", sorted(data['Tea Type'].dropna().unique()), key='sim_tea')
        
        # Localize chronologic string bindings mathematically 
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        sim_month = st.selectbox("Select Calendar Timeline:", list(month_map.keys()), key='sim_month')
        
        st.markdown("---")
        sim_scenario = st.radio("Define Climate Mutation:", ["Normal (Baseline)", "Dry & Hot", "Monsoon / Wet"])
        
        run_sim = st.button("Execute 500-Iteration Simulation", type="primary", use_container_width=True)
        
    with col_output:
        st.subheader("Simulated Yield Probability Layout")
        
        if run_sim:
            with st.spinner("Mathematically mapping 500 localized variations uniformly through SARIMAX and CatBoost..."):
                try:
                    import numpy as np
                    iters = 500
                    
                    # 1. Base Variables Extract
                    # Searching for historical structural means accurately matching the target combo
                    base_data = data[(data['Elevation'] == sim_elevation) & (data['Month'] == sim_month)]
                    if base_data.empty:
                        base_rain, base_hum, base_temp = 200.0, 80.0, 22.0
                    else:
                        base_rain = float(base_data['Rain fall'].mean())
                        base_hum = float(base_data['Humidity'].mean())
                        base_temp = float(base_data['Air Tempurature'].mean())
                        
                    # 2. Scenario Mutations
                    if sim_scenario == "Dry & Hot":
                        base_temp += 1.5
                        base_rain *= 0.85
                        base_hum -= 5.0
                    elif sim_scenario == "Monsoon / Wet":
                        base_temp -= 1.0
                        base_rain *= 1.20
                        base_hum += 10.0
                        
                    # 3. Generating Gaussian Vector Arrays natively scaling baseline variance
                    np.random.seed(42)
                    sim_rain = np.random.normal(base_rain, base_rain * 0.15, iters)
                    sim_temp = np.random.normal(base_temp, 2.0, iters)
                    sim_hum = np.random.normal(base_hum, base_hum * 0.10, iters)
                    
                    # 4. Preparing mathematical arrays for CatBoost Share Isolation
                    s_month_num = month_map[sim_month]
                    s_month_sin = np.sin(2 * np.pi * s_month_num / 12)
                    s_month_cos = np.cos(2 * np.pi * s_month_num / 12)
                    
                    # Extrapolating historical share dependencies thoroughly natively
                    def get_historical_share_lags_mc(dataset, elevation, tea_type):
                        combo_df = dataset[(dataset['Elevation'] == elevation) & (dataset['Tea Type'] == tea_type)]
                        combo_df = combo_df.sort_values(by='date')
                        total_per_month = dataset.groupby('date')['Tea Production'].sum()
                        
                        combo_df = combo_df.set_index('date')
                        combo_df['Month Total'] = total_per_month
                        combo_df['share'] = combo_df['Tea Production'] / combo_df['Month Total'].replace(0, 1)
                        
                        shares_history = combo_df['share'].tail(12).values
                        if len(shares_history) < 12:
                            shares_history = np.pad(shares_history, (12 - len(shares_history), 0), 'constant')
                            
                        share_roll3 = float(np.mean(shares_history[-3:]))
                        share_roll6 = float(np.mean(shares_history[-6:]))
                        return shares_history[::-1], share_roll3, share_roll6
                    
                    shares_lags, roll3, roll6 = get_historical_share_lags_mc(data, sim_elevation, sim_tea_type)
                    
                    features_dict = {
                        'Elevation': [sim_elevation]*iters,
                        'Tea Type': [sim_tea_type]*iters,
                        'Rain fall': sim_rain,
                        'Humidity': sim_hum,
                        'Air Tempurature': sim_temp,
                        'month_sin': [s_month_sin]*iters,
                        'month_cos': [s_month_cos]*iters,
                        'share_roll3': [roll3]*iters,
                        'share_roll6': [roll6]*iters
                    }
                    for idx_lag in range(12):
                        features_dict[f'share_lag_{idx_lag+1}'] = [shares_lags[idx_lag]]*iters
                    
                    features_df = pd.DataFrame(features_dict)
                    
                    # CatBoost Vectorized Array computation
                    sim_shares = share_model.predict(features_df)
                    sim_shares = np.clip(sim_shares, 0, 1) # Structurally bounding shares natively
                    
                    # SARIMAX Sequential array computation
                    sim_totals = np.zeros(iters)
                    for i in range(iters):
                        exog_row = pd.DataFrame({'Rain fall': [sim_rain[i]], 'Humidity': [sim_hum[i]], 'Air Tempurature': [sim_temp[i]]})
                        sim_totals[i] = total_model.forecast(steps=1, exog=exog_row).iloc[0]
                        
                    # Multiplying macroscopic total vectors by microscopic specific share vectors natively
                    final_yields = sim_totals * sim_shares
                    
                    # Enforcing Critical Structural Zeros natively matching framework limits
                    if sim_tea_type == 'INSTANT TEA' and sim_elevation in ['LOW', 'MEDIUM']:
                        final_yields = np.zeros(iters)
                        
                    # 5. Result Extraction Pipeline
                    mean_yield = float(np.mean(final_yields))
                    std_yield = float(np.std(final_yields))
                    worst_case = float(np.percentile(final_yields, 5))
                    best_case = float(np.percentile(final_yields, 95))
                    
                    # Visual Rendering Pipeline (Distribution Curve + Margin Boxes)
                    fig_sim = px.histogram(x=final_yields, nbins=30, marginal="box", 
                                           title=f"Statistical Volatility Matrix: {sim_scenario} [{iters} Realities Analyzed]",
                                           labels={'x': 'Computationally Derived Yield (kg)', 'y': 'Frequency Count'},
                                           color_discrete_sequence=['#AF7AC5'])
                    fig_sim.add_vline(x=mean_yield, line_dash="solid", line_color="#2E4053", annotation_text="Mean Yield Structure")
                    fig_sim.add_vline(x=worst_case, line_dash="dash", line_color="#E74C3C", annotation_text="5th Percentile Bound")
                    fig_sim.add_vline(x=best_case, line_dash="dash", line_color="#1E8449", annotation_text="95th Percentile Bound")
                    
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    st.markdown("### 📊 Financial Risk Readout")
                    met1, met2, met3 = st.columns(3)
                    met1.metric("Predicted Mean Target", f"{mean_yield:,.2f} kg")
                    met2.metric("Worst Case Bound (5%)", f"{worst_case:,.2f} kg", delta=f"{worst_case - mean_yield:,.2f} kg")
                    met3.metric("Best Case Bound (95%)", f"{best_case:,.2f} kg", delta=f"{best_case - mean_yield:,.2f} kg", delta_color="normal")
                    
                except Exception as e:
                    st.error(f"Mathematical derivation failed mapping Monte Carlo loops natively: {e}")
        else:
            st.info("The physics engine is awaiting the launch command. Adjust your independent parameters effectively and launch the calculation loop.")

elif page == "Limitations & Assumptions":
    # ---------------------------------------------------------
    # PAGE 4: Limitations and assumptions
    # ---------------------------------------------------------
    st.title("Framework Limits & Assumptions Matrix")
    st.markdown("This sector outlines critical analytical barriers within the global modeling approach.")
    
    st.markdown("""
    ### K-Means Weak Spatial Distinctions
    The structural cluster analysis evaluated in the root global report showcased identifiable constraints. Primarily, it resulted in comparatively low **silhouette scores**. 
    This quantifies weak structural cluster margins and indicates excessive feature overlap. Without incorporating localized geographic markers or temporal harvesting phase signals alongside just macro-weather averages, historical sub-feature grouping isolates weakly.

    ### Time-Invariant Climate Response Assumptions
    Leveraging exogenous metrics intrinsically enforces that future shocks identically match past sensitivity loops linearly. Extreme, hyper-localized black-swan climatic sequences fall deeply outside this boundary representation, forcing unpredictable biological crop behavior entirely beyond generic variable boundaries.
    
    ### Granular Limitations
    The data aggregates whole regions mathematically. Therefore, pinpoint localized plantation microclimates cannot individually resolve statistical significance through our standardized CatBoost distribution engine alone. Introducing recurrent frameworks (like LSTMs networks) to catch trailing non-linear correlations represents our future scope objective.
    """)
