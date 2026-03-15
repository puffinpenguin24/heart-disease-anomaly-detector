import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Heart Disease Anomaly Tracker",
    page_icon="🔍",
    layout="wide"
)

# COLOR MAP: Blue for Normal, Red for Outlier
COLOR_MAP = {'Normal': '#1f77b4', 'Outlier': '#d62728'}


# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # File specified: heart_2022_with_nans.csv
        df = pd.read_csv('heart_2022_with_nans.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please ensure 'heart_2022_with_nans.csv' is in the project folder.")
        return pd.DataFrame()


# --- 3. ML LOGIC ---
def preprocess_and_predict(data, selected_features, outlier_percent):
    if data.empty:
        return np.array([])

    X = data[selected_features].copy()
    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    if not num_cols.empty:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    if not cat_cols.empty:
        X[cat_cols] = X[cat_cols].fillna("Unknown")
        encoder = OrdinalEncoder()
        X[cat_cols] = encoder.fit_transform(X[cat_cols])

    model = IsolationForest(contamination=outlier_percent / 100, random_state=42)
    return model.fit_predict(X)


# --- 4. MAIN USER INTERFACE ---
def main():
    st.title("🔍 Heart Disease Anomaly Dashboard")
    st.markdown("---")

    df_raw = load_data()
    if df_raw.empty:
        return

    # --- SIDEBAR ---
    st.sidebar.header("📍 1. Data Selection")
    state_col = 'State' if 'State' in df_raw.columns else None
    if state_col:
        states = sorted(df_raw[state_col].astype(str).unique().tolist())
        selected_state = st.sidebar.selectbox("Filter Region", ["National (All States)"] + states)
    else:
        selected_state = "National (All States)"

    st.sidebar.header("🛠️ 2. Model Parameters")
    all_cols = df_raw.columns.tolist()
    suggested = ['BMI', 'WeightInKilograms', 'SleepTime', 'PhysicalHealth']
    default_features = [c for c in suggested if c in all_cols]

    selected_features = st.sidebar.multiselect(
        "Features to Analyze",
        options=all_cols,
        default=default_features if default_features else all_cols[:4]
    )

    outlier_percent = st.sidebar.slider(
        "Expected Outlier %",
        min_value=1.0, max_value=20.0, value=5.0, step=0.5
    )

    st.sidebar.divider()
    run_button = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

    if st.sidebar.button("🔄 Reset App", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # --- EXECUTION ENGINE ---
    if run_button or 'analysis_done' in st.session_state:
        if run_button:
            st.session_state.analysis_done = True
            st.session_state.current_state = selected_state
            st.session_state.current_features = selected_features
            st.session_state.current_outlier_pct = outlier_percent

        s_state = st.session_state.current_state
        s_features = st.session_state.current_features
        s_pct = st.session_state.current_outlier_pct

        if s_state != "National (All States)" and state_col:
            df_working = df_raw[df_raw[state_col] == s_state].copy()
        else:
            df_working = df_raw.copy()

        if not s_features:
            st.error("Please select features in the sidebar.")
            return

        with st.spinner(f"Analyzing {s_state}..."):
            preds = preprocess_and_predict(df_working, s_features, s_pct)
            df_working['Anomaly_Status'] = np.where(preds == -1, 'Outlier', 'Normal')

            # --- 5. DATA SUMMARY CARDS ---
            total_n = len(df_working)
            outlier_n = len(df_working[df_working['Anomaly_Status'] == 'Outlier'])
            normal_n = total_n - outlier_n

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Population", f"{total_n:,}")
            m2.metric("Normal Profiles", f"{normal_n:,}", delta=None)
            m3.metric("Detected Outliers", f"{outlier_n:,}", delta=f"{s_pct}% Expected", delta_color="inverse")
            st.markdown("---")

            # --- 6. POPULATION BREAKDOWN & AVERAGES ---
            st.subheader(f"Anomaly Results: {s_state}")
            col1, col2 = st.columns([1, 2])

            with col1:
                pie_option = st.selectbox("Breakdown Pie Chart by:", options=["Overall Status"] + s_features,
                                          key="pie_selector")
                if pie_option == "Overall Status":
                    fig_pie = px.pie(df_working, names='Anomaly_Status', hole=0.5, color='Anomaly_Status',
                                     color_discrete_map=COLOR_MAP, title="Proportion of Anomalies")
                else:
                    fig_pie = px.pie(df_working, names=pie_option, hole=0.5, title=f"Proportion by {pie_option}",
                                     template="plotly_white")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.write("**Group Comparison (Means)**")
                table_data = df_working.groupby('Anomaly_Status')[s_features].mean(numeric_only=True).round(2)
                table_data = table_data.dropna(how='all')
                st.table(table_data)

            # --- 7. CORRELATION HEATMAP ---
            st.divider()
            st.subheader("🌡️ Feature Correlation Heatmap")
            latest_numeric = df_working[s_features].select_dtypes(include=[np.number]).columns.tolist()
            if len(latest_numeric) > 1:
                corr_matrix = df_working[latest_numeric].corr().round(2)
                fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Select at least two numeric features to see correlations.")

            # --- 8. SCATTER PLOT (CLUSTER VIEW) ---
            st.divider()
            st.subheader("🎯 2D Cluster Visualization")
            if len(latest_numeric) >= 2:
                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    x_feat = st.selectbox("X-axis:", options=latest_numeric, index=0, key="scatter_x")
                with sc_col2:
                    y_feat = st.selectbox("Y-axis:", options=latest_numeric, index=1 if len(latest_numeric) > 1 else 0,
                                          key="scatter_y")

                fig_scatter = px.scatter(df_working, x=x_feat, y=y_feat, color='Anomaly_Status',
                                         color_discrete_map=COLOR_MAP, opacity=0.6, template="plotly_white",
                                         marginal_x="histogram", marginal_y="box")
                fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
                                          selector=dict(type='scatter'))
                fig_scatter.data = tuple(
                    sorted(fig_scatter.data, key=lambda trace: 1 if trace.name == "Outlier" else 0))
                st.plotly_chart(fig_scatter, use_container_width=True)

            # --- 9. STATISTICAL DISTRIBUTION ---
            st.divider()
            st.subheader("📊 Distribution (Quartiles & Median)")
            if latest_numeric:
                plot_feat = st.selectbox("Select feature for Boxplot:", options=latest_numeric, key="box_selector")
                fig_box = px.box(df_working, y=plot_feat, x='Anomaly_Status', color='Anomaly_Status',
                                 color_discrete_map=COLOR_MAP, notched=True, points="outliers", template="plotly_white")
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

            # --- 10. RAW DATA & DOWNLOAD ---
            st.divider()
            st.subheader("📋 Detected Outlier Profiles")
            outliers = df_working[df_working['Anomaly_Status'] == 'Outlier']
            outliers_clean = outliers.dropna(axis=1, how='all')
            st.dataframe(outliers_clean.head(100), use_container_width=True)

            csv = outliers_clean.to_csv(index=False).encode('utf-8')
            st.sidebar.success("Analysis Complete!")
            safe_state = s_state.replace(" ", "_").replace("(", "").replace(")", "")
            st.sidebar.download_button(label="📥 Download Results (CSV)", data=csv,
                                       file_name=f'outliers_{safe_state}.csv', mime='text/csv',
                                       use_container_width=True)
    else:
        st.info("👈 Set parameters and click 'Run Analysis' to see results.")
        st.write(f"**Loaded Dataset:** {df_raw.shape[0]:,} rows")


if __name__ == "__main__":
    main()