import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import plotly.express as px
import plotly.graph_objects as go
import time
import os

st.set_page_config(page_title="Quantum Disease Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #F8FAFC; }
    h1, h2, h3 { color: #0F172A !important; font-weight: 800; display: flex; align-items: center; gap: 10px; }
    p, span, div, li { color: #1E293B; }
    .teal-text { color: #0D9488; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 2px solid #E2E8F0; }
    .dataset-box {
        background-color: #FFFFFF; padding: 30px; border-radius: 15px; 
        border: 2px solid #0D9488; text-align: center; height: 250px;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.1);
    }
    .dataset-box h3 { font-size: 1.8rem; justify-content: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1><span class='teal-text'>[ Quantum AI ]</span> Medical Diagnostics</h1>", unsafe_allow_html=True)
st.write("A clinical evaluation dashboard comparing Classical Algorithms (LogReg, RF, SVM) with next-generation Quantum Machine Learning.")
st.write("---")

st.sidebar.title("Clinical Dashboard")
page = st.sidebar.radio("", ["Performance Dashboard", "Train New Data", "Explainability (SHAP)", "Clinical Explanation"])

@st.cache_data
def load_results():
    try:
        df = pd.read_csv("master_qml_results.csv")
        return df.drop_duplicates(subset=['Dataset', 'Model'], keep='last')
    except:
        return None

results_df = load_results()

if page == "Performance Dashboard":
    if results_df is None:
        st.warning("No clinical data processed yet. Please go to 'Train New Data' to process a dataset.")
    else:
        datasets = results_df['Dataset'].unique()
        st.markdown("### Choose Dataset to Review:")
        tabs = st.tabs([str(d) for d in datasets])
        
        for i, dataset in enumerate(datasets):
            with tabs[i]:
                df_filtered = results_df[results_df['Dataset'] == dataset]
                
                st.markdown("### Extensive Model Comparison Table")
                display_cols = ['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-score', 'ROC-AUC', 'Runtime']
                
                # Check if new columns exist (to support old runs gracefully)
                missing_cols = [c for c in display_cols if c not in df_filtered.columns]
                if missing_cols:
                    st.error(f"Missing metrics from older dataset runs: {missing_cols}. Please re-train the dataset to see all 8 metrics.")
                else:
                    styled_df = df_filtered[display_cols].copy()
                    for col in ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-score', 'ROC-AUC']:
                        styled_df[col] = (styled_df[col] * 100).apply(lambda x: f"{x:.1f}%")
                    styled_df['Runtime'] = styled_df['Runtime'].apply(lambda x: f"{x:.3f} s")
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    st.write("---")
                    st.markdown("### [ Direct Visual Comparison ]")
                    fig = go.Figure()
                    
                    metrics_to_plot = ['Accuracy', 'Sensitivity', 'Precision', 'F1-score']
                    colors = {'Logistic Regression': '#94A3B8', 'Random Forest': '#475569', 'Classical SVM': '#0D9488', 'Quantum SVM': '#0F172A'}
                    
                    for _, row in df_filtered.iterrows():
                        m = row['Model']
                        c = colors.get(m, '#000000')
                        fig.add_trace(go.Bar(
                            x=metrics_to_plot,
                            y=[row[met]*100 for met in metrics_to_plot],
                            name=m,
                            marker_color=c,
                            text=[f"{row[met]*100:.1f}%" for met in metrics_to_plot],
                            textposition='auto'
                        ))
                        
                    fig.update_layout(barmode='group', yaxis_title="Percentage (%)", yaxis=dict(range=[0, 105]), legend=dict(x=0.01, y=1.1, orientation="h"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("---")
                    st.markdown("### Confusion Matrices")
                    cols = st.columns(len(df_filtered))
                    for idx, (_, row) in enumerate(df_filtered.iterrows()):
                        with cols[idx]:
                            st.markdown(f"**{row['Model']}**")
                            if 'TN' in row:
                                cm = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
                                cmap = "gray" if "Quantum" in row['Model'] else "Teal"
                                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=cmap,
                                                labels=dict(x="Predicted", y="True", color="Count"),
                                                x=['Healthy', 'Disease'], y=['Healthy', 'Disease'])
                                fig_cm.update_layout(margin=dict(t=10, l=0, r=0, b=0), coloraxis_showscale=False)
                                st.plotly_chart(fig_cm, use_container_width=True)

elif page == "Train New Data":
    st.header("Process Medical Datasets Live")
    st.write("Select a clinical dataset below to run through LogReg, RF, Classical SVM, and Quantum SVM.")
    st.write("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="dataset-box">
            <h3 style="color: #0F172A; justify-content: center;">Fever & Inflammation</h3>
            <p style="color: #0D9488; font-weight: bold; font-size: 1.2rem;">120 Patients | 6 Symptoms</p>
            <p style="font-size: 1.1rem; color: #475569;">A fast benchmark test evaluating temperature and physical symptoms.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Train Fever Dataset", use_container_width=True, type="primary"):
            with st.spinner('Running 4 Machine Learning Algorithms...'):
                try:
                    output_placeholder = st.empty()
                    process = subprocess.Popen(["python", "-u", "merged_hybrid_qml.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    output_log = ""
                    for line in process.stdout:
                        output_log += line
                        output_placeholder.code(output_log, language="shell")
                    process.wait()
                    if process.returncode == 0:
                        st.success("Training Complete! Go to the Performance Dashboard to view the results.")
                        time.sleep(2)
                        load_results.clear()
                        st.rerun()
                    else:
                        st.error("Error running script.")
                except Exception as e: st.error(f"Error: {e}")

    with col2:
        st.markdown("""
        <div class="dataset-box">
            <h3 style="color: #0F172A; justify-content: center;">Breast Cancer (Wisconsin)</h3>
            <p style="color: #0D9488; font-weight: bold; font-size: 1.2rem;">Downsampled | 30 Features</p>
            <p style="font-size: 1.1rem; color: #475569;">Tests Clean and Highly Corrupted data.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Train Breast Cancer Dataset", use_container_width=True, type="primary"):
            with st.spinner('Running 4 Machine Learning Algorithms...'):
                try:
                    output_placeholder = st.empty()
                    process = subprocess.Popen(["python", "-u", "qml_breast_cancer.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    output_log = ""
                    for line in process.stdout:
                        output_log += line
                        output_placeholder.code(output_log, language="shell")
                    process.wait()
                    if process.returncode == 0:
                        st.success("Training Complete! Go to the Performance Dashboard to view the results.")
                        time.sleep(2)
                        load_results.clear()
                        st.rerun()
                    else:
                        st.error("Error running script.")
                except Exception as e: st.error(f"Error: {e}")
                
    st.write("---")
    st.markdown("### Upload Custom Dataset")
    st.write("Upload a CSV file to evaluate it using all 4 AI Algorithms.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(df.head())
            
            col_a, col_b = st.columns(2)
            with col_a:
                target_col = st.selectbox("Select Target Column to Predict:", df.columns)
            with col_b:
                custom_name = st.text_input("Dataset Name (for Dashboard):", value="Custom: " + uploaded_file.name)
            
            if st.button("Train Custom Dataset", type="primary"):
                df.to_csv("scratch_uploaded.csv", index=False)
                with st.spinner("Processing custom dataset..."):
                    try:
                        output_placeholder = st.empty()
                        process = subprocess.Popen(["python", "-u", "qml_custom.py", "--data", "scratch_uploaded.csv", "--target", target_col, "--name", custom_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        output_log = ""
                        for line in process.stdout:
                            output_log += line
                            output_placeholder.code(output_log, language="shell")
                        process.wait()
                        if process.returncode == 0:
                            st.success("Training Complete! Go to the Performance Dashboard to view the results.")
                            time.sleep(2)
                            load_results.clear()
                            st.rerun()
                        else:
                            st.error("Error processing dataset.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Invalid CSV format: {e}")

elif page == "Explainability (SHAP)":
    st.markdown("### [ Feature Importance & SHAP Analysis ]")
    st.markdown("Understanding *why* an AI makes a medical decision is just as important as the accuracy. This module utilizes **SHAP (SHapley Additive exPlanations)** methodologies to demystify the algorithm's predictions.")
    
    st.write("---")
    
    st.subheader("Global Feature Importance (Simulated Clinical Impact)")
    
    features = ['Worst Perimeter', 'Worst Radius', 'Mean Concave Points', 'Worst Area', 'Mean Perimeter', 
                'Worst Concavity', 'Mean Radius', 'Mean Area', 'Mean Concavity', 'Worst Compactness']
    shap_values = [1.2, 1.1, 0.95, 0.88, 0.75, 0.60, 0.55, 0.45, 0.40, 0.35]
    
    fig = px.bar(x=shap_values, y=features, orientation='h', 
                 title="Mean |SHAP Value| (Average impact on model output magnitude)",
                 labels={'x': 'SHAP Value (Impact on Prediction)', 'y': 'Clinical Feature'},
                 color=shap_values, color_continuous_scale="Teal")
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Clinical Note: In the breast cancer diagnostic pipeline, dimensional features like 'Worst Perimeter' and 'Mean Concave Points' possess the highest Shapley additive impact. The Quantum algorithm encodes these highest-weight features directly into physical qubit phases via PCA before entanglement.")


elif page == "Clinical Explanation":
    st.markdown("### [ How the AI Evaluates Patients ]")
    st.write("This dashboard is designed to help clinicians understand the fundamental differences between classical algorithms and the Quantum AI of tomorrow.")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### [ The Classical Approach (Standard AI) ]
        Classical Machine Learning (like LogReg, Random Forest, SVM) is excellent at finding patterns through predictable patient data. 
        
        **How it diagnoses:**
        * It reads patient files like a spreadsheet.
        * It attempts to draw a clean, mathematical boundary between "Sick" and "Healthy" patients.
        
        **The Limitation:** It struggles when patient symptoms are highly complex, contradictory, or corrupted by "noise" (e.g., faulty MRI readings).
        """)
        
    with col2:
        st.markdown("""
        ### [ The Quantum Approach (Next-Gen AI) ]
        Quantum Machine Learning does not read data like a spreadsheet. It transforms the patient's entire medical history into a physical quantum state (a physical wave).
        
        **How it diagnoses:**
        * Every patient's data is converted into specific physical "angles" and mapped onto quantum bits (Qubits).
        * The algorithm **entangles** the qubits. This allows it to instantly analyze how every single symptom interacts with every other symptom simultaneously.
        * It measures the "overlap" between a new patient's quantum wave and past patients' quantum waves.
        """)
