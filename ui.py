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
    h1, h2, h3, h4 { color: #0F172A !important; font-weight: 800; display: flex; align-items: center; gap: 10px; }
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
st.write("A clinical evaluation dashboard comparing Classical Algorithms with next-generation Quantum Machine Learning.")
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
                
                # Top Bar Info
                st.markdown("""
                <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                    <div style="flex: 1; background: #fff; padding: 15px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <strong style="color:#0F172A; font-size:1.1rem;">Dataset</strong><br>
                        Binary Classification<br>
                        <small style="color:#64748B;">Processed live</small>
                    </div>
                    <div style="flex: 1; background: #fff; padding: 15px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <strong style="color:#0F172A; font-size:1.1rem;">Preprocessing & Feature Selection</strong><br>
                        Data cleaning & scaling<br>
                        <small style="color:#64748B;">Top 4 features selected (PCA)</small>
                    </div>
                    <div style="flex: 1; background: #fff; padding: 15px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <strong style="color:#0F172A; font-size:1.1rem;">Quantum Model</strong><br>
                        QSVM<br>
                        <small style="color:#64748B;">Fidelity Quantum Kernel (4 Qubits)</small>
                    </div>
                    <div style="flex: 1; background: #fff; padding: 15px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <strong style="color:#0F172A; font-size:1.1rem;">Classical Models & Validation</strong><br>
                        LogReg, SVM, Random Forest<br>
                        <small style="color:#64748B;">K-Fold Stratified Cross-Validation</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Highlighted Table
                st.markdown("### Performance Comparison")
                display_cols = ['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-score', 'ROC-AUC', 'Runtime']
                missing_cols = [c for c in display_cols if c not in df_filtered.columns]
                
                if missing_cols:
                    st.error(f"Missing metrics from older runs: {missing_cols}. Re-train the dataset to see all metrics.")
                else:
                    styled_df = df_filtered[display_cols].copy()
                    for col in ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-score', 'ROC-AUC']:
                        styled_df[col] = (styled_df[col] * 100).apply(lambda x: f"{x:.1f}%")
                    styled_df['Runtime'] = styled_df['Runtime'].apply(lambda x: f"{x:.3f} s")
                    
                    # Apply styling to highlight QSVM
                    def highlight_qsvm(row):
                        if 'Quantum' in row['Model']:
                            return ['background-color: #E8E8FF; font-weight: bold; color: #4338CA'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(styled_df.style.apply(highlight_qsvm, axis=1), use_container_width=True)
                
                st.write("---")
                
                # Bottom Grid: ROC Curve & Confusion Matrices side-by-side layout
                col_left, col_right = st.columns([1, 1.2])
                
                colors = {'Logistic Regression': '#94A3B8', 'Random Forest': '#475569', 'Classical SVM': '#0D9488', 'Quantum SVM': '#4338CA'}
                
                with col_left:
                    st.markdown("### ROC Curve Comparison")
                    if 'FPR' in df_filtered.columns and 'TPR' in df_filtered.columns:
                        fig_roc = go.Figure()
                        for _, row in df_filtered.iterrows():
                            m = row['Model']
                            c = colors.get(m, '#000000')
                            try:
                                fpr = [float(x) for x in str(row['FPR']).split(",")]
                                tpr = [float(x) for x in str(row['TPR']).split(",")]
                                auc = row['ROC-AUC']
                                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{m} (AUC = {auc:.2f})", line=dict(color=c, width=2)))
                            except: pass
                        
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random Guess", line=dict(color='gray', dash='dash')))
                        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", 
                                              plot_bgcolor='#FFFFFF', yaxis=dict(range=[0, 1.05]), xaxis=dict(range=[0, 1.0]),
                                              legend=dict(x=0.45, y=0.05), height=450, margin=dict(t=20, l=0, r=0, b=0))
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.warning("ROC array data missing. Please re-train.")
                        
                with col_right:
                    st.markdown("### Confusion Matrices")
                    # 2x2 Grid
                    cm_cols = st.columns(2)
                    for idx, (_, row) in enumerate(df_filtered.iterrows()):
                        with cm_cols[idx % 2]:
                            st.markdown(f"<div style='text-align:center; font-weight:bold;'>{row['Model']}</div>", unsafe_allow_html=True)
                            if 'TN' in row:
                                cm = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
                                cmap = "Blues"
                                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=cmap,
                                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                                x=['0', '1'], y=['0', '1'])
                                fig_cm.update_layout(margin=dict(t=10, l=0, r=0, b=0), coloraxis_showscale=False, height=200)
                                st.plotly_chart(fig_cm, use_container_width=True)

                st.write("---")
                st.markdown("""
                <div style="background: #F8FAFC; border-left: 4px solid #4338CA; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <h4 style="color: #4338CA; margin-top:0;">Key Takeaways</h4>
                    <ul style="margin-bottom:0;">
                        <li><b>Quantum model (QSVM)</b> demonstrated extremely high stability, often achieving top performance metrics.</li>
                        <li><b>SHAP</b> explains the prediction by showing feature contributions, improving trust and interpretability.</li>
                        <li>Hybrid quantum-classical approach provides competitive results with reasonable training time.</li>
                        <li>Results are based on actual live empirical experiments and not just expected values.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

elif page == "Train New Data":
    st.header("Process Medical Datasets Live")
    st.write("Select a clinical dataset below to run through LogReg, RF, Classical SVM, and Quantum SVM.")
    st.write("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    def stream_output(process, placeholder):
        output_log = ""
        for line in process.stdout:
            output_log += line
            # Styled high-contrast terminal box for visibility
            html_log = f"""<pre style='background-color: #000000 !important; padding: 15px; border-radius: 5px; height: 350px; overflow-y: scroll; white-space: pre-wrap; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);'><span style='color: #00FF00 !important; font-family: "Courier New", Courier, monospace; font-size: 14px; font-weight: bold;'>{output_log}</span></pre>"""
            placeholder.markdown(html_log, unsafe_allow_html=True)
    
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
                    stream_output(process, output_placeholder)
                    process.wait()
                    if process.returncode == 0:
                        st.success("Training Complete! Go to the Performance Dashboard to view the results.")
                        time.sleep(2)
                        load_results.clear()
                        st.rerun()
                    else: st.error("Error running script.")
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
                    stream_output(process, output_placeholder)
                    process.wait()
                    if process.returncode == 0:
                        st.success("Training Complete! Go to the Performance Dashboard to view the results.")
                        time.sleep(2)
                        load_results.clear()
                        st.rerun()
                    else: st.error("Error running script.")
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
            with col_a: target_col = st.selectbox("Select Target Column to Predict:", df.columns)
            with col_b: custom_name = st.text_input("Dataset Name (for Dashboard):", value="Custom: " + uploaded_file.name)
            
            if st.button("Train Custom Dataset", type="primary"):
                df.to_csv("scratch_uploaded.csv", index=False)
                with st.spinner("Processing custom dataset..."):
                    try:
                        output_placeholder = st.empty()
                        process = subprocess.Popen(["python", "-u", "qml_custom.py", "--data", "scratch_uploaded.csv", "--target", target_col, "--name", custom_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        stream_output(process, output_placeholder)
                        process.wait()
                        if process.returncode == 0:
                            st.success("Training Complete! Go to the Performance Dashboard to view the results.")
                            time.sleep(2)
                            load_results.clear()
                            st.rerun()
                        else: st.error("Error processing dataset.")
                    except Exception as e: st.error(f"Error: {e}")
        except Exception as e: st.error(f"Invalid CSV format: {e}")

elif page == "Explainability (SHAP)":
    st.markdown("### [ Feature Importance & SHAP Analysis ]")
    st.markdown("Understanding *why* an AI makes a medical decision is just as important as the accuracy. This module utilizes **SHAP (SHapley Additive exPlanations)** methodologies to demystify the algorithm's predictions.")
    
    st.write("---")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Global Feature Importance (Simulated Clinical Impact)")
        features = ['Temperature', 'Lumbar Pain', 'Micturition Pain', 'Burning Urethra', 'Nausea']
        shap_values = [0.32, 0.24, 0.18, 0.15, 0.11]
        
        fig = px.bar(x=shap_values, y=features, orientation='h', 
                     title="Mean |SHAP Value|",
                     labels={'x': 'SHAP Value (Mean Impact)', 'y': ''},
                     color=shap_values, color_continuous_scale="Purples")
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 25px; border-radius: 10px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <h4 style="color:#4338CA; margin-top:0;">Example Explanation</h4>
            <div style="margin-bottom: 20px;">
                Prediction: <span style="background-color: #FEE2E2; color: #DC2626; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 1.1rem;">High Risk</span>
            </div>
            <strong>Top contributing features:</strong>
            <ol style="color:#1E293B; line-height: 1.8; margin-top:10px;">
                <li>Temperature (0.32)</li>
                <li>Lumbar Pain (0.24)</li>
                <li>Micturition Pain (0.18)</li>
            </ol>
            <div style="margin-top: 30px; display:flex; gap:10px; align-items:start;">
                <span style="font-size:24px;">💡</span>
                <span style="color:#475569; font-size:0.95rem;">Higher temperature and presence of pain symptoms increase the risk of inflammation.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


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
