import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Quantum Disease Predictor", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR HIGH CONTRAST & HIDING TASK BAR ---
st.markdown("""
<style>
    /* 1. Hide Streamlit top black task bar and hamburger menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 2. Main Background for clean medical look */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* 3. High contrast text (Dark Slate) */
    h1, h2, h3 {
        color: #0F172A !important; 
        font-weight: 800;
    }
    p, span, div, li {
        color: #1E293B;
    }
    
    /* 4. Highlight Teal Accent */
    .teal-text {
        color: #0D9488;
    }

    /* 5. Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border-top: 8px solid #0D9488;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: -5px;
    }
    .metric-label {
        color: #64748B;
        font-size: 1.1rem;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .metric-value-sub {
        font-size: 2.2rem;
        color: #0D9488;
        font-weight: 800;
    }
    
    /* 6. Sidebar styling for better contrast */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #E2E8F0;
    }
    
    /* Dataset Box styling for visibility */
    .dataset-box {
        background-color: #FFFFFF; 
        padding: 30px; 
        border-radius: 15px; 
        border: 2px solid #0D9488; 
        text-align: center;
        height: 250px;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.1);
    }
    .dataset-box h3 {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1><span class='teal-text'>⚕️ Quantum AI</span> Medical Diagnostics</h1>", unsafe_allow_html=True)
st.write("A clinical evaluation dashboard comparing Classical Support Vector Machines with next-generation Quantum Machine Learning.")
st.write("---")

# Sidebar - Using a Medical/AI cross logo
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
st.sidebar.title("Clinical Dashboard")
page = st.sidebar.radio("", ["📊 Performance Dashboard", "⚙️ Train New Data", "🧠 Clinical Explanation"])

# Load Data Function - Reading from master CSV so datasets accumulate
@st.cache_data
def load_results():
    try:
        df = pd.read_csv("master_qml_results.csv")
        # Keep only the latest run for each dataset and model combination to avoid duplicates
        return df.drop_duplicates(subset=['Dataset', 'Model'], keep='last')
    except:
        return None

results_df = load_results()

# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
if page == "📊 Performance Dashboard":
    if results_df is None:
        st.warning("⚠️ No clinical data processed yet. Please go to 'Train New Data' to process a dataset.")
    else:
        # User Friendly Tabs
        datasets = results_df['Dataset'].unique()
        st.markdown("### Choose Dataset to Review:")
        tabs = st.tabs([str(d) for d in datasets])
        
        for i, dataset in enumerate(datasets):
            with tabs[i]:
                df_filtered = results_df[results_df['Dataset'] == dataset]
                
                c_data = df_filtered[df_filtered['Model'] == 'Classical SVM'].iloc[0] if not df_filtered[df_filtered['Model'] == 'Classical SVM'].empty else None
                q_data = df_filtered[df_filtered['Model'] == 'Quantum SVM'].iloc[0] if not df_filtered[df_filtered['Model'] == 'Quantum SVM'].empty else None
                
                if c_data is not None and q_data is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h3 style='text-align: center; color: #1E293B;'>💻 Classical SVM (Standard)</h3>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Overall Accuracy</div>
                            <div class="metric-value">{c_data['Accuracy']*100:.1f}%</div>
                            <hr style="border-top: 1px solid #E2E8F0; margin: 25px 0;">
                            <div class="metric-label">Sensitivity (True Positive Rate)</div>
                            <div class="metric-value-sub">{c_data['Sensitivity']*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown("<h3 style='text-align: center; color: #1E293B;'>⚛️ Quantum SVM (Next-Gen)</h3>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card" style="border-top-color: #0F172A;">
                            <div class="metric-label">Overall Accuracy</div>
                            <div class="metric-value">{q_data['Accuracy']*100:.1f}%</div>
                            <hr style="border-top: 1px solid #E2E8F0; margin: 25px 0;">
                            <div class="metric-label">Sensitivity (True Positive Rate)</div>
                            <div class="metric-value-sub">{q_data['Sensitivity']*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("---")
                    st.markdown("### 📈 Visual Comparison: Classical vs Quantum")
                    
                    # Professional Plotly Chart for easy reading
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=['Accuracy', 'Sensitivity', 'Precision'],
                        y=[c_data['Accuracy']*100, c_data['Sensitivity']*100, c_data['Precision']*100],
                        name='Classical SVM',
                        marker_color='#0D9488',
                        text=[f"{val*100:.1f}%" for val in [c_data['Accuracy'], c_data['Sensitivity'], c_data['Precision']]],
                        textposition='auto'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=['Accuracy', 'Sensitivity', 'Precision'],
                        y=[q_data['Accuracy']*100, q_data['Sensitivity']*100, q_data['Precision']*100],
                        name='Quantum SVM',
                        marker_color='#0F172A',
                        text=[f"{val*100:.1f}%" for val in [q_data['Accuracy'], q_data['Sensitivity'], q_data['Precision']]],
                        textposition='auto'
                    ))

                    fig.update_layout(
                        barmode='group',
                        xaxis_title="Clinical Metric",
                        yaxis_title="Percentage (%)",
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(range=[0, 105]),
                        legend=dict(x=0.8, y=1.1, orientation="h")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2: TRAIN MODELS 
# ==========================================
elif page == "⚙️ Train New Data":
    st.header("Process Medical Datasets Live")
    st.write("Select a clinical dataset below to run through both the standard classical algorithm and the quantum algorithm. **The datasets will accumulate in your dashboard so you can switch between them!**")
    
    st.write("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="dataset-box">
            <h3 style="color: #0F172A;">Fever & Inflammation</h3>
            <p style="color: #0D9488; font-weight: bold; font-size: 1.2rem;">120 Patients | 6 Symptoms</p>
            <p style="font-size: 1.1rem; color: #475569;">A fast benchmark test evaluating temperature and physical symptoms.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Train Fever Dataset 🚀", use_container_width=True, type="primary"):
            with st.spinner('Initializing Quantum simulation for Fever dataset...'):
                try:
                    # Stream output to UI
                    output_placeholder = st.empty()
                    process = subprocess.Popen(["python", "merged_hybrid_qml.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    output_log = ""
                    for line in process.stdout:
                        output_log += line
                        output_placeholder.code(output_log, language="shell")
                    
                    process.wait()
                    if process.returncode == 0:
                        st.balloons()
                        st.success("✅ Training Complete! Go to the Performance Dashboard to view the results.")
                        time.sleep(2)
                        load_results.clear()
                        st.rerun()
                    else:
                        st.error("Error running script.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.markdown("""
        <div class="dataset-box">
            <h3 style="color: #0F172A;">Breast Cancer (Wisconsin)</h3>
            <p style="color: #0D9488; font-weight: bold; font-size: 1.2rem;">Downsampled | 30 Features</p>
            <p style="font-size: 1.1rem; color: #475569;">Tests both Clean and Highly Corrupted data. Takes ~1 minute.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Train Breast Cancer Dataset 🚀", use_container_width=True, type="primary"):
            with st.spinner('Simulating complex Quantum entanglement...'):
                try:
                    # Stream output to UI
                    output_placeholder = st.empty()
                    process = subprocess.Popen(["python", "qml_breast_cancer.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    output_log = ""
                    for line in process.stdout:
                        output_log += line
                        output_placeholder.code(output_log, language="shell")
                        
                    process.wait()
                    if process.returncode == 0:
                        st.balloons()
                        st.success("✅ Training Complete! Go to the Performance Dashboard to view the results.")
                        time.sleep(2)
                        load_results.clear()
                        st.rerun()
                    else:
                        st.error("Error running script.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# PAGE 3: HOW IT WORKS (CLINICAL EXPLANATION)
# ==========================================
elif page == "🧠 Clinical Explanation":
    st.header("How the AI Evaluates Patients")
    st.write("This dashboard is designed to help clinicians understand the fundamental differences between the classical AI they use today, and the Quantum AI of tomorrow.")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 💻 The Classical Approach (Standard AI)
        Classical Machine Learning (like our SVM) is excellent at drawing straight lines through predictable patient data. 
        
        **How it diagnoses:**
        * It reads patient files like a spreadsheet.
        * It looks at symptoms (e.g., age, tumor size) and mathematical trends.
        * It attempts to draw a clean, mathematical boundary between "Sick" and "Healthy" patients.
        
        **The Limitation:** It struggles when patient symptoms are highly complex, contradictory, or corrupted by "noise" (e.g., faulty MRI readings).
        """)
        
    with col2:
        st.markdown("""
        ### ⚛️ The Quantum Approach (Next-Gen AI)
        Quantum Machine Learning does not read data like a spreadsheet. It transforms the patient's entire medical history into a physical quantum state (a physical wave).
        
        **How it diagnoses:**
        * Every patient's data is converted into specific physical "angles" and mapped onto quantum bits (Qubits).
        * The algorithm **entangles** the qubits. This allows it to instantly analyze how every single symptom interacts with every other symptom simultaneously.
        * It measures the "overlap" between a new patient's quantum wave and past patients' quantum waves.
        
        **The Advantage:** Because it looks at data multidimensionally, Quantum AI is theoretically much more resilient to "noisy" or corrupted patient records that would normally confuse standard AI.
        """)
