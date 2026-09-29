import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve
import os

st.set_page_config(page_title="Quantum Genomics Platform", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Courier New', Courier, monospace; }
    .metric-card { 
        background: #161b22; border: 1px solid #30363d; 
        padding: 20px; border-radius: 8px; text-align: center;
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #3fb950; }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Quantum-Classical Hybrid Genomic Predictor")
st.markdown("### Lung Cancer (OVA) Gene Expression Inference")
st.write("---")

results_path = os.path.join("..", "..", "QADAPT", "12_v12_final_results.csv")
preds_path = os.path.join("..", "..", "QADAPT", "12_v12_predictions.csv")

if not os.path.exists(results_path):
    st.error("Results not found. Please run the training script first.")
    st.stop()

df_res = pd.read_csv(results_path)
df_preds = pd.read_csv(preds_path)

c_res = df_res[df_res["model"] == "Classical_12"].iloc[0]
q_res = df_res[df_res["model"] == "Residual_Hybrid_V12"].iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><h3>Classical SVM</h3><div class="metric-value">{c_res["accuracy"]*100:.2f}%</div>Accuracy</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><h3>Hybrid QML</h3><div class="metric-value" style="color: #bc8cff;">{q_res["accuracy"]*100:.2f}%</div>Accuracy</div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><h3>Quantum Sensitivity</h3><div class="metric-value">{q_res["sensitivity"]*100:.2f}%</div>True Positive Rate</div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><h3>Dimensionality</h3><div class="metric-value">10,935 ➡️ 4</div>Qubit Entanglement</div>', unsafe_allow_html=True)

st.write("---")

colA, colB = st.columns(2)

with colA:
    st.markdown("### ROC-AUC Curves")
    fig = go.Figure()
    
    # Classical
    fpr_c, tpr_c, _ = roc_curve(df_preds["y_true"], df_preds["prob_Classical_12"])
    fig.add_trace(go.Scatter(x=fpr_c, y=tpr_c, name=f'Classical (AUC={c_res["auc"]:.3f})', line=dict(color='#3fb950', width=3)))
    
    # Quantum
    fpr_q, tpr_q, _ = roc_curve(df_preds["y_true"], df_preds["prob_Residual_Hybrid_V12"])
    fig.add_trace(go.Scatter(x=fpr_q, y=tpr_q, name=f'Hybrid QML (AUC={q_res["auc"]:.3f})', line=dict(color='#bc8cff', width=3)))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash='dash', color='#8b949e')))
    fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#161b22', font_color='#c9d1d9', xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

with colB:
    st.markdown("### Confusion Matrix (Hybrid QML)")
    z = [[q_res['tn'], q_res['fp']], [q_res['fn'], q_res['tp']]]
    fig_cm = px.imshow(z, text_auto=True, color_continuous_scale='Purples', 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Other', 'Lung Cancer'], y=['Other', 'Lung Cancer'])
    fig_cm.update_layout(paper_bgcolor='#0d1117', font_color='#c9d1d9')
    st.plotly_chart(fig_cm, use_container_width=True)

st.write("---")
st.markdown("""
### Quantum Pre-processing Pipeline
1. **Raw Data:** 1,545 patients × 10,935 Gene Expressions.
2. **Classical Bottleneck:** `SelectKBest` isolates the top 12 most mathematically significant genes.
3. **Quantum Encoding:** The top 4 genes are injected via Angle-Encoding ($R_Y, R_Z$ gates) into 4 Qubits.
4. **Entanglement:** A ring topology of $CNOT$ gates maps the non-linear biological epistatic relationships.
5. **Residual Correction:** Qiskit Statevector Simulator calibrates expectations into the classical logit curve.
""")
