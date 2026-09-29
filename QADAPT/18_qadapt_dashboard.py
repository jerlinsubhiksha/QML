import streamlit as st

st.set_page_config(
    page_title="QADAPT",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================
# QADAPT DESIGN SYSTEM
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700&display=swap');

:root {
    --burgundy: #790D16;
    --ivory: #F5EFE1;
    --champagne: #E5D3AF;
    --blue: #AEC4D4;
    --ink: #241D1D;
    --muted: #756F6B;
    --white: #FFFFFF;
}

.stApp {
    background: var(--ivory);
    color: var(--ink);
    font-family: "DM Sans", sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 1180px;
    padding: 42px 48px 70px;
}

.q-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 72px;
}

.q-brand {
    font-family: "Manrope", sans-serif;
    font-size: 25px;
    font-weight: 700;
    letter-spacing: -0.8px;
    color: var(--burgundy);
}

.q-doctor {
    display: flex;
    align-items: center;
    gap: 11px;
    color: var(--muted);
    font-size: 14px;
}

.q-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--champagne);
    color: var(--burgundy);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
}

.q-eyebrow {
    color: var(--burgundy);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 13px;
}

.q-title {
    font-family: "Manrope", sans-serif;
    font-size: 47px;
    line-height: 1.08;
    letter-spacing: -2px;
    font-weight: 600;
    margin: 0;
    color: var(--ink);
}

.q-subtitle {
    max-width: 650px;
    margin-top: 16px;
    color: var(--muted);
    font-size: 17px;
    line-height: 1.65;
}

.q-panel {
    margin-top: 52px;
    background: var(--white);
    border: 1px solid rgba(121,13,22,.10);
    border-radius: 26px;
    padding: 36px;
    box-shadow: 0 18px 55px rgba(60,34,24,.07);
}

.q-panel-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 25px;
    margin-bottom: 30px;
}

.q-panel-title {
    font-family: "Manrope", sans-serif;
    font-size: 24px;
    font-weight: 700;
}

.q-panel-text {
    margin-top: 7px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.5;
}

.q-step {
    background: var(--ivory);
    color: var(--burgundy);
    border-radius: 999px;
    padding: 8px 13px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
}

div[data-testid="stButton"] > button {
    border-radius: 18px !important;
    min-height: 132px !important;
    text-align: left !important;
    padding: 22px 24px !important;
    font-family: "DM Sans", sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(121,13,22,.13) !important;
    background: #fff !important;
    color: var(--ink) !important;
    box-shadow: none !important;
    transition: all .18s ease !important;
}

div[data-testid="stButton"] > button:hover {
    border-color: var(--burgundy) !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(60,34,24,.08) !important;
}

.q-section {
    margin-top: 56px;
}

.q-section-title {
    font-family: "Manrope", sans-serif;
    font-size: 19px;
    font-weight: 700;
}

.q-section-copy {
    color: var(--muted);
    font-size: 13px;
    margin-top: 5px;
}

.q-empty {
    margin-top: 18px;
    border: 1px dashed rgba(121,13,22,.20);
    border-radius: 18px;
    padding: 23px;
    color: var(--muted);
    background: rgba(255,255,255,.38);
    font-size: 13px;
}

.stFormSubmitButton button {
    background: var(--burgundy) !important;
    color: white !important;
}

@media (max-width: 800px) {
    .block-container { padding: 28px 20px 50px; }
    .q-topbar { margin-bottom: 48px; }
    .q-title { font-size: 36px; }
    .q-panel { padding: 24px; }
}
</style>
""", unsafe_allow_html=True)

if "qadapt_screen" not in st.session_state:
    st.session_state.qadapt_screen = "dashboard"

st.markdown("""
<div class="q-topbar">
    <div class="q-brand">qadapt</div>
    <div class="q-doctor">
        <div class="q-avatar">DR</div>
        <span>Doctor</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================
# DASHBOARD
# =============================
if st.session_state.qadapt_screen == "dashboard":

    st.markdown("""
    <div class="q-eyebrow">Cognitive health assessment</div>
    <div class="q-title">Good morning, Doctor.</div>
    <div class="q-subtitle">
        Begin a new patient assessment by providing the patient's existing
        information or entering the assessment details directly.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="q-panel">
        <div class="q-panel-head">
            <div>
                <div class="q-panel-title">Start a patient assessment</div>
                <div class="q-panel-text">
                    Choose how you'd like to provide the patient's information.
                </div>
            </div>
            <div class="q-step">STEP 01</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div style="background:#F5EFE1;border-radius:18px;padding:22px 24px 8px;margin-bottom:-126px;">
            <div style="color:#790D16;font-size:20px;margin-bottom:13px;">↑</div>
            <div style="font-family:Manrope,sans-serif;font-size:19px;font-weight:700;">Upload patient record</div>
            <div style="color:#756F6B;font-size:13px;line-height:1.5;margin-top:7px;">
                Use an existing patient file and review the information before analysis.
                <br><br><b>CSV · XLSX</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Upload patient record  →", key="upload", use_container_width=True):
            st.session_state.qadapt_screen = "upload"
            st.rerun()

    with c2:
        st.markdown("""
        <div style="background:#AEC4D4;border-radius:18px;padding:22px 24px 8px;margin-bottom:-126px;">
            <div style="color:#790D16;font-size:20px;margin-bottom:13px;">＋</div>
            <div style="font-family:Manrope,sans-serif;font-size:19px;font-weight:700;">Enter patient details</div>
            <div style="color:#756F6B;font-size:13px;line-height:1.5;margin-top:7px;">
                Enter the patient's assessment information directly into QADAPT.
                <br><br><b>Guided assessment</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Enter patient details  →", key="manual", use_container_width=True):
            st.session_state.qadapt_screen = "manual"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="q-section">
        <div class="q-section-title">Recent assessments</div>
        <div class="q-section-copy">Patient assessments completed in this session.</div>
        <div class="q-empty">No assessments yet. Start a patient assessment above.</div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# UPLOAD
# =============================
elif st.session_state.qadapt_screen == "upload":

    st.markdown("""
    <div class="q-eyebrow">New assessment</div>
    <div class="q-title">Upload patient record.</div>
    <div class="q-subtitle">
        Upload the patient's existing data. QADAPT will prepare the available
        information for review before analysis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Patient record",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    if uploaded:
        st.success(f"Record received: {uploaded.name}")
        st.info("Review and field mapping will appear here before analysis.")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("← Back", key="back_upload"):
            st.session_state.qadapt_screen = "dashboard"
            st.rerun()

    with c2:
        if uploaded:
            if st.button("Continue to review →", key="review_upload"):
                st.info("The review step will be connected to the V12 inference engine next.")

# =============================
# MANUAL ENTRY
# =============================
elif st.session_state.qadapt_screen == "manual":

    st.markdown("""
    <div class="q-eyebrow">New assessment</div>
    <div class="q-title">Enter patient details.</div>
    <div class="q-subtitle">
        We'll guide you through the patient's cognitive, functional and relevant
        health information.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    with st.form("patient_details"):

        st.markdown("### Patient information")
        age = st.number_input("Age", min_value=18, max_value=120, value=65)
        education = st.selectbox(
            "Education level", [1, 2, 3, 4],
            help="Use the category definitions from the trained dataset."
        )

        st.markdown("### Cognitive assessment")
        mmse = st.number_input("MMSE score", min_value=0.0, max_value=30.0, value=27.0)
        memory = st.radio("Memory complaints", ["No", "Yes"], horizontal=True)
        behavioral = st.radio("Behavioral problems", ["No", "Yes"], horizontal=True)
        personality = st.radio("Personality changes", ["No", "Yes"], horizontal=True)

        st.markdown("### Functional assessment")
        functional = st.number_input("Functional assessment", 0.0, 10.0, 8.0)
        adl = st.number_input("ADL score", 0.0, 10.0, 8.0)

        st.markdown("### Health factors")
        sleep = st.number_input("Sleep quality", 0.0, 10.0, 7.0)
        cardiovascular = st.radio("Cardiovascular disease", ["No", "Yes"], horizontal=True)
        systolic = st.number_input("Systolic blood pressure", 70.0, 250.0, 125.0)
        hdl = st.number_input("HDL cholesterol", 10.0, 150.0, 55.0)
        triglycerides = st.number_input("Triglycerides", 20.0, 1000.0, 120.0)

        submitted = st.form_submit_button(
            "Review patient information →",
            use_container_width=True
        )

    if submitted:
        st.session_state.patient_form = {
            "Age": age,
            "EducationLevel": education,
            "MMSE": mmse,
            "MemoryComplaints": int(memory == "Yes"),
            "BehavioralProblems": int(behavioral == "Yes"),
            "PersonalityChanges": int(personality == "Yes"),
            "FunctionalAssessment": functional,
            "ADL": adl,
            "SleepQuality": sleep,
            "CardiovascularDisease": int(cardiovascular == "Yes"),
            "SystolicBP": systolic,
            "CholesterolHDL": hdl,
            "CholesterolTriglycerides": triglycerides,
        }
        st.success("Patient information captured. Review can be connected to the model next.")

    if st.button("← Back", key="back_manual"):
        st.session_state.qadapt_screen = "dashboard"
        st.rerun()
