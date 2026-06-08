import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
}

.result-success {
    background-color: #0f5132;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.result-danger {
    background-color: #842029;
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.metric-box {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_model.pkl")

# =========================
# HEADER
# =========================
st.title("⚙️ Predictive Maintenance System")

st.markdown("""
### Machine Failure Prediction using XGBoost

Aplikasi ini digunakan untuk memprediksi potensi kegagalan mesin berdasarkan parameter operasional.
""")

st.divider()

# =========================
# INPUT SECTION
# =========================

col1, col2 = st.columns(2)

with col1:

    product_type = st.selectbox(
        "Product Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        value=300.0,
        step=0.1
    )

    proc_temp = st.number_input(
        "Process Temperature (K)",
        value=310.0,
        step=0.1
    )

with col2:

    rpm = st.number_input(
        "Rotational Speed (rpm)",
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        value=50
    )

st.divider()

# =========================
# PREDICT BUTTON
# =========================

if st.button("🔍 Predict Failure Risk", use_container_width=True):

    # =========================
    # ENCODING
    # =========================

    type_map = {
        "L": 0,
        "M": 1,
        "H": 2
    }

    type_enc = type_map[product_type]

    # =========================
    # FEATURE ENGINEERING
    # =========================

    temp_diff = proc_temp - air_temp
    power = rpm * torque
    wear_speed = tool_wear * rpm
    torque_sq = torque ** 2

    # =========================
    # MODEL INPUT
    # =========================

    data = pd.DataFrame({
        "Type_enc": [type_enc],
        "Air_temp": [air_temp],
        "Proc_temp": [proc_temp],
        "Rot_speed": [rpm],
        "Torque": [torque],
        "Tool_wear": [tool_wear],
        "temp_diff": [temp_diff],
        "power": [power],
        "wear_speed": [wear_speed],
        "torque_sq": [torque_sq]
    })

    try:

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        # =========================
        # DERIVED FEATURES
        # =========================

        st.subheader("📊 Derived Features")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Temp Difference",
            f"{temp_diff:.2f}"
        )

        c2.metric(
            "Power",
            f"{power:,.0f}"
        )

        c3.metric(
            "Wear Speed",
            f"{wear_speed:,.0f}"
        )

        c4.metric(
            "Torque²",
            f"{torque_sq:.2f}"
        )

        st.divider()

        # =========================
        # RISK SCORE
        # =========================

        st.subheader("🎯 Risk Assessment")

        st.metric(
            "Failure Probability",
            f"{prob:.2%}"
        )

        st.progress(float(prob))

        st.write("Risk Score")

        # =========================
        # RESULT
        # =========================

        if pred == 1:

            st.markdown(
                f"""
                <div class="result-danger">
                ⚠️ HIGH FAILURE RISK
                <br><br>
                Probability : {prob:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-success">
                ✅ MACHINE NORMAL
                <br><br>
                Probability : {prob:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # =========================
        # RECOMMENDATION
        # =========================

        st.subheader("🛠 Maintenance Recommendation")

        if prob >= 0.80:

            st.error("""
• Segera lakukan inspeksi mesin

• Periksa kondisi tool wear

• Jadwalkan preventive maintenance

• Verifikasi kondisi sistem pendingin
""")

        elif prob >= 0.50:

            st.warning("""
• Lakukan monitoring berkala

• Jadwalkan pemeriksaan dalam waktu dekat

• Evaluasi parameter operasi mesin
""")

        else:

            st.success("""
• Mesin dalam kondisi normal

• Operasikan sesuai SOP

• Lanjutkan maintenance sesuai jadwal rutin
""")

    except Exception as e:

        st.error(f"Error saat prediksi: {e}")

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "Predictive Maintenance Dashboard | XGBoost Machine Learning Model"
)