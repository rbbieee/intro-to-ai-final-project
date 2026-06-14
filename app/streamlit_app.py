import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# page config
st.set_page_config(
    page_title="GTD Severity Classifier",
    layout="wide"
)

# load models
@st.cache_resource
def load_models():
    rf = joblib.load("app/model/rf_model.pkl")
    le = joblib.load("app/model/label_encoder.pkl")
    sc = joblib.load("app/model/scaler.pkl")
    return rf, le, sc

rf_model, label_encoder, scaler = load_models()

# fuzzy functions
def trimf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9),
                                     (c - x) / (c - b + 1e-9)))

def trapmf(x, a, b, c, d):
    return np.maximum(0, np.minimum(
        np.minimum((x - a) / (b - a + 1e-9), 1),
        (d - x) / (d - c + 1e-9)
    ))

def fuzzify_nkill(val):
    x = np.array([val], dtype=float)
    return {
        "Low":     float(trapmf(x, 0, 0, 1, 4)[0]),
        "Medium":  float(trimf(x, 2, 6, 12)[0]),
        "High":    float(trimf(x, 6, 15, 30)[0]),
        "Extreme": float(trapmf(x, 25, 40, 50, 50)[0]),
    }

def fuzzify_nwound(val):
    x = np.array([val], dtype=float)
    return {
        "Low":     float(trapmf(x, 0, 0, 2, 6)[0]),
        "Medium":  float(trimf(x, 3, 10, 20)[0]),
        "High":    float(trimf(x, 15, 35, 60)[0]),
        "Extreme": float(trapmf(x, 45, 65, 80, 80)[0]),
    }

def fuzzify_propextent(val):
    x = np.array([val], dtype=float)
    return {
        "None":         float(trapmf(x, 0, 0, 0, 0.5)[0]),
        "Minor":        float(trimf(x, 0.5, 1, 1.5)[0]),
        "Major":        float(trimf(x, 1.5, 2, 2.5)[0]),
        "Catastrophic": float(trapmf(x, 2.5, 3, 3, 3)[0]),
    }

def fuzzify_attack(val):
    x = np.array([val], dtype=float)
    return {
        "Low":     float(trapmf(x, 1, 1, 1, 1.8)[0]),
        "Medium":  float(trimf(x, 1.5, 2, 2.5)[0]),
        "High":    float(trimf(x, 2.5, 3, 3.5)[0]),
        "Extreme": float(trapmf(x, 3.2, 3.6, 5, 5)[0]),
    }

def fuzzify_weapon(val):
    x = np.array([val], dtype=float)
    return {
        "Low":     float(trapmf(x, 1, 1, 1, 1.8)[0]),
        "Medium":  float(trimf(x, 1.5, 2, 2.5)[0]),
        "High":    float(trimf(x, 2.5, 3, 3.5)[0]),
        "Extreme": float(trapmf(x, 3.2, 3.6, 5, 5)[0]),
    }

rules = [
    ("Low",    "Low",    "None",         "Low",    "Low",    "Low"),
    ("Low",    "Low",    "None",         "Low",    "Medium", "Low"),
    ("Low",    "Low",    "Minor",        "Low",    "Low",    "Low"),
    ("Low",    "Low",    "Major",        "Low",    "Low",    "Medium"),
    ("Medium", "Low",    "None",         "Low",    "Low",    "Medium"),
    ("Medium", "Medium", "Minor",        "Medium", "Medium", "Medium"),
    ("Low",    "Medium", "Major",        "Low",    "Medium", "Medium"),
    ("Low",    "Low",    "None",         "High",   "High",   "Medium"),
    ("Low",    "Low",    "Minor",        "Medium", "High",   "Medium"),
    ("Low",    "Low",    "None",         "Medium", "Medium", "Medium"),
    ("Low",    "Low",    "None",         "High",   "Medium", "Medium"),
    ("Low",    "Low",    "None",         "Medium", "High",   "Medium"),
    ("Low",    "Low",    "Minor",        "High",   "Medium", "Medium"),
    ("Low",    "Low",    "None",         "Extreme","Medium", "Medium"),
    ("Low",    "Low",    "None",         "Medium", "Extreme","Medium"),
    ("Medium", "Medium", "Major",        "Medium", "High",   "High"),
    ("High",   "Low",    "Minor",        "High",   "Medium", "High"),
    ("High",   "Medium", "None",         "High",   "High",   "High"),
    ("Medium", "High",   "Major",        "Medium", "High",   "High"),
    ("High",   "High",   "Minor",        "High",   "Medium", "High"),
    ("Low",    "High",   "Major",        "Extreme","High",   "High"),
    ("Medium", "Low",    "Major",        "High",   "Extreme","High"),
    ("Low",    "Low",    "None",         "Extreme","Extreme","High"),
    ("Low",    "Low",    "Minor",        "Extreme","High",   "High"),
    ("Medium", "Low",    "None",         "Extreme","High",   "High"),
    ("Low",    "Medium", "None",         "Extreme","Extreme","High"),
    ("High",   "Medium", "Major",        "High",   "Extreme","Critical"),
    ("High",   "High",   "Major",        "Extreme","Extreme","Critical"),
    ("Extreme","High",   "Major",        "Extreme","High",   "Critical"),
    ("Extreme","Extreme","Catastrophic", "Extreme","Extreme","Critical"),
    ("High",   "High",   "Catastrophic", "High",   "Extreme","Critical"),
    ("Extreme","Medium", "Major",        "Extreme","High",   "Critical"),
    ("Extreme", "High",   "Major", "High", "High", "Critical"),
    ("Extreme", "Medium", "Major", "High", "High", "Critical"),
    ("High",    "High",   "Major", "High", "High", "Critical"),
    ("Extreme", "High",   "None",  "High", "High", "Critical"),
]

x_out = np.linspace(0, 100, 1000)

output_mf = {
    "Low":      trapmf(x_out, 0, 0, 15, 30),
    "Medium":   trimf(x_out, 20, 40, 55),
    "High":     trimf(x_out, 45, 60, 75),
    "Critical": trapmf(x_out, 65, 80, 100, 100),
}

crisp_output = {
    "Low":      12.5,
    "Medium":   37.5,
    "High":     62.5,
    "Critical": 87.5,
}

def mamdani_infer(nkill_val, nwound_val, prop_val, atk_val, wpn_val):
    fk  = fuzzify_nkill(nkill_val)
    fw  = fuzzify_nwound(nwound_val)
    fp  = fuzzify_propextent(prop_val)
    fa  = fuzzify_attack(atk_val)
    fwp = fuzzify_weapon(wpn_val)
    strengths  = np.array([
        min(fk[k], fw[w], fp[p], fa[a], fwp[wp])
        for (k, w, p, a, wp, out) in rules
    ])
    out_labels = [out for (_, _, _, _, _, out) in rules]
    out_matrix = np.array([output_mf[out] for out in out_labels])
    clipped    = np.minimum(strengths[:, np.newaxis], out_matrix)
    aggregated = clipped.max(axis=0)
    return aggregated

def defuzzify_centroid(aggregated):
    denom = np.sum(aggregated)
    if denom == 0:
        return 0.0
    return float(np.sum(x_out * aggregated) / denom)

def sugeno_infer(nkill_val, nwound_val, prop_val, atk_val, wpn_val):
    fk  = fuzzify_nkill(nkill_val)
    fw  = fuzzify_nwound(nwound_val)
    fp  = fuzzify_propextent(prop_val)
    fa  = fuzzify_attack(atk_val)
    fwp = fuzzify_weapon(wpn_val)
    numerator   = 0.0
    denominator = 0.0
    for (k, w, p, a, wp, out) in rules:
        strength     = min(fk[k], fw[w], fp[p], fa[a], fwp[wp])
        numerator   += strength * crisp_output[out]
        denominator += strength
    if denominator == 0:
        return 0.0
    return numerator / denominator

def score_to_label(score):
    if score < 25:
        return "Low"
    elif score < 50:
        return "Medium"
    elif score < 75:
        return "High"
    else:
        return "Critical"

def label_color(label):
    return {
        "Low":      "#2ecc71",
        "Medium":   "#f39c12",
        "High":     "#e67e22",
        "Critical": "#e74c3c",
    }.get(label, "#95a5a6")

# ui
st.title("GTD Terrorism Attack Severity Classifier")
st.markdown("**The Best Team Ever | IF-48-INT | DKA Tubes 2025/2026**")
st.markdown("Robbie Yudistira (103012400001) | Frederick Octo Ramadani (103012440019) | Rei Hashimoto (103012450001)")
st.markdown("Classify terrorism attack severity using Fuzzy Logic (Mamdani & Sugeno) and Random Forest.")
st.divider()

st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    nkill  = st.slider("Number of Fatalities (nkill)",  0, 50, 0)
    nwound = st.slider("Number of Injuries (nwound)",   0, 80, 0)

with col2:
    attack_type = st.selectbox("Attack Type", [
        "Facility/Infrastructure Attack (1)",
        "Assassination / Hijacking / Hostage Taking (2)",
        "Armed Assault (3)",
        "Bombing/Explosion (4)",
    ])
    weapon_type = st.selectbox("Weapon Type", [
        "Melee / Unknown / Sabotage (1)",
        "Incendiary (2)",
        "Firearms (3)",
        "Explosives / Chemical (4)",
    ])

with col3:
    prop_damage = st.selectbox("Property Damage", [
        "Unknown / None (0)",
        "Minor (1)",
        "Major (2)",
        "Catastrophic (3)",
    ])

attack_encoded = int(attack_type.split("(")[1].replace(")", ""))
weapon_encoded = int(weapon_type.split("(")[1].replace(")", ""))
prop_inverted  = int(prop_damage.split("(")[1].replace(")", ""))

st.divider()

if st.button("Classify", type="primary", use_container_width=True):

    with st.spinner("Running classification..."):

        # fuzzy
        agg           = mamdani_infer(nkill, nwound, prop_inverted, attack_encoded, weapon_encoded)
        mamdani_score = defuzzify_centroid(agg)
        mamdani_label = score_to_label(mamdani_score)

        sugeno_score  = sugeno_infer(nkill, nwound, prop_inverted, attack_encoded, weapon_encoded)
        sugeno_label  = score_to_label(sugeno_score)

        # ml
        X_input  = np.array([[nkill, nwound, prop_inverted, attack_encoded, weapon_encoded]])
        rf_label = rf_model.predict(X_input)[0]

    st.subheader("Classification Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Mamdani Fuzzy**")
        st.markdown(f"Score: `{mamdani_score:.2f}`")
        st.markdown(
            f"<div style='background:{label_color(mamdani_label)};padding:12px;border-radius:8px;"
            f"text-align:center;color:white;font-size:20px;font-weight:bold'>{mamdani_label}</div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown("**Sugeno Fuzzy**")
        st.markdown(f"Score: `{sugeno_score:.2f}`")
        st.markdown(
            f"<div style='background:{label_color(sugeno_label)};padding:12px;border-radius:8px;"
            f"text-align:center;color:white;font-size:20px;font-weight:bold'>{sugeno_label}</div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown("**Random Forest**")
        st.markdown("&nbsp;")
        st.markdown(
            f"<div style='background:{label_color(rf_label)};padding:12px;border-radius:8px;"
            f"text-align:center;color:white;font-size:20px;font-weight:bold'>{rf_label}</div>",
            unsafe_allow_html=True
        )

    with c4:
        st.markdown("**Neural Network**")
        st.markdown("&nbsp;")
        st.markdown(
            "<div style='background:#7f8c8d;padding:12px;border-radius:8px;"
            "text-align:center;color:white;font-size:14px'>"
            "See notebook 09 for DL results (96.73% accuracy)</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader("Mamdani Aggregated Output")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x_out, agg, color="#c0392b", linewidth=2, label="Aggregated output")
    ax.axvline(mamdani_score, color="black", linestyle="--", label=f"Centroid = {mamdani_score:.2f}")
    ax.fill_between(x_out, agg, alpha=0.2, color="#c0392b")
    ax.set_xlabel("Severity score")
    ax.set_ylabel("Membership degree")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.divider()

    st.subheader("Active Membership Degrees")

    fk  = fuzzify_nkill(nkill)
    fw  = fuzzify_nwound(nwound)
    fp  = fuzzify_propextent(prop_inverted)
    fa  = fuzzify_attack(attack_encoded)
    fwp = fuzzify_weapon(weapon_encoded)

    mf_data = {
        "nkill":          fk,
        "nwound":         fw,
        "propextent":     fp,
        "attack_encoded": fa,
        "weapon_encoded": fwp,
    }

    fig2, axes = plt.subplots(1, 5, figsize=(18, 3))
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

    for ax, (var, mf_vals) in zip(axes, mf_data.items()):
        ax.bar(mf_vals.keys(), mf_vals.values(), color=colors)
        ax.set_title(var)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Degree")

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()