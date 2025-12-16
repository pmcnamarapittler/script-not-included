import streamlit as st
import pandas as pd

# -----------------------------
# Adjustment Engine (RB v1)
# -----------------------------
def adjust_projection(rb: dict) -> dict:
    """
    Takes a single RB input dict and returns adjusted projection outputs.
    This is intentionally conservative + explainable.
    """
    name = rb["name"].strip() if rb["name"] else "Unnamed RB"
    baseline = float(rb["baseline"])

    # --- Trend adjustment ---
    trend = rb["trend"]
    trend_delta = {"Up": 1.5, "Flat": 0.0, "Down": -1.5}.get(trend, 0.0)

    # --- Context adjustments ---
    temp_f = rb["temperature_f"]
    surface = rb["surface"]
    short_week = rb["short_week"]
    return_status = rb["return_status"]

    surface_delta = -0.5 if surface == "Turf" else 0.0

    # Context flags influence confidence + range more than points
    flags = []
    if temp_f >= 90:
        flags.append("HOT GAME")
    if temp_f <= 35:
        flags.append("COLD GAME")
    if surface == "Turf":
        flags.append("TURF")
    if short_week:
        flags.append("SHORT WEEK")
    if return_status:
        flags.append("RETURN STATUS")

    # Small point deltas (conservative)
    adjusted = baseline + trend_delta + surface_delta

    # --- Range logic ---
    # Start with +/- 20% and widen under more uncertainty
    base_range_pct = 0.20
    if len(flags) >= 2:
        base_range_pct = 0.30
    elif len(flags) == 1:
        base_range_pct = 0.25

    range_low = max(0.0, adjusted * (1 - base_range_pct))
    range_high = adjusted * (1 + base_range_pct)

    # --- Confidence logic ---
    if len(flags) == 0 and trend != "Down":
        confidence = "High"
    elif len(flags) <= 1:
        confidence = "Medium"
    else:
        confidence = "Low"

    # --- Reasons (keep it short + human) ---
    reasons = []
    if trend == "Up":
        reasons.append("Role trending up")
    elif trend == "Down":
        reasons.append("Role trending down")
    else:
        reasons.append("Role stable")

    if surface == "Turf":
        reasons.append("Surface: turf")
    if temp_f >= 90:
        reasons.append("High heat widens range")
    if temp_f <= 35:
        reasons.append("Cold conditions widen range")
    if short_week:
        reasons.append("Short week increases uncertainty")
    if return_status:
        reasons.append("Return status lowers confidence")

    # Keep reasons to top 3–4 to stay crisp
    reasons = reasons[:4]

    return {
        "name": name,
        "baseline": round(baseline, 1),
        "adjusted": round(adjusted, 1),
        "range": f"{round(range_low, 1)} – {round(range_high, 1)}",
        "confidence": confidence,
        "flags": ", ".join(flags) if flags else "—",
        "reasons": " • ".join(reasons),
    }


# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Script Not Included", layout="wide")

st.title("Script Not Included")
st.caption("Fantasy projections, missing context added.")

# Initialize session state for roster entries
if "roster" not in st.session_state:
    st.session_state.roster = []

# Top controls
col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    league_format = st.selectbox("League format", ["PPR", "Half PPR", "Standard"], index=0)
with col_b:
    st.write("")  # spacer
    if st.button("➕ Add RB", use_container_width=True):
        st.session_state.roster.append(
            {
                "name": "",
                "baseline": 12.0,
                "trend": "Flat",
                "temperature_f": 70,
                "surface": "Grass",
                "short_week": False,
                "return_status": False,
            }
        )
with col_c:
    st.info(
        "v1 focuses on **Running Backs**. Inputs are manual on purpose so you can ship the system spine first.",
        icon="ℹ️",
    )

st.divider()

# If no RBs added yet, guide the user
if len(st.session_state.roster) == 0:
    st.warning("Click **Add RB** to start building your roster.", icon="🧾")
    st.stop()

# Roster editor
st.subheader("Roster Inputs (RB v1)")

for i, rb in enumerate(st.session_state.roster):
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2])

        with c1:
            rb["name"] = st.text_input(f"RB Name #{i+1}", value=rb["name"], key=f"name_{i}")
        with c2:
            rb["baseline"] = st.number_input(
                f"Baseline proj #{i+1}",
                min_value=0.0,
                max_value=60.0,
                value=float(rb["baseline"]),
                step=0.5,
                key=f"baseline_{i}",
            )
        with c3:
            rb["trend"] = st.selectbox(
                f"Role trend #{i+1}",
                ["Up", "Flat", "Down"],
                index=["Up", "Flat", "Down"].index(rb["trend"]),
                key=f"trend_{i}",
            )
        with c4:
            remove = st.button("Remove", key=f"remove_{i}", use_container_width=True)
            if remove:
                st.session_state.roster.pop(i)
                st.rerun()

        d1, d2, d3, d4 = st.columns([1.2, 1.2, 1.2, 1.4])
        with d1:
            rb["temperature_f"] = st.number_input(
                f"Temp (°F) #{i+1}",
                min_value=-10,
                max_value=120,
                value=int(rb["temperature_f"]),
                step=1,
                key=f"temp_{i}",
            )
        with d2:
            rb["surface"] = st.selectbox(
                f"Surface #{i+1}",
                ["Grass", "Turf"],
                index=["Grass", "Turf"].index(rb["surface"]),
                key=f"surface_{i}",
            )
        with d3:
            rb["short_week"] = st.checkbox(
                f"Short week #{i+1}",
                value=bool(rb["short_week"]),
                key=f"short_{i}",
            )
        with d4:
            rb["return_status"] = st.checkbox(
                f"Return status (declared) #{i+1}",
                value=bool(rb["return_status"]),
                key=f"return_{i}",
                help="Binary flag only. This tool does not infer injuries or risk.",
            )

st.divider()

# Apply adjustments
st.subheader("Adjusted Outputs")

results = [adjust_projection(rb) for rb in st.session_state.roster]
df = pd.DataFrame(results)

left, right = st.columns([1.6, 1.0])
with left:
    st.dataframe(
        df[["name", "baseline", "adjusted", "range", "confidence", "flags", "reasons"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"League format: **{league_format}** (format affects scoring, but v1 keeps projections simple.)")

with right:
    st.markdown("#### Details (select a RB)")
    names = df["name"].tolist()
    selected = st.selectbox("RB", names, index=0)
    row = df[df["name"] == selected].iloc[0]

    st.metric("Baseline", row["baseline"])
    st.metric("Adjusted", row["adjusted"])
    st.write("**Range:**", row["range"])
    st.write("**Confidence:**", row["confidence"])
    st.write("**Flags:**", row["flags"])
    st.write("**Why:**", row["reasons"])

st.divider()
st.caption(
    "Disclaimer: Script Not Included is a fantasy analysis tool. It provides interpretive adjustments and does not predict outcomes or injuries."
)
