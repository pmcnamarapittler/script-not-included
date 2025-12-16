import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

# --- Compatibility patch for streamlit_drawable_canvas on newer Streamlit ---
# streamlit_drawable_canvas expects streamlit.elements.image.image_to_url(img, width, clamp, channels, output_format).
# Newer Streamlit moved/changed internals to image_utils.image_to_url(img, layout_config, image_format).
try:
    import streamlit.elements.image as st_image  # type: ignore
    if not hasattr(st_image, "image_to_url"):
        from types import SimpleNamespace
        from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url  # type: ignore

        def _image_to_url_compat(image, width=None, clamp=False, channels="RGB", output_format="PNG", *args, **kwargs):
            # streamlit_drawable_canvas may pass an extra positional arg in some versions.
            # We only need width + output_format for the newer Streamlit image_to_url.
            layout_config = SimpleNamespace(width=width, height=None)
            # Streamlit has changed this signature across versions. Try the common forms.
            try:
                # Newer: image_to_url(image, layout_config, channels, output_format, image_id)
                return _new_image_to_url(image, layout_config, channels, output_format, kwargs.get("image_id", "canvas_bg"))
            except TypeError:
                try:
                    # Slight variant: image_to_url(image, layout_config, output_format)
                    return _new_image_to_url(image, layout_config, output_format)
                except TypeError:
                    # Oldest: image_to_url(image, width, clamp, channels, output_format)
                    return _new_image_to_url(image, width, clamp, channels, output_format)

        st_image.image_to_url = _image_to_url_compat  # type: ignore[attr-defined]
except Exception:
    # If this fails, the canvas import below will show an error message to the user.
    pass

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

from cv_signals import compute_burst_signal, load_first_frame


# -----------------------------
# Adjustment Engine (RB v1)
# -----------------------------
def adjust_projection(rb: dict) -> dict:
    """Takes a single RB input dict and returns adjusted projection outputs."""
    name = rb["name"].strip() if rb["name"] else "Unnamed RB"
    baseline = float(rb["baseline"])

    trend = rb["trend"]
    trend_delta = {"Up": 1.5, "Flat": 0.0, "Down": -1.5}.get(trend, 0.0)

    temp_f = rb["temperature_f"]
    surface = rb["surface"]
    short_week = rb["short_week"]
    return_status = rb["return_status"]

    surface_delta = -0.5 if surface == "Turf" else 0.0

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

    adjusted = baseline + trend_delta + surface_delta

    base_range_pct = 0.20
    if len(flags) >= 2:
        base_range_pct = 0.30
    elif len(flags) == 1:
        base_range_pct = 0.25

    range_low = max(0.0, adjusted * (1 - base_range_pct))
    range_high = adjusted * (1 + base_range_pct)

    if len(flags) == 0 and trend != "Down":
        confidence = "High"
    elif len(flags) <= 1:
        confidence = "Medium"
    else:
        confidence = "Low"

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

if "roster" not in st.session_state:
    st.session_state.roster = []

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    league_format = st.selectbox("League format", ["PPR", "Half PPR", "Standard"], index=0)
with col_b:
    st.write("")
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

if len(st.session_state.roster) == 0:
    st.warning("Click **Add RB** to start building your roster.", icon="🧾")
    st.stop()

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
    st.markdown("#### Receipts (select a RB)")

    names = df["name"].tolist()
    selected = st.selectbox("RB", names, index=0)
    row = df[df["name"] == selected].iloc[0]

    st.metric("Baseline", row["baseline"])
    st.metric("Adjusted", row["adjusted"])

    st.write("**Range:**", row["range"])
    st.write("**Confidence:**", row["confidence"])

    st.markdown("#### Context Flags")
    if row["flags"] != "—":
        for flag in row["flags"].split(","):
            st.markdown(f"- {flag.strip()}")
    else:
        st.write("No significant context flags.")

    st.markdown("#### Why This Changed")
    st.write(row["reasons"])



# Insert Film Signals section as standalone full-width section below Adjusted Outputs
st.divider()

st.header("Film Signals")
st.caption("Use film-derived signals to sanity-check projections. Click-to-target is the most reliable on broadcast footage.")

input_mode = st.radio(
    "Provide video via:",
    ["Upload file", "Paste video URL"],
    horizontal=True,
    key="video_input_mode",
)

clip = None
video_url = None

if input_mode == "Upload file":
    clip = st.file_uploader(
        "Upload a short clip (5–15s)",
        type=["mp4", "mov"],
        key="clip_uploader",
    )
else:
    video_url = st.text_input(
        "Paste direct video URL (.mp4 / .mov)",
        placeholder="https://video.twimg.com/...mp4",
        key="clip_url",
    )

if not (clip or video_url):
    st.info("Upload a clip or paste a direct .mp4/.mov URL to enable Film Signals.")
else:
    # Key controls should be immediately visible
    burst_start = st.slider(
        "Burst start (seconds)",
        min_value=0.0,
        max_value=8.0,
        value=2.5,
        step=0.1,
        help="Set this to the handoff / moment the RB begins the run.",
        key="burst_start_seconds",
    )

    show_overlays = st.checkbox(
        "Show what the computer sees (pose overlay)",
        value=True,
        key="show_pose_overlay",
    )

    st.subheader("Pick the correct player")
    c_pick1, c_pick2 = st.columns([1.2, 1.0])

    with c_pick1:
        use_click_target = st.checkbox(
            "Click-to-target: draw a box around the RB (recommended)",
            value=True,
            help="Most reliable. Draw a rectangle on the first frame around your RB.",
            key="use_click_target",
        )

        auto_pick = st.checkbox(
            "Auto-pick moving player (experimental)",
            value=False,
            help="Uses motion to suggest an ROI so the tracker is more likely to lock onto the ballcarrier.",
            key="auto_pick_player",
        )

    roi_pct = None

    with c_pick2:
        use_roi = st.checkbox(
            "Manual ROI sliders",
            value=False,
            help="Use sliders if you want to tune the region precisely.",
            key="use_roi",
        )

        if use_roi:
            x0 = st.slider("ROI left (%)", 0.0, 100.0, 25.0, 1.0, key="roi_x0")
            y0 = st.slider("ROI top (%)", 0.0, 100.0, 20.0, 1.0, key="roi_y0")
            x1 = st.slider("ROI right (%)", 0.0, 100.0, 75.0, 1.0, key="roi_x1")
            y1 = st.slider("ROI bottom (%)", 0.0, 100.0, 90.0, 1.0, key="roi_y1")

            if x1 <= x0:
                x1 = min(100.0, x0 + 1.0)
            if y1 <= y0:
                y1 = min(100.0, y0 + 1.0)

            roi_pct = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    # Click-to-target canvas
    if use_click_target:
        if st_canvas is None:
            st.error(
                "Click-to-target requires `streamlit-drawable-canvas`.\n\n"
                "Install in your venv:\n"
                "`pip install streamlit-drawable-canvas pillow`",
            )
        else:
            frame_rgb, fw, fh = load_first_frame(uploaded_file=clip, video_url=video_url)
            if frame_rgb is None:
                st.error("Could not load a preview frame from this clip/URL. Try a direct .mp4 URL or upload.")
            else:
                display_w = 900
                scale = display_w / float(fw)
                display_h = int(fh * scale)

                bg = Image.fromarray(frame_rgb)
                bg = bg.resize((display_w, display_h))

                st.caption("Draw ONE rectangle around the RB. Last rectangle wins.")

                canvas_result = st_canvas(
                    fill_color="rgba(255, 0, 255, 0.10)",
                    stroke_width=3,
                    stroke_color="#FF00FF",
                    background_image=bg,
                    update_streamlit=True,
                    height=display_h,
                    width=display_w,
                    drawing_mode="rect",
                    key="target_canvas",
                )

                if canvas_result.json_data and "objects" in canvas_result.json_data:
                    rects = [o for o in canvas_result.json_data["objects"] if o.get("type") == "rect"]
                    if rects:
                        r = rects[-1]
                        left = float(r.get("left", 0.0))
                        top = float(r.get("top", 0.0))
                        width = float(r.get("width", 1.0))
                        height = float(r.get("height", 1.0))

                        x0_px = max(0.0, left / scale)
                        y0_px = max(0.0, top / scale)
                        x1_px = min(float(fw), (left + width) / scale)
                        y1_px = min(float(fh), (top + height) / scale)

                        roi_pct = {
                            "x0": (x0_px / float(fw)) * 100.0,
                            "y0": (y0_px / float(fh)) * 100.0,
                            "x1": (x1_px / float(fw)) * 100.0,
                            "y1": (y1_px / float(fh)) * 100.0,
                        }

                        st.success(
                            f"Target ROI set: x {roi_pct['x0']:.1f}–{roi_pct['x1']:.1f}% · y {roi_pct['y0']:.1f}–{roi_pct['y1']:.1f}%"
                        )
                    else:
                        st.info("Draw a rectangle to set the target ROI.")

    # Analyze button so it doesn't re-run constantly while adjusting controls
    do_analyze = st.button("Analyze clip", type="primary")

    if do_analyze:
        with st.spinner("Analyzing clip…"):
            sig = compute_burst_signal(
                uploaded_file=clip,
                video_url=video_url,
                preview_frames=6 if show_overlays else 0,
                roi_pct=roi_pct,
                auto_roi=(auto_pick and not use_click_target),
            )

        if "error" in sig:
            st.error(sig["error"])
        else:
            # Results should appear immediately under the upload section
            st.subheader("Results")
            st.success(f"Burst Score (raw): {sig['burst_score']} / 100")
            st.caption(sig["notes"])

            start_idx = int(burst_start * sig["fps"])
            end_idx = int(min(len(sig["accels"]), start_idx + 2.0 * sig["fps"]))

            window_accels = np.array(sig["accels"][start_idx:end_idx], dtype=np.float32)
            if len(window_accels) < 5:
                st.warning("Burst window too short—move the slider later/earlier.")
            else:
                max_accel_window = float(np.max(window_accels))
                clip_p95 = float(np.percentile(np.abs(np.array(sig["accels"])), 95))
                denom = clip_p95 if clip_p95 > 1e-6 else 1.0
                burst_window_score = int(np.clip((max_accel_window / denom) * 70 + 30, 0, 100))

                st.info(
                    f"Burst (handoff-aligned): {burst_window_score} / 100  (window: {burst_start:.1f}s → {burst_start+2.0:.1f}s)"
                )

            t = np.arange(len(sig["speeds"])) / sig["fps"]
            plot_df = pd.DataFrame({"t_seconds": t, "speed_px_per_frame": sig["speeds"]})
            st.line_chart(plot_df, x="t_seconds", y="speed_px_per_frame")

            if show_overlays:
                previews = sig.get("preview_frames", [])
                if previews:
                    st.subheader("What the computer is tracking")
                    st.caption("Pose skeleton + hip center overlay (and ROI/pose boxes if enabled).")
                    if sig.get("roi_auto"):
                        st.caption("Auto ROI was used to focus tracking on the most-moving region.")
                    st.image(previews, caption=[f"Frame {i+1}" for i in range(len(previews))])
                else:
                    st.caption("No preview frames available for this clip.")

st.divider()
st.caption(
    "Disclaimer: Script Not Included is a fantasy analysis tool. It provides interpretive adjustments and does not predict outcomes or injuries."
)
