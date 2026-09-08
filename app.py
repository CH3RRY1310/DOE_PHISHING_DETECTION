import json

import pandas as pd
import streamlit as st

from prototype import URLPhishingPrototype

st.set_page_config(
    page_title="URL Sentinel",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root {
        --ink: #16232b;
        --muted: #6c7b80;
        --paper: #f4f7f2;
        --panel: #ffffff;
        --line: #d7e2dc;
        --teal: #087f78;
        --coral: #d75b45;
        --gold: #d49b2a;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    .block-container { max-width: 1180px; padding: 3.5rem 2rem 4rem; }
    h1, h2, h3, p, label, .stMarkdown { font-family: 'Space Grotesk', sans-serif; }
    h1 { font-size: clamp(2.5rem, 6vw, 5.6rem); line-height: .95; letter-spacing: 0; margin: 0; color: var(--ink); }
    h2 { letter-spacing: 0; color: var(--ink); }
    .eyebrow { color: var(--teal); font: 500 .78rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 1rem; }
    .lede { color: var(--muted); font-size: 1.05rem; max-width: 640px; margin-top: 1rem; }
    .hero { border-bottom: 1px solid var(--line); padding-bottom: 2.5rem; margin-bottom: 2rem; }
    .hero-mark { color: var(--coral); }
    .result-card { border: 1px solid var(--line); background: var(--panel); padding: 1.5rem; min-height: 190px; }
    .result-label { font: 500 .76rem 'DM Mono', monospace; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
    .prediction { font: 700 2.7rem 'Space Grotesk', sans-serif; margin: .6rem 0; }
    .phishing { color: var(--coral); }
    .legitimate { color: var(--teal); }
    .probability { color: var(--ink); font: 500 1rem 'DM Mono', monospace; }
    .section-kicker { border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 2.5rem; color: var(--teal); font: 500 .76rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
    .stTextInput input { background: white; border: 1px solid var(--line); border-radius: 2px; color: var(--ink); font: 400 1rem 'DM Mono', monospace; padding: .85rem; }
    .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button { border-radius: 2px; border: 1px solid var(--teal); background: var(--teal); color: #ffffff !important; font-family: 'Space Grotesk', sans-serif; font-weight: 600; }
    .stButton button *, .stDownloadButton button *, [data-testid="stFormSubmitButton"] button * { color: #ffffff !important; }
    .stButton button:hover, .stDownloadButton button:hover, [data-testid="stFormSubmitButton"] button:hover { border-color: var(--ink); background: var(--ink); color: #ffffff !important; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); padding: 1rem; }
    [data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace; color: var(--muted); }
    [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    .note { color: var(--muted); font-size: .86rem; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_prototype():
    return URLPhishingPrototype()


prototype = load_prototype()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">URL SENTINEL <span class="hero-mark">//</span> RESEARCH PROTOTYPE</div>
      <h1>Read the shape<br>of a URL.</h1>
      <p class="lede">A five-feature XGBoost detector with local explanations for every decision.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.6, 1], gap="large")
with left:
    st.markdown('<div class="section-kicker">01 / Inspect</div>', unsafe_allow_html=True)
    with st.form("analysis_form"):
        url = st.text_input(
            "URL",
            value="https://www.southbankmosaics.com",
            placeholder="https://example.com/path",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze URL", use_container_width=True)
    st.markdown(
        '<p class="note">The prototype reads lexical URL structure only. It does not visit the URL or verify website reputation.</p>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="section-kicker">Quick samples</div>', unsafe_allow_html=True)
    samples = {
        "Known legitimate": "https://www.southbankmosaics.com",
        "Known phishing": "http://www.f0519141.xsph.ru",
        "Long query URL": "http://account-confirmation.example.com/a1/b2/c3/d4/e5?x=123456",
    }
    selected_sample = st.selectbox("Sample URL", ["Choose a sample"] + list(samples), label_visibility="collapsed")
    if selected_sample != "Choose a sample":
        st.code(samples[selected_sample], language=None)
        if st.button("Use sample", use_container_width=True):
            st.session_state["sample_url"] = samples[selected_sample]
            st.rerun()

if "sample_url" in st.session_state and not submitted:
    url = st.session_state["sample_url"]
    submitted = True

if submitted:
    url = url.strip()
    if not url:
        st.error("Enter a URL before analyzing.")
    else:
        result = prototype.predict(url)
        probability = result["phishing_probability"]
        prediction_class = "phishing" if result["prediction"] == "PHISHING" else "legitimate"
        st.markdown('<div class="section-kicker">02 / Result</div>', unsafe_allow_html=True)
        result_col, probability_col = st.columns([1, 1], gap="large")
        with result_col:
            st.markdown(
                f'<div class="result-card"><div class="result-label">Classification</div><div class="prediction {prediction_class}">{result["prediction"]}</div><div class="probability">{probability:.2%} phishing probability</div></div>',
                unsafe_allow_html=True,
            )
        with probability_col:
            st.metric("Phishing probability", f"{probability:.2%}")
            st.progress(probability)

        st.markdown('<div class="section-kicker">03 / URL signature</div>', unsafe_allow_html=True)
        feature_frame = pd.DataFrame(
            [{"Feature": name, "Value": value} for name, value in result["features"].items()]
        )
        st.dataframe(feature_frame, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-kicker">04 / Why this result</div>', unsafe_allow_html=True)
        explanation = pd.DataFrame(result["explanation"])
        explanation["direction"] = explanation["direction"].map(
            {"toward_phishing": "toward PHISHING", "toward_legitimate": "toward LEGITIMATE"}
        )
        explanation = explanation.rename(
            columns={
                "feature": "Feature",
                "value": "Value",
                "shap_value": "SHAP contribution",
                "direction": "Direction",
            }
        )
        st.dataframe(explanation, hide_index=True, use_container_width=True)
        st.download_button(
            "Download JSON result",
            data=json.dumps(result, indent=2),
            file_name="url_analysis.json",
            mime="application/json",
        )
else:
    st.markdown('<div class="section-kicker">02 / Result</div>', unsafe_allow_html=True)
    st.info("Enter a URL and select Analyze URL to see the model decision.")
