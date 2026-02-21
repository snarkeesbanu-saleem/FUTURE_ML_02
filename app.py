# app.py - Enhanced Support Ticket Classifier Demo
# Creative version with emojis, icons, better UX

import streamlit as st
import pickle
import re

# ── Same stopwords as in training ──
STOP_WORDS = {'a', 'an', 'the', 'is', 'in', 'at', 'of', 'on', 'and', 'or', 'to', 'for', 'with', 'by', 'from'}

# ── Load models ──
try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    cat_model  = pickle.load(open('cat_model.pkl', 'rb'))
    pri_model  = pickle.load(open('pri_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("⚠️  Model files missing!\nPlease run the training notebook first to generate the .pkl files.")
    st.stop()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    return ' '.join(word for word in words if word not in STOP_WORDS)

# ── Page config & styling ──
st.set_page_config(
    page_title="Ticket Triage Assistant",
    page_icon="🚨",
    layout="centered"
)

# ── Header with emoji flair ──
st.markdown(
    """
    <h1 style='text-align: center; color: #1E88E5;'>
        🚀 Ticket Triage Assistant
    </h1>
    <h4 style='text-align: center; color: #555;'>
        Automatically classify & prioritize support tickets
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ── Input area with icon ──
st.markdown("### 📝 Describe the customer's issue")
ticket_text = st.text_area(
    label="",
    height=140,
    placeholder="Example: Urgent help needed: Can't process payment right now!",
    label_visibility="collapsed"
)

# ── Classify button with icon ──
if st.button("🔍 Classify Ticket", type="primary", use_container_width=True):
    if ticket_text.strip():
        with st.spinner("Analyzing ticket... 🔄"):
            cleaned = clean_text(ticket_text)
            vec = vectorizer.transform([cleaned])
            category = cat_model.predict(vec)[0].capitalize()
            priority = pri_model.predict(vec)[0].capitalize()

        # ── Result cards with emojis ──
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style="background:#e3f2fd; padding:16px; border-radius:10px; text-align:center;">
                    <h3 style="margin:0; color:#1976D2;">Category</h3>
                    <p style="font-size:28px; margin:8px 0;">{category} 📋</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div style="background:#fff3e0; padding:16px; border-radius:10px; text-align:center;">
                    <h3 style="margin:0; color:#F57C00;">Priority</h3>
                    <p style="font-size:28px; margin:8px 0;">{priority} {'🚨' if priority == 'High' else '⚠️' if priority == 'Medium' else 'ℹ️'}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ── Priority alert banner ──
        if priority == 'High':
            st.error("🚨 **HIGH PRIORITY** – Handle immediately!")
        elif priority == 'Medium':
            st.warning("⚠️ **MEDIUM PRIORITY** – Should be addressed soon")
        else:
            st.info("ℹ️ **LOW PRIORITY** – Can be handled in normal queue")

        # Quick confidence note (optional – remove if not wanted)
        # st.caption("🔍 Model confidence is approximate – trained on limited synthetic data")

    else:
        st.warning("Please type something in the ticket field 😊")

# ── Footer ──
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#777; font-size:0.9em;'>
        Built with ❤️ &nbsp; Python + scikit-learn + Streamlit<br>
        For learning & demonstration purposes
    </div>
    """,
    unsafe_allow_html=True
)