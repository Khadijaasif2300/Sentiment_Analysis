import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random


# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)


# ----------------------------
# Custom CSS for a cleaner, eye-catching look
# ----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%);
        }

        .main-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7F00FF, #E100FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }

        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.05rem;
            margin-top: 0.2rem;
            margin-bottom: 1.5rem;
        }

        .result-card {
            padding: 2rem;
            border-radius: 18px;
            text-align: center;
            background: white;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }

        .result-emoji {
            font-size: 3.5rem;
        }

        .result-label {
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border-radius: 12px;
            padding: 0.8rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        .stButton > button {
            background: linear-gradient(90deg, #7F00FF, #E100FF);
            color: white;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1rem;
            transition: transform 0.15s ease;
        }

        .stButton > button:hover {
            transform: scale(1.02);
            color: white;
        }

        footer, #MainMenu {visibility: hidden;}

        .app-footer {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Load VADER analyzer
# ----------------------------
@st.cache_resource
def load_vader():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


analyzer = load_vader()


# ----------------------------
# Header
# ----------------------------
st.markdown('<p class="main-title">😊 Sentiment Analysis App</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Analyze the sentiment and emotional tone of any text using '
    '<b>VADER Sentiment Analysis</b></p>',
    unsafe_allow_html=True
)


# ----------------------------
# Sidebar - about & examples
# ----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses **VADER** (Valence Aware Dictionary and sEntiment "
        "Reasoner), a lexicon and rule-based sentiment analysis tool "
        "tuned for social media text, to score the sentiment of your text."
    )

    st.header("💡 Try an example")
    examples = [
        "I am extremely happy and excited about my new project!",
        "This is the worst day of my life, everything is going wrong.",
        "I wonder what tomorrow will bring, it's all so curious.",
        "The weather today is okay, nothing special.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["user_input"] = ex

    st.markdown("---")
    st.caption("Built with Python, Streamlit, NLTK & VADER")


# ----------------------------
# Emotion categories
# ----------------------------
positive_emotions = ["Happy", "Joyful", "Excited", "Delighted"]
negative_emotions = ["Sad", "Depressed", "Upset"]
wonder_emotions = ["Wonder", "Curious", "Interested", "Fascinated"]

sentiment_emojis = {
    "Happy": "😊", "Joyful": "😄", "Excited": "🤩", "Delighted": "🥰",
    "Sad": "😢", "Depressed": "😔", "Upset": "😠",
    "Wonder": "😲", "Curious": "🤔", "Interested": "🧐", "Fascinated": "😍",
    "Positive": "🙂", "Negative": "🙁", "Neutral": "😐"
}


# ----------------------------
# Helper functions
# ----------------------------
def analyze_sentiment(text):
    """Analyze the sentiment of text using VADER."""
    return analyzer.polarity_scores(text)


def get_sentiment_category(text, compound):
    text = text.lower()

    if "wonder" in text or "curious" in text:
        return random.choice(wonder_emotions)
    elif compound > 0.5:
        return random.choice(positive_emotions)
    elif compound < -0.5:
        return random.choice(negative_emotions)
    elif compound > 0:
        return "Positive"
    elif compound < 0:
        return "Negative"
    else:
        return "Neutral"


# ----------------------------
# User input
# ----------------------------
user_input = st.text_area(
    "Enter your text below:",
    placeholder="Example: I am extremely happy and excited about my new project!",
    height=150,
    key="user_input"
)

analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True)


# ----------------------------
# Analyze & display results
# ----------------------------
if analyze_clicked:

    if user_input.strip():
        sentiment = analyze_sentiment(user_input)
        compound = sentiment["compound"]
        sentiment_category = get_sentiment_category(user_input, compound)
        emoji = sentiment_emojis[sentiment_category]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{sentiment_category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positive", f"{sentiment['pos']:.2f}")
        col2.metric("Negative", f"{sentiment['neg']:.2f}")
        col3.metric("Neutral", f"{sentiment['neu']:.2f}")
        col4.metric("Compound", f"{sentiment['compound']:.2f}")

        st.progress(min(max((compound + 1) / 2, 0.0), 1.0))

        st.info(f"The overall compound sentiment score is **{compound:.2f}**.")

    else:
        st.warning("⚠️ Please enter some text before analyzing.")


# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    '<p class="app-footer">Developed with ❤️ using Python, Streamlit, NLTK & VADER'
    '<br><b>👩‍💻 Khadija Asif</b></p>',
    unsafe_allow_html=True
)
