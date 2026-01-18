import streamlit as st
import re
import pandas as pd
import joblib
import numpy as np
import base64
from utils import preprocess

# 1. SETUP PAGE
st.set_page_config(
    page_title="Breaking Bad Chat Analyzer",
    page_icon="⚗️",
    layout="centered"
)

# --- FUNCTION TO ADD LOCAL BACKGROUND ---
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"⚠️ Image not found: '{image_file}'. Using default dark mode.")

# --- 🟢 WALLPAPER CONFIGURATION 🟢 ---
add_bg_from_local('my_wallpaper.jpg') 

# 2. INJECT REFINED CSS
st.markdown("""
<style>
    /* --- GLOBAL FONTS & COLORS --- */
    .stApp {
        color: #ffffff;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* --- SMOOTH ANIMATIONS --- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 20px, 0); }
        to { opacity: 1; transform: none; }
    }
    
    /* --- GLASSMORPHISM CONTAINERS --- */
    .glass-container {
        background: rgba(16, 28, 35, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        animation: fadeInUp 0.8s ease-out;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }

    /* --- LOGO STYLING --- */
    .element-box {
        display: inline-block;
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        color: white;
        padding: 8px 14px;
        margin-right: 4px;
        font-weight: 700;
        border: 1px solid #66bb6a;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.4);
    }
    
    .title-text {
        font-size: 50px;
        font-weight: 300;
        text-align: center;
        margin-bottom: 0px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8);
        color: #fff;
    }
    
    .subtitle {
        text-align: center;
        color: #cfd8dc;
        font-size: 16px;
        margin-top: 10px;
        font-weight: 400;
        letter-spacing: 0.5px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.8);
    }

    /* --- METRIC CARDS --- */
    div[data-testid="stMetric"] {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #00e5ff;
        box-shadow: 0 10px 20px rgba(0, 229, 255, 0.15);
    }
    div[data-testid="stMetricLabel"] {
        color: #00e5ff !important;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* --- FILE UPLOADER --- */
    div[data-testid="stFileUploader"] section {
        background-color: rgba(0, 0, 0, 0.6);
        border: 2px dashed #546e7a;
        transition: border-color 0.3s;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #00e5ff;
        background-color: rgba(0, 229, 255, 0.05);
    }

    /* --- TABLES --- */
    .stDataFrame {
        background: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* --- BUTTONS --- */
    div.stButton > button {
        background: linear-gradient(90deg, #0288d1 0%, #00acc1 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        box-shadow: 0 5px 15px rgba(0, 229, 255, 0.4);
        transform: translateY(-2px);
    }

    div.stDownloadButton > button {
        background: linear-gradient(90deg, #fbc02d 0%, #ffa000 100%);
        color: #263238;
        border: none;
        border-radius: 8px;
        font-weight: 700;
    }
    div.stDownloadButton > button:hover {
        box-shadow: 0 5px 15px rgba(255, 160, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* --- HEADERS --- */
    .section-header {
        color: #fff;
        font-size: 20px;
        font-weight: 600;
        margin-top: 40px; /* Added spacing here instead of '---' */
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(0, 229, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

</style>
""", unsafe_allow_html=True)

# 3. LOAD MODELS
try:
    svm_clf = joblib.load('svm_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("⚠️ CRITICAL ERROR: Model files (pkl) not found. Run 'train_model.py' first.")
    st.stop()

# 4. FUNCTIONS
def clean_whatsapp_content(file_content):
    lines = file_content.splitlines()
    
    ios_date_pattern = re.compile(r'^[\u2000-\u206F]*\[\d{1,2}/\d{1,2}/\d{2,4},.*?:.*?\]')
    ios_msg_pattern = re.compile(r'^[\u2000-\u206F]*\[.*?\] (.*?): (.*)$')
    android_date_pattern = re.compile(r'^[\u2000-\u206F]*\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}.*? -')
    android_msg_pattern = re.compile(r'^[\u2000-\u206F]*\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}.*? - (.*?): (.*)$')

    junk_filters = [
        "Messages and calls are end-to-end encrypted", "created group", "added", "left", 
        "changed the subject", "security code changed", "image omitted", "sticker omitted", 
        "video omitted", "GIF omitted", "This message was deleted", "document omitted", 
        "waiting for this message", "pinned a message", "joined using a group link",
        "changed the group description", "removed", "joined using this group's invite link"
    ]

    data, current_sender, current_buffer = [], None, []
    def clean(t): return re.sub(r'[\u2000-\u206F]', '', t).replace('~', '').strip()

    for line in lines:
        line = line.strip()
        if not line: continue

        is_ios = ios_date_pattern.match(line)
        is_android = android_date_pattern.match(line)

        if is_ios or is_android:
            if current_sender and current_buffer:
                data.append([current_sender, " ".join(current_buffer)])
            current_sender, current_buffer = None, []
            match = ios_msg_pattern.match(line) if is_ios else android_msg_pattern.match(line)

            if match:
                sender, message = map(clean, match.groups())
                if any(j in message for j in junk_filters): continue
                if any(j in sender for j in ["created group", "added", "left"]): continue
                current_sender = sender
                current_buffer.append(message)
        else:
            if current_sender: current_buffer.append(clean(line))

    if current_sender and current_buffer:
        data.append([current_sender, " ".join(current_buffer)])

    return pd.DataFrame(data, columns=["sender", "message"])

# --- UI HEADER ---
st.markdown("""
<div class="glass-container" style="text-align: center;">
    <div class="title-text">
        <span class="element-box">Br</span>eaking 
        <span class="element-box">Ba</span>d
    </div>
    <div style="font-size: 24px; font-weight: 300; letter-spacing: 4px; margin-top: 5px; color: #fff;">
        CHAT ANALYZER
    </div>
    <div class="subtitle">
        Predicting personalities based on communication patterns.
    </div>
</div>
""", unsafe_allow_html=True)

# --- UPLOAD SECTION ---
col_up1, col_up2, col_up3 = st.columns([1, 8, 1])
with col_up2:
    st.markdown('<div style="background: rgba(0,0,0,0.5); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">', unsafe_allow_html=True)
    uploaded_csv = st.file_uploader("📂 Upload WhatsApp CSV", type="csv")
    uploaded_txt = st.file_uploader("📜 Upload WhatsApp TXT", type="txt")
    st.markdown('</div>', unsafe_allow_html=True)

dk = None

# --- PROCESS FILE ---
if uploaded_csv:
    dk = pd.read_csv(uploaded_csv)
elif uploaded_txt:
    string_data = uploaded_txt.getvalue().decode("utf-8")
    with st.spinner('Processing file content...'):
        dk = clean_whatsapp_content(string_data)

# --- MAIN LOGIC ---
if dk is not None and not dk.empty:
    
    # Removed the 'st.markdown("---")' which was causing the black bar
    
    # METRICS
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", len(dk))
    with col2:
        st.metric("Participants", dk["sender"].nunique())

    # PREVIEW
    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(dk.head(10), use_container_width=True)

    # --- PREDICTION LOGIC ---
    try:
        # Preprocess
        dk = dk.sort_values(by=['sender'], ascending=True)
        dk = dk[dk['message'].str.len() >= 1]
        
        value_counts = dk['message'].value_counts()
        dk = dk[~dk['message'].isin(value_counts[value_counts < 1].index)]
        
        # CLEANER PROGRESS BAR
        progress_text = "Initializing Analysis..."
        my_bar = st.progress(0, text=progress_text)
        
        dk['message'] = dk['message'].apply(preprocess)
        my_bar.progress(30, text="Processing Text Vectors...")

        if len(dk) > 0:
            # Get Raw Confidence Scores
            X_to_predict = dk['message'].values.astype('U')
            tfidf_test_vectors2 = tfidf_vectorizer.transform(X_to_predict)
            my_bar.progress(60, text="Computing Model Predictions...")
            
            # Get raw distance scores
            raw_scores = svm_clf.decision_function(tfidf_test_vectors2)
            
            # Create a DataFrame of scores
            score_df = pd.DataFrame(raw_scores, columns=svm_clf.classes_)
            score_df['sender'] = dk['sender'].values
            
            # Average scores
            avg_scores = score_df.groupby('sender').mean()
            
            # SOFTMAX FUNCTION
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            prob_scores = avg_scores.apply(softmax, axis=1)
            
            # DRAFT PICK ALGORITHM
            results = []
            friends_queue = list(prob_scores.index)
            assignments = {}
            import numpy as np 

            while friends_queue:
                taken_characters = set()
                potential_matches = []
                
                for friend in friends_queue:
                    friend_probs = prob_scores.loc[friend]
                    sorted_matches = friend_probs.sort_values(ascending=False)
                    for char_name, prob in sorted_matches.items():
                        potential_matches.append({
                            'friend': friend, 'character': char_name, 'score': prob
                        })
                
                potential_matches.sort(key=lambda x: x['score'], reverse=True)
                
                round_assignments = []
                for match in potential_matches:
                    friend = match['friend']
                    char = match['character']
                    
                    if friend in assignments: continue
                    if char in taken_characters: continue
                        
                    assignments[friend] = char
                    taken_characters.add(char)
                    round_assignments.append(friend)
                
                friends_queue = [f for f in friends_queue if f not in assignments]
                
                if not round_assignments and friends_queue:
                    friend_to_force = friends_queue[0]
                    best_char = prob_scores.loc[friend_to_force].idxmax()
                    assignments[friend_to_force] = best_char
                    friends_queue.pop(0)

            # FINAL RESULTS
            final_data = []
            for friend, character in assignments.items():
                confidence = prob_scores.loc[friend, character]
                final_data.append({
                    'Participant': friend,
                    'Character Match': character,
                    'Confidence': f"{confidence:.1%}"
                })
            
            my_bar.progress(100, text="Analysis Complete")
            output_df = pd.DataFrame(final_data)

            st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
            
            # Display result with distinct styling
            st.dataframe(output_df, use_container_width=True, hide_index=True)

            # CSV Download
            csv_data = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Results CSV",
                csv_data,
                "analysis_results.csv",
                "text/csv"
            )
        else:
            st.warning("⚠️ Insufficient Data: The chat log is too short for accurate analysis.")
            
    except Exception as e:
        st.error(f"Analysis Error: {e}")