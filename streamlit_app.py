import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to get video as base64
def get_video_as_base64(path):
    with open(path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode()

# Note: You need to have the video file in a path accessible by the script.
# For this example, let's assume it's in a 'static/images' folder relative to the script.
# If you are deploying on Streamlit Cloud, you'll need to include this video file in your repo.
try:
    video_base64 = get_video_as_base64("static/images/166823-835662276.mp4")
except FileNotFoundError:
    # A fallback solid color or gradient if the video is not found
    video_base64 = None


# --- CSS Styling ---
# This CSS is a combination of your style.css and some Streamlit-specific tweaks.
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Hide Streamlit's default components */
    #MainMenu, .stDeployButton, footer, .stApp > header {{
        visibility: hidden;
    }}

    /* Video Background */
    .video-background {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -2;
        overflow: hidden;
    }}

    #bg-video {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}

    /* Overlay */
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.6);
        z-index: -1;
    }}


    /* Main container styling */
    .main-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 2rem;
        text-align: center;
        color: white;
        font-family: 'Poppins', sans-serif;
    }}

    .main-title {{
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
        margin-bottom: 0.5rem;
    }}

    .main-subtitle {{
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
        max-width: 600px;
    }}

    .content-wrapper {{
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 2rem;
        width: 100%;
        max-width: 1200px;
    }}

    .upload-box, .result-box {{
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        flex-grow: 1;
        width: 100%;
    }}
    
    .upload-box {{
         max-width: 500px;
    }}

    .result-box {{
        text-align: left;
    }}

    .upload-box h2, .result-box h2 {{
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(26, 188, 156, 0.8);
        padding-bottom: 0.5rem;
        display: inline-block;
    }}

    /* Styling for the file uploader */
    .stFileUploader > div > div > button {{
        background-color: #1abc9c;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s ease;
    }}
    .stFileUploader > div > div > button:hover {{
        background-color: #16a085;
    }}
    
    .stFileUploader > div > div > [data-testid="stMarkdownContainer"] > p {{
        font-style: italic;
        color: #ddd;
    }}

    /* Styling for the Diagnose button */
    .stButton > button {{
        width: 100%;
        padding: 15px;
        border: none;
        background-color: #1abc9c;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: #16a085;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(26, 188, 156, 0.4);
    }}
    .stButton > button:disabled {{
        background-color: #7f8c8d;
        cursor: not-allowed;
    }}
    
    /* Result styling */
    .result-content {{
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }}
    .result-image {{
        width: 150px;
        height: 150px;
        border-radius: 10px;
        object-fit: cover;
        border: 3px solid #1abc9c;
    }}
    .result-details h3 {{
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }}
    .result-details p {{
        margin-bottom: 0.75rem;
        line-height: 1.6;
    }}
    .result-details strong {{
        font-weight: 600;
        color: #1abc9c;
    }}

    /* Responsive Design */
    @media (max-width: 992px) {{
        .content-wrapper {{
            flex-direction: column;
            align-items: center;
        }}
    }}
     @media (max-width: 768px) {{
        .result-content {{
            flex-direction: column;
            align-items: center;
            text-align: center;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# --- Background ---
if video_base64:
    st.markdown(f"""
    <div class="video-background">
        <video autoplay muted loop playsinline id="bg-video">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
    </div>
    <div class="overlay"></div>
    """, unsafe_allow_html=True)


# --- Model and Data Loading ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/plant_disease_recog_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_disease_info():
    try:
        with open("plant_disease.json", 'r') as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading disease data: {e}")
        return None

model = load_model()
plant_disease_info = load_disease_info()


# --- Prediction Function ---
def model_predict(image_data, model, disease_info):
    img = Image.open(image_data).resize((160, 160))
    feature = tf.keras.utils.img_to_array(img)
    feature = np.array([feature])
    
    prediction = model.predict(feature)
    predicted_class_index = prediction.argmax()
    
    if disease_info and 0 <= predicted_class_index < len(disease_info):
        return disease_info[predicted_class_index]
    return {"name": "Unknown", "cause": "N/A", "cure": "N/A"}


# --- Streamlit App Layout ---
st.markdown('<main class="main-container">', unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Plant Disease Recognition System</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Upload an image of a plant leaf to identify diseases and find cures.</p>', unsafe_allow_html=True)

# Use columns to manage layout
col1, col2 = st.columns([1.5, 2]) # Adjust ratio as needed

with col1:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.markdown('<h2>Analyze Leaf Image</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['jpg', 'jpeg', 'png'], 
        label_visibility="collapsed"
    )
    
    diagnose_button = st.button("Diagnose", use_container_width=True, disabled=not uploaded_file)
    
    st.markdown('</div>', unsafe_allow_html=True)


with col2:
    if diagnose_button and uploaded_file is not None and model is not None:
        with st.spinner('Analyzing...'):
            prediction = model_predict(uploaded_file, model, plant_disease_info)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<h2>Diagnosis Result</h2>', unsafe_allow_html=True)
            
            # Display result
            st.markdown(f"""
            <div class="result-content">
                <img src="data:image/png;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}" class="result-image">
                <div class="result-details">
                    <h3><strong>Plant:</strong> {prediction.get('name', 'N/A')}</h3>
                    <p><strong>Cause:</strong> {prediction.get('cause', 'N/A')}</p>
                    <p><strong>Cure:</strong> {prediction.get('cure', 'N/A')}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Placeholder so the layout doesn't shift
        st.markdown('<div style="height: 300px;"></div>', unsafe_allow_html=True)


st.markdown('</main>', unsafe_allow_html=True)
