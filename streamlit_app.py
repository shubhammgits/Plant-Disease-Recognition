import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS to match Flask app exactly
st.markdown("""
<style>
    /* Hide Streamlit branding completely */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    .stSidebar {display: none;}
    
    /* Full screen background like Flask */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080"><defs><radialGradient id="bg" cx="50%" cy="50%" r="50%"><stop offset="0%" style="stop-color:%23134e5e;stop-opacity:1" /><stop offset="100%" style="stop-color:%23071e26;stop-opacity:1" /></radialGradient></defs><rect width="1920" height="1080" fill="url(%23bg)"/><circle cx="200" cy="200" r="3" fill="rgba(255,255,255,0.1)"/><circle cx="600" cy="400" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="1200" cy="300" r="4" fill="rgba(255,255,255,0.1)"/><circle cx="1600" cy="700" r="2" fill="rgba(255,255,255,0.1)"/></svg>');
        background-size: cover;
        background-attachment: fixed;
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container styling exactly like Flask */
    .main-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 2rem;
        text-align: center;
    }
    
    /* Title styling exactly like Flask */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
        max-width: 600px;
        color: rgba(255,255,255,0.9);
    }
    
    /* Content wrapper exactly like Flask */
    .content-wrapper {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 2rem;
        width: 100%;
        max-width: 1200px;
        margin-top: 2rem;
    }
    
    /* Upload box styling exactly like Flask */
    .upload-box, .result-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        flex-grow: 1;
        max-width: 500px;
    }
    
    .upload-box h2, .result-box h2 {
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(26, 188, 156, 0.8);
        padding-bottom: 0.5rem;
        display: inline-block;
        color: white;
    }
    
    /* File uploader styling like Flask */
    .stFileUploader {
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
        padding: 10px;
        border: 2px dashed rgba(26, 188, 156, 0.5);
        margin: 1.5rem 0;
    }
    
    .stFileUploader label {
        color: #ddd !important;
        font-style: italic;
    }
    
    /* Button styling exactly like Flask */
    .stButton > button {
        width: 100%;
        padding: 15px;
        border: none;
        background: linear-gradient(45deg, #1abc9c, #16a085);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #16a085, #1abc9c);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(26, 188, 156, 0.4);
    }
    
    /* Image styling */
    .stImage > img {
        border-radius: 10px;
        border: 3px solid #1abc9c;
    }
    
    /* Result content styling like Flask */
    .result-content {
        display: flex;
        gap: 1.5rem;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }
    
    .result-details h3 {
        font-size: 1.3rem;
        margin-bottom: 1rem;
        color: white;
    }
    
    .result-details p {
        margin-bottom: 0.75rem;
        line-height: 1.6;
        font-weight: 300;
        color: rgba(255,255,255,0.9);
    }
    
    .result-details strong {
        font-weight: 600;
        color: #1abc9c;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        background: rgba(26, 188, 156, 0.2);
        border: 1px solid #1abc9c;
        border-radius: 10px;
        color: white;
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background-color: #1abc9c;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .content-wrapper {
            flex-direction: column;
            align-items: center;
        }
        
        .upload-box, .result-box {
            width: 100%;
            max-width: 600px;
        }
        
        .main-title {
            font-size: 2.2rem;
        }
        
        .main-subtitle {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Load model and disease info
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('models/plant_disease_recog_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_disease_info():
    try:
        with open('plant_disease.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading disease information: {e}")
        return []

def predict_disease(image, model, disease_info):
    # Save uploaded image temporarily to match Flask processing
    import tempfile
    import os
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        # Use EXACT same preprocessing as Flask app
        loaded_image = tf.keras.utils.load_img(temp_path, target_size=(160, 160))
        img_array = tf.keras.utils.img_to_array(loaded_image)
        img_array = np.array([img_array])  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)  # Convert to Python float
        
        return disease_info[predicted_class], confidence
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

# Main app
# Main app exactly like Flask layout
def main():
    # Main content container
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Title and subtitle exactly like Flask
    st.markdown("""
    <h1 class="main-title">Plant Disease Recognition System</h1>
    <p class="main-subtitle">Upload an image of a plant leaf to identify diseases and find cures.</p>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model = load_model()
    disease_info = load_disease_info()
    
    if model is None or not disease_info:
        st.error("Failed to load model or disease information.")
        return
    
    # Content wrapper exactly like Flask
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)
    
    # Create two columns exactly like Flask
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Upload box exactly like Flask
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown('<h2>Analyze Leaf Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose File", 
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.write(f"📁 {uploaded_file.name}")
        else:
            st.write("📁 No file chosen")
        
        # Diagnose button
        diagnose_clicked = st.button('Diagnose', disabled=(uploaded_file is None))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Result box exactly like Flask
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<h2>Diagnosis Result</h2>', unsafe_allow_html=True)
        
        if uploaded_file and diagnose_clicked:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Plant Leaf", width=200)
            
            # Analyze
            with st.spinner('Analyzing...'):
                try:
                    result, confidence = predict_disease(image, model, disease_info)
                    
                    # Display results exactly like Flask
                    st.markdown('<div class="result-content">', unsafe_allow_html=True)
                    
                    st.markdown(f"### 🏷️ Plant: {result['name']}")
                    st.markdown(f"**📊 Confidence:** {confidence:.1f}%")
                    
                    # Progress bar
                    st.progress(confidence/100)
                    
                    # Cause and cure
                    st.markdown("**🦠 Cause:**")
                    st.info(result['cause'])
                    
                    st.markdown("**💊 Cure:**")
                    st.success(result['cure'])
                    
                    # Health status
                    if 'healthy' in result['name'].lower():
                        st.balloons()
                        st.success("🎉 Great News! Your plant appears to be healthy!")
                    else:
                        st.warning("⚠️ Action Required: Please follow the treatment recommendations.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.info("Upload a plant leaf image and click 'Diagnose' to see results here.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close content-wrapper
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-content

if __name__ == "__main__":
    main()