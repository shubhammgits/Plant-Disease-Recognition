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
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    }
    .uploadedFile {
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #1abc9c, #16a085);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #1abc9c;
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
def main():
    # Header with Flask-like styling
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(0,0,0,0.1); border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: #1abc9c; font-size: 3rem; margin-bottom: 1rem;'>🌱 Plant Disease Recognition System</h1>
        <p style='font-size: 1.2rem; color: #ecf0f1;'>Upload an image of a plant leaf to identify diseases and get treatment recommendations.</p>
        <p style='font-size: 1rem; color: #bdc3c7;'><strong>Supported plants:</strong> Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    with st.spinner('Loading AI model...'):
        model = load_model()
        disease_info = load_disease_info()
    
    if model is None or not disease_info:
        st.error("Failed to load model or disease information. Please check the files.")
        return
    
    st.success("✅ AI model loaded successfully!")
    st.markdown("---")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 Model Information")
        st.write("**Accuracy:** 97.11%")
        st.write("**Model Type:** CNN")
        st.write("**Input Size:** 160x160 pixels")
        st.write("**Diseases:** 39 types")
        st.write("**Plants:** 14 species")
        
        st.header("📖 How to Use")
        st.write("1. Upload a clear image of a plant leaf")
        st.write("2. Click 'Analyze Disease'")
        st.write("3. Get instant diagnosis and treatment")
        
        st.header("💡 Tips")
        st.write("• Use clear, well-lit images")
        st.write("• Focus on the leaf")
        st.write("• Avoid blurry photos")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf for disease analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, caption='Your Plant Leaf Image', use_column_width=True)
        
        with col2:
            st.subheader("🔍 Analysis")
            
            # Analyze button
            if st.button('🔬 Analyze Disease', type="primary"):
                with st.spinner('Analyzing your plant...'):
                    try:
                        result, confidence = predict_disease(image, model, disease_info)
                        
                        # Display results
                        st.success('✅ Analysis Complete!')
                        
                        # Results in an attractive format
                        st.markdown("### 📋 Diagnosis Results")
                        
                        # Disease name with confidence
                        st.markdown(f"**🏷️ Disease:** {result['name']}")
                        st.markdown(f"**📊 Confidence:** {confidence:.1f}%")
                        
                        # Progress bar for confidence
                        st.progress(float(confidence/100))
                        
                        # Cause and treatment
                        st.markdown("**🦠 Cause:**")
                        st.info(result['cause'])
                        
                        st.markdown("**💊 Treatment:**")
                        st.success(result['cure'])
                        
                        # Health status indicator
                        if 'healthy' in result['name'].lower():
                            st.balloons()
                            st.markdown("### 🎉 Great News!")
                            st.success("Your plant appears to be healthy!")
                        else:
                            st.markdown("### ⚠️ Action Required")
                            st.warning("Please follow the treatment recommendations above.")
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🌿 About This System")
    st.write("This AI-powered system uses deep learning to identify plant diseases with 97.11% accuracy. "
             "It can recognize 39 different diseases across 14 plant species, helping farmers and gardeners "
             "make informed decisions about plant health management.")
    
    # Additional information
    with st.expander("📚 Supported Diseases and Plants"):
        st.write("**Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy")
        st.write("**Corn:** Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy")
        st.write("**Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy")
        st.write("**Grape:** Black Rot, Esca, Leaf Blight, Healthy")
        st.write("**And many more...**")

if __name__ == "__main__":
    main()