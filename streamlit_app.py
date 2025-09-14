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
    # Resize image to model's expected input size
    image = image.resize((160, 160))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return disease_info[predicted_class], confidence

# Main app
def main():
    # Header
    st.title("🌱 Plant Disease Recognition System")
    st.markdown("---")
    st.write("Upload an image of a plant leaf to identify diseases and get treatment recommendations.")
    st.write("**Supported plants:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato")
    
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
                        st.progress(confidence/100)
                        
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