from flask import Flask, render_template,request,redirect,send_from_directory,url_for,jsonify
import numpy as np
import json
import uuid
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import TensorFlow with fallback
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logger.info("TensorFlow imported successfully")
except ImportError as e:
    logger.error(f"Failed to import TensorFlow: {e}")
    tf = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model with error handling
try:
    if tf is None:
        raise Exception("TensorFlow not available")
    model_path = "models/plant_disease_recog_model.keras"
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('uploadimages', filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/',methods = ['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    if tf is None:
        raise Exception("TensorFlow not available")
    image = tf.keras.utils.load_img(image,target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    if model is None:
        raise Exception("Model not loaded")
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

@app.route('/upload/',methods = ['POST','GET'])
def uploadimage():
    if request.method == "POST":
        try:
            image = request.files['img']
            if not image or image.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                })
            
            upload_folder = 'uploadimages'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
                
            filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
            image_path = os.path.join(upload_folder, filename)
            image.save(image_path)
            
            print(image_path)
            prediction = model_predict(image_path)
            
            # Clean up uploaded file after prediction
            try:
                os.remove(image_path)
            except:
                pass  # If file deletion fails, continue
            
            image_url = url_for('uploaded_images', filename=filename)
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'imagepath': image_url
            })
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    return redirect('/')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
