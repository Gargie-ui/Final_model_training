import os
import sys
import json
import flask
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Ensure the workspace path is available for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config
from src.threshold_tuning import predict_with_thresholds

app = flask.Flask(__name__, template_folder='templates')

# Global variables for model, classes, and thresholds
model = None
class_thresholds = None
classes = config.CLASSES

def load_model_safely():
    """
    Loads the trained model and per-class decision thresholds into memory.
    """
    global model, class_thresholds
    model_path = os.path.join(config.MODELS_DIR, "final_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")
    
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully and compiled.")
    
    # Load per-class decision thresholds (tuned on validation set)
    thresholds_path = os.path.join(config.MODELS_DIR, "class_thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            class_thresholds = json.load(f)
        print(f"Loaded per-class thresholds: {class_thresholds}")
    else:
        print("WARNING: class_thresholds.json not found. Falling back to plain argmax predictions.")
        class_thresholds = None

def preprocess_image(image_bytes):
    """
    Decodes the raw uploaded image bytes, applies CLAHE (if enabled),
    converts to 3-channel RGB, resizes to target IMG_SIZE (224x224), and normalizes.
    This exactly mirrors the preprocessing pipeline used in training.
    """
    # 1. Decode bytes into a 1-channel grayscale tensor
    image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
    
    # 1.5 Apply CLAHE preprocessing BEFORE resizing and BEFORE preprocess_input
    if getattr(config, "USE_CLAHE", False):
        from src.preprocessing import apply_clahe_tf
        image = apply_clahe_tf(image)
        
    # 2. Replicate grayscale channel to 3-channel RGB
    image = tf.image.grayscale_to_rgb(image)
    
    # 3. Cast to float32
    image = tf.cast(image, tf.float32)
    
    # 4. Resize to configuration size (224, 224)
    image = tf.image.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    
    # 5. Apply MobileNetV2-specific preprocessing (-1.0 to 1.0 scaling)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    # 6. Add batch dimension: shape becomes (1, 224, 224, 3)
    image = tf.expand_dims(image, axis=0)
    
    return image

@app.route('/')
def index():
    """
    Serves the main testing frontend HTML page.
    """
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an uploaded image file, runs inference, and returns prediction
    confidences with threshold-tuned final prediction.
    """
    if model is None:
        return flask.jsonify({'error': 'Model is not loaded.'}), 500
        
    if 'file' not in flask.request.files:
        return flask.jsonify({'error': 'No file part in request.'}), 400
        
    file = flask.request.files['file']
    if file.filename == '':
        return flask.jsonify({'error': 'No file selected.'}), 400
        
    try:
        image_bytes = file.read()
        
        # Preprocess and execute prediction
        processed_tensor = preprocess_image(image_bytes)
        raw_probs = model.predict(processed_tensor)[0]
        
        # Use relative softmax probabilities (argmax) for unbiased multi-class prediction
        max_idx = int(np.argmax(raw_probs))
        predicted_class = classes[max_idx]
        prediction_method = "softmax-argmax"
        
        # Format prediction percentages
        results = []
        for i, class_name in enumerate(classes):
            results.append({
                'class': class_name,
                'probability': float(raw_probs[i])
            })
            
        # Sort results from highest probability to lowest
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        return flask.jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'prediction_method': prediction_method,
            'predictions': results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return flask.jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model upon startup
    try:
        load_model_safely()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have trained the model and saved final_model.keras before running.")
        sys.exit(1)
        
    # Start local server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=False)
