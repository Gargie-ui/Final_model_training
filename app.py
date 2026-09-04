import os
import sys
import io
import json
import flask
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = flask.Flask(__name__, template_folder='templates')

interpreter = None
input_details = None
output_details = None
class_thresholds = None
CLASSES = ["Atelectasis", "Infiltration", "Lung_Tumor", "No_Finding"]

def load_tflite_model():
    global interpreter, input_details, output_details, class_thresholds
    
    tflite_path = os.path.join("SELECTED_4CLASS_DATASET", "experiments", "exp_clahe_focal_threshold", "model.tflite")
    if not os.path.exists(tflite_path):
        tflite_path = os.path.join(os.path.dirname(__file__), "model.tflite")

    print(f"Loading TFLite model from {tflite_path}...")
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.")

    thresholds_path = os.path.join("SELECTED_4CLASS_DATASET", "experiments", "exp_clahe_focal_threshold", "class_thresholds.json")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r") as f:
            class_thresholds = json.load(f)
    else:
        class_thresholds = {"Atelectasis": 0.30, "Infiltration": 0.15, "Lung_Tumor": 0.15, "No_Finding": 0.15}

def apply_clahe_numpy(img_np):
    import cv2
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return rgb

def preprocess_image_bytes(image_bytes):
    import cv2
    img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img_pil)
    
    # 1. Apply CLAHE
    img_rgb = apply_clahe_numpy(img_np)
    
    # 2. Resize to 224x224
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # 3. MobileNetV2 preprocessing (-1.0 to 1.0)
    img_float = img_resized.astype(np.float32) / 127.5 - 1.0
    input_tensor = np.expand_dims(img_float, axis=0)
    return input_tensor

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        load_tflite_model()
        
    if 'file' not in flask.request.files:
        return flask.jsonify({'error': 'No file part'}), 400
        
    file = flask.request.files['file']
    if file.filename == '':
        return flask.jsonify({'error': 'No file selected'}), 400
        
    try:
        image_bytes = file.read()
        input_data = preprocess_image_bytes(image_bytes)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        raw_probs = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Apply threshold decision rule
        thresh_list = [class_thresholds.get(c, 0.20) for c in CLASSES]
        margins = raw_probs - np.array(thresh_list)
        exceeded = margins >= 0
        if np.any(exceeded):
            masked = np.where(exceeded, margins, -np.inf)
            pred_idx = int(np.argmax(masked))
        else:
            pred_idx = int(np.argmax(raw_probs))
            
        final_diagnosis = CLASSES[pred_idx]
        confidences = {CLASSES[i]: float(raw_probs[i]) for i in range(len(CLASSES))}
        
        return flask.jsonify({
            'success': True,
            'prediction': final_diagnosis,
            'confidence': float(raw_probs[pred_idx]),
            'probabilities': confidences
        })
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500

# Initialize on startup
load_tflite_model()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
