from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import librosa
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename
import tempfile
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define genres (same as in your notebook)
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load trained models
def load_models():
    """Load all trained models"""
    models = {}
    try:
        models['rf'] = joblib.load('models/rf_clf.joblib')
        models['gb'] = joblib.load('models/gb_clf.joblib')
        models['svm'] = joblib.load('models/svm_clf.joblib')
        models['lr'] = joblib.load('models/lr_clf.joblib')
        models['ensemble'] = joblib.load('models/ensemble_clf.joblib')
        print("All models loaded successfully!")
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# Load models on startup
models = load_models()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    """Extract audio features (same as in your notebook)"""
    try:
        y, sr = librosa.load(file_path, duration=30)
        features = {}

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i+1}_std'] = np.std(chroma[i])

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(7):
            features[f'contrast_{i+1}_mean'] = np.mean(contrast[i])
            features[f'contrast_{i+1}_std'] = np.std(contrast[i])

        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_genre(features, model_name='ensemble'):
    """Predict genre using specified model"""
    if models is None or model_name not in models:
        return None, None
    
    try:
        # Convert features to DataFrame with proper column order
        feature_df = pd.DataFrame([features])
        
        # Get the model
        model = models[model_name]
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_df)[0]
            confidence_scores = dict(zip(genres, probabilities))
        else:
            confidence_scores = {prediction: 1.0}
        
        return prediction, confidence_scores
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', genres=genres)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    model_choice = request.form.get('model', 'ensemble')
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            # Extract features
            features = extract_features(temp_path)
            
            # Clean up uploaded file
            os.remove(temp_path)
            
            if features is None:
                flash('Error processing audio file')
                return redirect(url_for('index'))
            
            # Make prediction
            prediction, confidence_scores = predict_genre(features, model_choice)
            
            if prediction is None:
                flash('Error making prediction')
                return redirect(url_for('index'))
            
            # Sort confidence scores
            sorted_scores = sorted(confidence_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
            
            return render_template('result.html', 
                                 prediction=prediction,
                                 confidence_scores=sorted_scores,
                                 model_used=model_choice,
                                 filename=filename)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload WAV, MP3, FLAC, M4A, or OGG files.')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    model_choice = request.form.get('model', 'ensemble')
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            
            # Extract features
            features = extract_features(temp_file.name)
            
            # Clean up
            os.unlink(temp_file.name)
            
            if features is None:
                return jsonify({'error': 'Could not extract features'}), 500
            
            # Make prediction
            prediction, confidence_scores = predict_genre(features, model_choice)
            
            if prediction is None:
                return jsonify({'error': 'Could not make prediction'}), 500
            
            return jsonify({
                'prediction': prediction,
                'confidence_scores': confidence_scores,
                'model_used': model_choice
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = {name: model is not None for name, model in models.items()} if models else {}
    return jsonify({
        'status': 'healthy' if models else 'unhealthy',
        'models_loaded': model_status
    })

if __name__ == '__main__':
    if models is None:
        print("Warning: Models not loaded. Please make sure model files exist in the 'models' directory.")
    app.run(debug=True, host='0.0.0.0', port=5000)