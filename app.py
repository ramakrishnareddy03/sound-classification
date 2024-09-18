from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
app = Flask(__name__)
model = load_model('cnnmodel.keras')
def extract_feature(file_name):
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1: X = X[:, 0]
    X = X.T
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = 'temp_audio_file.ogg'
        file.save(file_path)
        features = extract_feature(file_path)
        features = np.expand_dims(features, axis=0)
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions, axis=1)[0]
        pred = {1:'Dog Bark',2:'Rain',3:'Sea waves',4:'Baby cry',5:'Clock tick',
                6:'Person Sneeze',7:'Helicopter',8:'Chainsaw',9:'Rooster',10:'Fire crackling'}
        return jsonify({'predicted_class': pred[int(predicted_class)+2]})
if __name__ == '__main__':
    app.run(debug=True)
