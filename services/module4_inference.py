import base64
import pickle
import librosa
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Load GreenArcade globally
print("Descargando/Cargando modelo GreenArcade...")
pkl_path = "cough_classification_model.pkl"
try:
    with open(pkl_path, 'rb') as f:
        c = pickle.load(f)
except FileNotFoundError:
    hf_hub_download(repo_id="greenarcade/cough-classification-model",
                    filename="cough_classification_model.pkl", local_dir=".")
    with open(pkl_path, 'rb') as f:
        c = pickle.load(f)

ga_model = c['model']
ga_scaler = c['scaler']
ga_encoder = c['label_encoder']
ga_fnames = c['feature_names']
print("GreenArcade OK")

def extract_all_features(y, sr=22050):
    """Extrae features in-memory (sin disco) para GreenArcade (del script original)."""
    if len(y) == 0:
        return None
    features = {}
    # Temporal
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'], features['rms_std'] = np.mean(rms), np.std(rms)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'], features['zcr_std'] = np.mean(zcr), np.std(zcr)
    # Spectral
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'], features['spectral_centroid_std'] = np.mean(sc), np.std(sc)
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'], features['spectral_bandwidth_std'] = np.mean(sb), np.std(sb)
    scon = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(scon.shape[0]):
        features[f'spectral_contrast_{i}_mean'] = np.mean(scon[i])
        features[f'spectral_contrast_{i}_std'] = np.std(scon[i])
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['rolloff_mean'], features['rolloff_std'] = np.mean(ro), np.std(ro)
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        features[f'chroma_{i}_std'] = np.std(chroma[i])
    return features


def process_inference_and_format(audio_filepath, mel_spectrogram_base64):
    """
    Module 4: Secondary Inference and Response Formatting.
    Runs GreenArcade inference, reads audio to Base64, and crafts the JSON output.
    """
    # Load the 1 second audio at 22050 for GreenArcade
    seg_22k, sr_22k = librosa.load(audio_filepath, sr=22050, mono=True)
    
    feat = extract_all_features(seg_22k, sr=sr_22k)
    
    if feat is None:
        raise ValueError("No se pudieron extraer los features de audio.")
        
    feat_df = pd.DataFrame([feat])
    for col in ga_fnames:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
            
    feat_df = feat_df[ga_fnames]
    feat_scaled = ga_scaler.transform(feat_df)

    pred_idx = ga_model.predict(feat_scaled)[0]
    pred_label = ga_encoder.inverse_transform([pred_idx])[0]
    proba = ga_model.predict_proba(feat_scaled)[0]
    
    # 'valordecerteza' equals the probability of the predicted class
    certainty = float(np.max(proba))

    if certainty < 0.65:
        pred_label = "Tos detectada pero sin certeza de diagnostico"

    # Read the audio into Base64
    with open(audio_filepath, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')

    response_data = {
        "audio": audio_b64,
        "diagrama": mel_spectrogram_base64,
        "diagnostico": pred_label,
        "certeza": certainty
    }
    
    return response_data
