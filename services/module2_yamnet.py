import io
import base64
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow_hub as hub

# Load YAMNet Model globally for the module
print("Cargando YAMNet...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet OK")

def process_yamnet(wav16, sr, base_filename):
    """
    Module 2: YAMNet Processing and Labelling
    Loads YAMNet, extracts cough segment, applies `_tos` label, generates Mel spectrogram.
    """
    if len(wav16) == 0:
        return None
        
    # YAMNet inference
    scores, embeddings, spectrogram = yamnet(wav16)
    scores_np = scores.numpy()
    cough_scores = scores_np[:, 42]  # Class 42 is 'Cough'
    max_prob = float(np.max(cough_scores))
    best_frame = int(np.argmax(cough_scores))
    
    # FILTRO ESTRICTO
    UMBRAL_TOS = 0.65
    if max_prob < UMBRAL_TOS:
        return None  # YAMNet didn't detect cough with enough confidence

    # Segmentation: crop 1.5s centered on cough peak (per base code)
    center = best_frame * 0.48
    t0 = max(0.0, center - 0.75)
    t1 = min(len(wav16) / sr, center + 0.75)

    seg = wav16[int(t0 * sr):int(t1 * sr)]
    if len(seg) < sr // 4:
        seg = wav16

    # FILTRO DE ENERGÍA MÍNIMA: discard silence/white noise
    seg_rms = float(np.sqrt(np.mean(seg ** 2)))
    if seg_rms < 0.01:
        return None 
        
    labeled_filename = f"{base_filename}_tos"
        
    # Generate Mel spectrogram diagram
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=f'Espectrograma Mel para {labeled_filename}')
    
    # Save plot to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    mel_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return {
        'audio_segment': seg,
        'mel_spectrogram': mel_base64,
        'labeled_filename': labeled_filename
    }
