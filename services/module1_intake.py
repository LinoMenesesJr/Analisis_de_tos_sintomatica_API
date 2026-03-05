import os
from datetime import datetime
from fastapi import UploadFile
import librosa
import numpy as np

async def process_intake(audio_file: UploadFile, cedula: str):
    """
    Module 1: Reception and Preprocessing.
    Receives payload and metadata. Renames and saves to /tmp/audio_intake.
    Transforms to 16kHz for YAMNet.
    """
    intake_dir = "/tmp/audio_intake"
    os.makedirs(intake_dir, exist_ok=True)
    
    # Format: Cedula_fecha_hora_01 (using 01 as an initial static suffix per requirement or sequence)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{cedula}_{now_str}_01.wav"
    filepath = os.path.join(intake_dir, filename)
    
    # Save Uploaded file
    with open(filepath, "wb") as f:
        content = await audio_file.read()
        f.write(content)
        
    # Transform file exactly as base code for YAMNet (16kHz, Mono, Normalized)
    wav16, sr = librosa.load(filepath, sr=16000, mono=True)
    
    if len(wav16) > 0:
        peak = np.max(np.abs(wav16))
        if peak > 0:
            wav16 = wav16 / peak
            
    return filepath, wav16, sr

