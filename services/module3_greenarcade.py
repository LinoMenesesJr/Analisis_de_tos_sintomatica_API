import os
import soundfile as sf
import numpy as np

def process_greenarcade_transfer(audio_segment, sr, base_filename):
    """
    Module 3: Transfer to GreenArcade.
    Ensures segments are exactly 1 second long and saves to `/tmp/greenarcade_input`
    """
    output_dir = "/tmp/greenarcade_input"
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure exactly 1 second duration
    target_length = sr * 1  # 1 second = sample rate * 1
    
    if len(audio_segment) > target_length:
        # crop to exactly 1 second center
        start = (len(audio_segment) - target_length) // 2
        segment_1s = audio_segment[start:start + target_length]
    elif len(audio_segment) < target_length:
        # pad to exactly 1 second
        padding = target_length - len(audio_segment)
        pad_left = padding // 2
        pad_right = padding - pad_left
        segment_1s = np.pad(audio_segment, (pad_left, pad_right), mode='constant')
    else:
        segment_1s = audio_segment

    # Base filename includes _tos from the router / previous steps
    filepath = os.path.join(output_dir, f"{base_filename}_tos.wav")
    
    sf.write(filepath, segment_1s, sr, subtype='PCM_16')
    
    return filepath
