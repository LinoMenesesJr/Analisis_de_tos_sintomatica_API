import os
import shutil
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from services.module1_intake import process_intake
from services.module2_yamnet import process_yamnet
from services.module3_greenarcade import process_greenarcade_transfer
from services.module4_inference import process_inference_and_format

router = APIRouter(prefix="/api/v1", tags=["Audio Analysis"])

@router.post("/analyze-audio")
async def analyze_audio(
    cedula: str = Form(...),
    audio_file: UploadFile = File(...)
):
    try:
        # Module 1: Reception and Preprocessing
        original_filepath, model_ready_audio, sr = await process_intake(audio_file, cedula)

        if len(model_ready_audio) == 0:
            raise HTTPException(status_code=400, detail="El audio provisto está vacío o no es válido.")

        # Module 2: YAMNet Processing and Labelling
        base_filename = os.path.basename(original_filepath).replace(".wav", "")
        yamnet_result = process_yamnet(model_ready_audio, sr, base_filename)

        if yamnet_result is None:
            raise HTTPException(status_code=200, detail="No se detectó tos con suficiente confianza o el audio no tiene energía suficiente.")

        # Module 3: Transfer to GreenArcade
        transfer_filepath = process_greenarcade_transfer(yamnet_result['audio_segment'], sr, base_filename)

        # Module 4: Secondary Inference and Response Formatting
        response_data = process_inference_and_format(transfer_filepath, yamnet_result['mel_spectrogram'])

        return JSONResponse(content={"analisis_tos": response_data})

    except HTTPException as ht_e:
        raise ht_e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup routine
        temp_dirs = ["/tmp/audio_intake", "/tmp/greenarcade_input"]
        for d in temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
