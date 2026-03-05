# Guía de Integración: API Detección de Tos

Esta guía documenta cómo consumir la API de Detección de Tos desde otros proyectos o microservicios.

## Información General del Servicio

*   **URL Base:** `http://localhost:5005` (o la IP/dominio donde se despliegue el contenedor Docker).
*   **Endpoint Principal:** `/api/v1/analyze-audio`
*   **Método Permitido:** `POST`
*   **Tipo de Contenido a enviar:** `multipart/form-data`

## ¿Qué espera la API?

El endpoint requiere siempre el envío de dos (2) campos en el formulario:

1.  **`cedula`** (Campo de texto/string): La cédula o un identificador de paciente. Se usará para firmar e identificar temporalmente al archivo internamente. (Ej. `"123456"`).
2.  **`audio_file`** (Archivo binario): El archivo de audio (idealmente en formato `.wav`). 

> [!WARNING]
> La API descartará automáticamente los silencios o audios sin tos relevante con error 200 detallando el incidente. Por otro lado, si la inferencia de GreenArcade produce una estimación de certeza menor al `65% (0.65)`, la API seguirá retornando la respuesta exitosa (Http 200), pero el campo de `diagnostico` dictará específicamente: `"Tos detectada pero sin certeza de diagnostico"`.

## Estructura de la Respuesta Exitoso (Status 200)

Si la detección es exitosa (pasó el análisis estricto de YAMNet y superó el umbral de certeza de GreenArcade), la API retorna el siguiente objeto JSON:

```json
{
  "analisis_tos": {
    "audio": "<cadena_string_base64>",
    "diagrama": "<cadena_string_base64>",
    "diagnostico": "COVID-19 | healthy | symptomatic",
    "certeza": 0.9854
  }
}
```

*   **`audio`**: Archivo codificado a Base64 del segmento exacto de **1 segundo** que usó el modelo para diagnosticarte.
*   **`diagrama`**: La imagen (espectrograma Mel) PNG codificada en formato Base64.
*   **`diagnostico`**: La etiqueta resultante tras evaluar el audio.
*   **`certeza`**: Porcentaje probabilístico asignado por el modelo Random Forest al diagnóstico.

---

## 💻 Ejemplos de Implementación

### Opción 1: Python (usando la librería `requests`)

La manera más común para conectar servicios backend (ej. Flask conectando a esta API FastAPI):

```python
import requests
import json

def analizar_audio(ruta_archivo_wav, numero_cedula):
    url = "http://localhost:5005/api/v1/analyze-audio"
    
    # Preparamos los datos del formulario (string) y el archivo binario
    data = {
        "cedula": numero_cedula
    }
    
    # Abrimos el archivo en modo lectura binaria ('rb')
    with open(ruta_archivo_wav, 'rb') as f:
        files = {
            "audio_file": (ruta_archivo_wav, f, "audio/wav")
        }
        
        try:
            # Enviamos la petición POST multipart/form-data automáticamente
            response = requests.post(url, data=data, files=files)
            
            # Si el modelo encontró certeza baja (HTTP 400), o no detectó tos
            if response.status_code != 200:
                print(f"La API respondió con error o advertencia: {response.json()}")
                return None
                
            # Procesar el resultado favorable
            json_response = response.json()
            return json_response['analisis_tos']
            
        except requests.exceptions.RequestException as e:
            print(f"Fallo de conexión crítico contra la API: {e}")
            return None

# Llamada de ejemplo:
resultado = analizar_audio("mi_grabacion.wav", "24584794")
if resultado:
    print(f"Diagnóstico Final: {resultado['diagnostico']} ({resultado['certeza'] * 100}%)")
```

### Opción 2: JavaScript Moderno (Fetch API / Frontend o Node.js)

```javascript
async function mandarAudioAAnalizar(fileBlobObject, cedula) {
    const url = 'http://localhost:5005/api/v1/analyze-audio';
    
    // Creamos la estructura multipart/form-data nativa
    const formData = new FormData();
    formData.append('cedula', cedula);
    formData.append('audio_file', fileBlobObject);
    
    try {
        const fetchResponse = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        const data = await fetchResponse.json();
        
        if (!fetchResponse.ok) {
            console.error("No se pudo analizar con fiabilidad:", data.detail);
            return null;
        }
        
        console.log("¡Análisis Terminado!");
        console.log("Diagnosticado como:", data.analisis_tos.diagnostico);
        console.log("Certeza:", data.analisis_tos.certeza);
        
        return data.analisis_tos;
        
    } catch(err) {
        console.error("El servidor de la API está caído o no accesible:", err);
    }
}
```

### Opción 3: Curl (Por consola)

Ideal para pruebas directas desde tu terminal bash/zsh sin escribir código:

```bash
curl -X POST "http://localhost:5005/api/v1/analyze-audio" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "cedula=123456" \
     -F "audio_file=@/ruta/a/tu/audio/prueba.wav"
```

---

## 💡 ¿Cómo decodificar los Base64 de respuesta?

A veces tu nuevo proyecto no querrá imprimir las cadenas de super texto (Base64) sino guardar el audio real validado o mandar esta imagen a AWS S3. Aquí te explico cómo guardarlas en disco usando **Python**:

```python
import base64

# Asumiendo que 'resultado' es el bloque "analisis_tos" del JSON dict devuelto
audio_base64_string = resultado['audio']
imagen_base64_string = resultado['diagrama']

# 1. Guardar o reconstruir el archivo de Audio (Wav Ogg) reconstruido y filtrado
with open("audio_reconstruido_paciente.wav", "wb") as f:
    f.write(base64.b64decode(audio_base64_string))

# 2. Guardar imagen diagrama Espectrograma devuelta (PNG)
with open("espectrograma_paciente.png", "wb") as f:
    f.write(base64.b64decode(imagen_base64_string))
```
