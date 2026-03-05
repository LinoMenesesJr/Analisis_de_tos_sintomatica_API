"""
modulo_7_dual_pipeline.py
Pipeline Dual "Portero y Experto" — YAMNet (filtro) + GreenArcade (diagnóstico)
"""
import os, shutil, pickle, warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ═══════════════════════════════════════════════════════
#  Extracción de Features para GreenArcade (Random Forest)
# ═══════════════════════════════════════════════════════
def extract_all_features(y, sr=22050):
    """Extrae features in-memory (sin disco) para GreenArcade."""
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


class DualPipeline:
    def __init__(self):
        self.breathing_dir = Path("/home/user13/Documents/PruebaTOS/Datasets/breathing/open")
        self.coughvid_dir  = Path("/home/user13/Documents/PruebaTOS/Datasets/coughvid-3")
        self.output_dir    = Path("/home/user13/Documents/PruebaTOS/Resultados_Prueba_Dual")
        self.categories    = ['COVID-19', 'healthy', 'symptomatic']
        self.yamnet = None
        self.ga_model = self.ga_scaler = self.ga_encoder = self.ga_fnames = None
        self._breathing_pool = []
        self._cough_pool = []

    # ── Tarea 0: Preparación ──────────────────────────
    def tarea0_preparacion(self):
        print("═══ TAREA 0: Preparación de Modelos y Entorno ═══")
        # GreenArcade
        pkl_path = Path("/home/user13/Documents/PruebaTOS/cough_classification_model.pkl")
        if not pkl_path.exists():
            from huggingface_hub import hf_hub_download
            print("  Descargando modelo GreenArcade...")
            hf_hub_download(repo_id="greenarcade/cough-classification-model",
                            filename="cough_classification_model.pkl", local_dir=".")
        with open(pkl_path, 'rb') as f:
            c = pickle.load(f)
        self.ga_model   = c['model']
        self.ga_scaler  = c['scaler']
        self.ga_encoder = c['label_encoder']
        self.ga_fnames  = c['feature_names']
        print(f"  GreenArcade OK — Clases: {list(self.ga_encoder.classes_)}")

        # YAMNet
        print("  Cargando YAMNet...")
        self.yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
        print("  YAMNet OK")

        # Directorios de salida
        for cat in self.categories:
            d = self.output_dir / cat
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        print(f"  Directorios creados en {self.output_dir}")

        # Pre-escanear pools de archivos válidos
        print("  Escaneando archivos válidos (>0.5s)...")
        self._breathing_pool = self._scan_valid(self.breathing_dir)
        self._cough_pool     = self._scan_valid(self.coughvid_dir)
        print(f"  Breathing válidos: {len(self._breathing_pool)}")
        print(f"  CoughVid válidos:  {len(self._cough_pool)}")

    def _scan_valid(self, base_dir, min_dur=0.5):
        valid = []
        for f in base_dir.rglob("*.wav"):
            try:
                dur = librosa.get_duration(path=str(f))
                if dur >= min_dur:
                    valid.append(str(f))
            except:
                pass
        return valid

    # ── Procesamiento atómico de un audio ─────────────
    def _process_one(self, ruta, origen):
        nombre = Path(ruta).name
        # Cargar 16kHz para YAMNet
        wav16, _ = librosa.load(ruta, sr=16000, mono=True)
        if len(wav16) == 0:
            return None
        peak = np.max(np.abs(wav16))
        if peak > 0:
            wav16 = wav16 / peak

        # YAMNet inference
        scores, _, _ = self.yamnet(wav16)
        scores_np = scores.numpy()
        cough_scores = scores_np[:, 42]
        max_prob = float(np.max(cough_scores))
        best_frame = int(np.argmax(cough_scores))
        sr = 16000

        # ── FILTRO ESTRICTO: Solo pasan audios con tos detectada ──
        UMBRAL_TOS = 0.65
        if max_prob < UMBRAL_TOS:
            return None  # Descartado: YAMNet no detectó tos con suficiente confianza

        # Segmentación: recortar 1.5s centrado en el pico de tos
        center = best_frame * 0.48
        t0 = max(0.0, center - 0.75)
        t1 = min(len(wav16) / sr, center + 0.75)

        seg = wav16[int(t0 * sr):int(t1 * sr)]
        if len(seg) < sr // 4:
            seg = wav16

        # ── FILTRO DE ENERGÍA MÍNIMA: descartar silencio/ruido blanco ──
        seg_rms = float(np.sqrt(np.mean(seg ** 2)))
        if seg_rms < 0.01:
            return None  # Descartado: segmento con energía insuficiente

        # GreenArcade: extraer features del segmento a 22050
        seg_22k = librosa.resample(seg, orig_sr=16000, target_sr=22050)
        feat = extract_all_features(seg_22k, sr=22050)
        if feat is None:
            return None

        feat_df = pd.DataFrame([feat])
        for col in self.ga_fnames:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[self.ga_fnames]
        feat_scaled = self.ga_scaler.transform(feat_df)

        pred_idx = self.ga_model.predict(feat_scaled)[0]
        pred_label = self.ga_encoder.inverse_transform([pred_idx])[0]
        proba = self.ga_model.predict_proba(feat_scaled)[0]

        # Guardar segmento en carpeta de resultado
        dest = self.output_dir / pred_label / nombre
        sf.write(str(dest), seg, sr, subtype='PCM_16')

        classes = list(self.ga_encoder.classes_)
        return {
            'Archivo': nombre,
            'Origen': origen,
            'Confianza_YAMNet': round(max_prob, 4),
            'Prediccion_GreenArcade': pred_label,
            'P_COVID': round(float(proba[classes.index('COVID-19')]), 4),
            'P_healthy': round(float(proba[classes.index('healthy')]), 4),
            'P_symptomatic': round(float(proba[classes.index('symptomatic')]), 4),
        }

    # ── Tarea 1: Smoke Test (5+5) ────────────────────
    def tarea1_smoke_test(self):
        print("\n═══ TAREA 1: Smoke Test (5+5) ═══")
        rng = np.random.RandomState(RANDOM_SEED)
        sel_b = list(rng.choice(self._breathing_pool, size=min(5, len(self._breathing_pool)), replace=False))
        sel_c = list(rng.choice(self._cough_pool, size=min(5, len(self._cough_pool)), replace=False))

        results = []
        for p in sel_b:
            r = self._process_one(p, 'breathing')
            if r: results.append(r)
        for p in sel_c:
            r = self._process_one(p, 'coughvid')
            if r: results.append(r)

        df = pd.DataFrame(results)
        print("\n  ┌─ Resultados Smoke Test ───────────────────────────┐")
        for _, row in df.iterrows():
            print(f"  │ {row['Archivo']:30s} │ {row['Origen']:10s} → {row['Prediccion_GreenArcade']:12s} │ YAMNet:{row['Confianza_YAMNet']:.2f} │")
        print("  └──────────────────────────────────────────────────┘")
        print(f"  Smoke Test OK: {len(df)}/10 procesados correctamente.\n")
        return df

    # ── Tarea 2+3: Muestreo Masivo + Pipeline ────────
    def tarea2_3_masivo(self):
        print("═══ TAREA 2: Muestreo Masivo (100+100) ═══")
        rng = np.random.RandomState(RANDOM_SEED + 1)  # Semilla diferente al smoke test
        sel_b = list(rng.choice(self._breathing_pool, size=min(100, len(self._breathing_pool)), replace=False))
        sel_c = list(rng.choice(self._cough_pool, size=min(100, len(self._cough_pool)), replace=False))

        all_items = [(p, 'breathing') for p in sel_b] + [(p, 'coughvid') for p in sel_c]
        rng.shuffle(all_items)

        print(f"  Total a procesar: {len(all_items)} audios")
        print("\n═══ TAREA 3: Pipeline Atómico ═══")
        results = []
        for path, origen in tqdm(all_items, desc="Pipeline Dual"):
            try:
                r = self._process_one(path, origen)
                if r:
                    results.append(r)
            except Exception as e:
                print(f"  Error: {Path(path).name}: {e}")

        self.df_results = pd.DataFrame(results)
        print(f"  Procesados exitosamente: {len(self.df_results)}/{len(all_items)}")

    # ── Tarea 4: Reporte Técnico ─────────────────────
    def tarea4_reporte(self):
        print("\n═══ TAREA 4: Consolidación y Log ═══")
        csv_path = self.output_dir / "reporte_final_dual.csv"
        self.df_results.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"  CSV guardado: {csv_path}")

        counts = self.df_results['Prediccion_GreenArcade'].value_counts()
        print("\n╔═══════════════════════════════════════════════════╗")
        print("║       RESUMEN FINAL DE DISTRIBUCIÓN              ║")
        print("╠═══════════════════════════════════════════════════╣")
        for cat in self.categories:
            n = counts.get(cat, 0)
            print(f"║  {cat:20s} : {n:4d} audios                  ║")
        print(f"║  {'TOTAL':20s} : {len(self.df_results):4d} audios                  ║")
        print("╚═══════════════════════════════════════════════════╝")

        print("\n  Desglose por Origen:")
        cross = pd.crosstab(self.df_results['Origen'], self.df_results['Prediccion_GreenArcade'])
        print(cross.to_string())
        print()

    def execute(self):
        self.tarea0_preparacion()
        self.tarea1_smoke_test()
        self.tarea2_3_masivo()
        self.tarea4_reporte()


if __name__ == '__main__':
    pipeline = DualPipeline()
    pipeline.execute()
