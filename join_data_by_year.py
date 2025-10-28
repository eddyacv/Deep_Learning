import os
import numpy as np
import json
import pickle

# ===============================
# CONFIGURACIÓN
# ===============================

# Temporada que quieres procesar
SEASON = "2016-2017"  # <-- cambia esto para otras temporadas

# Carpeta base de features y raws (ajusta según tu estructura)
BASE_FEATURES_PATH = f"data/features/england_epl/{SEASON}/"
BASE_RAW_PATH = f"data/raw/england_epl/{SEASON}/"

# Archivo de salida dinámico según temporada
OUTPUT_FILE = f"data/dataset/england_epl_{SEASON}.pkl"

# ===============================
# FUNCIONES
# ===============================

def load_match_data(features_path, raw_path):
    """
    Carga los features (.npy) y labels (.json) de un partido.
    Retorna:
        X: lista de arrays de features de cada cámara o fuente
        y: diccionario con labels
    """
    # Cargar todos los .npy de la carpeta
    feature_files = [f for f in os.listdir(features_path) if f.endswith('.npy')]
    feature_files.sort()  # opcional: mantener orden 1_, 2_, ...

    X_list = []
    for f in feature_files:
        full_path = os.path.join(features_path, f)
        arr = np.load(full_path)
        X_list.append(arr)

    # Cargar labels
    json_file = os.path.join(raw_path, "Labels-v2.json")
    with open(json_file, "r", encoding="utf-8") as jf:
        y = json.load(jf)

    return X_list, y

# ===============================
# RECORRER TODAS LAS CARPETAS
# ===============================

all_X = []
all_y = []

for match_folder in os.listdir(BASE_FEATURES_PATH):
    match_features_path = os.path.join(BASE_FEATURES_PATH, match_folder)
    match_raw_path = os.path.join(BASE_RAW_PATH, match_folder)

    if os.path.isdir(match_features_path) and os.path.isdir(match_raw_path):
        X, y = load_match_data(match_features_path, match_raw_path)
        all_X.append(X)
        all_y.append(y)
        shapes = [arr.shape for arr in X]
        print(f"Cargado partido: {match_folder}, features shapes: {shapes}")

# ===============================
# GUARDAR DATASET
# ===============================

# Guardar usando pickle para manejar listas de arrays de distintos tamaños
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)  # crea carpeta si no existe
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"X": all_X, "y": all_y}, f)

print(f"Dataset guardado en {OUTPUT_FILE}, partidos cargados: {len(all_X)}")
