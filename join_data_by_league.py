import os
import numpy as np
import json
import pickle

# ===============================
# CONFIGURACIÓN
# ===============================

# Liga que quieres procesar
LEAGUE = "england_epl"

# Carpeta base de features y raws
BASE_FEATURES_PATH = f"data/features/{LEAGUE}/"
BASE_RAW_PATH = f"data/raw/{LEAGUE}/"

# Archivo de salida (dataset por liga)
OUTPUT_FILE = f"data/dataset/{LEAGUE}.pkl"

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
    feature_files = [f for f in os.listdir(features_path) if f.endswith('.npy')]
    feature_files.sort()

    X_list = [np.load(os.path.join(features_path, f)) for f in feature_files]

    json_file = os.path.join(raw_path, "Labels-v2.json")
    with open(json_file, "r", encoding="utf-8") as jf:
        y = json.load(jf)

    return X_list, y

# ===============================
# RECORRER TODAS LAS TEMPORADAS Y PARTIDOS
# ===============================

all_X = []
all_y = []

# Listar todas las temporadas en la liga
seasons = [s for s in os.listdir(BASE_FEATURES_PATH) if os.path.isdir(os.path.join(BASE_FEATURES_PATH, s))]
seasons.sort()

for season in seasons:
    season_features_path = os.path.join(BASE_FEATURES_PATH, season)
    season_raw_path = os.path.join(BASE_RAW_PATH, season)

    for match_folder in os.listdir(season_features_path):
        match_features_path = os.path.join(season_features_path, match_folder)
        match_raw_path = os.path.join(season_raw_path, match_folder)

        if os.path.isdir(match_features_path) and os.path.isdir(match_raw_path):
            X, y = load_match_data(match_features_path, match_raw_path)
            all_X.append(X)
            all_y.append(y)
            shapes = [arr.shape for arr in X]
            print(f"Temporada {season}, cargado partido: {match_folder}, features shapes: {shapes}")

# ===============================
# GUARDAR DATASET
# ===============================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"X": all_X, "y": all_y}, f)

print(f"Dataset guardado en {OUTPUT_FILE}, partidos cargados: {len(all_X)}")
