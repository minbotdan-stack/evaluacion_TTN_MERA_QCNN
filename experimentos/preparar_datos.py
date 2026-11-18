
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ============================================================

SEMILLA_GENERAL = 42
NUM_PARTICIONES = 5
PROPORCION_VALIDACION = 0.20

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_PROYECTO / "datos"
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"

RUTA_DATOS.mkdir(parents=True, exist_ok=True)
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CARGAR Y FILTRAR EL CONJUNTO IRIS
# ============================================================

print("\nCargando conjunto de datos Iris...")

iris = load_iris()

X_original = iris.data
y_original = iris.target

# Conservar únicamente:
# 1 = Iris versicolor
# 2 = Iris virginica

mascara = np.isin(y_original, [1, 2])

X = X_original[mascara]
y_original_filtrado = y_original[mascara]

# Etiquetas binarias:
# versicolor = -1
# virginica = +1

y = np.where(y_original_filtrado == 1, -1, 1)

nombres_caracteristicas = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

print(f"Total de muestras: {len(X)}")
print(f"Número de características: {X.shape[1]}")
print(f"Iris versicolor: {np.sum(y == -1)}")
print(f"Iris virginica: {np.sum(y == 1)}")


# ============================================================
# 2. CREAR LAS CINCO PARTICIONES ESTRATIFICADAS
# ============================================================

cv = StratifiedKFold(
    n_splits=NUM_PARTICIONES,
    shuffle=True,
    random_state=SEMILLA_GENERAL
)

resumen = []

for numero_fold, (indices_desarrollo, indices_test) in enumerate(
    cv.split(X, y), start=1
):

    print(f"\n{'=' * 55}")
    print(f"PARTICIÓN {numero_fold}")
    print(f"{'=' * 55}")

    # Separar desarrollo en entrenamiento y validación.
    indices_train, indices_val = train_test_split(
        indices_desarrollo,
        test_size=PROPORCION_VALIDACION,
        random_state=1000 + numero_fold,
        stratify=y[indices_desarrollo]
    )

    X_train = X[indices_train]
    X_val = X[indices_val]
    X_test = X[indices_test]

    y_train = y[indices_train]
    y_val = y[indices_val]
    y_test = y[indices_test]

    # ========================================================
    # 3. NORMALIZACIÓN SIN FUGA DE INFORMACIÓN
    # ========================================================

    scaler = MinMaxScaler(feature_range=(0, 1))

    # El escalador se ajusta EXCLUSIVAMENTE con entrenamiento.
    X_train_norm = scaler.fit_transform(X_train)

    # Validación y prueba solamente se transforman.
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    # ========================================================
    # 4. CODIFICACIÓN PARA LOS CIRCUITOS CUÁNTICOS
    # ========================================================

    # Transformación:
    # x normalizado [0, 1] -> x codificado [0, pi/2]
    #
    # La rotación RY utilizará el ángulo 2*x_codificado.

    X_train_encoded = X_train_norm * (np.pi / 2)
    X_val_encoded = X_val_norm * (np.pi / 2)
    X_test_encoded = X_test_norm * (np.pi / 2)

    X_train_angles = 2 * X_train_encoded
    X_val_angles = 2 * X_val_encoded
    X_test_angles = 2 * X_test_encoded

    # ========================================================
    # 5. GUARDAR LOS ARCHIVOS DE CADA PARTICIÓN
    # ========================================================

    carpeta_fold = RUTA_DATOS / f"fold_{numero_fold}"
    carpeta_fold.mkdir(parents=True, exist_ok=True)

    def guardar_split(
        nombre,
        indices,
        X_original_split,
        X_norm,
        X_encoded,
        X_angles,
        etiquetas
    ):

        columnas = {}

        columnas["sample_id"] = indices

        for j, nombre_feature in enumerate(nombres_caracteristicas):
            columnas[f"{nombre_feature}_original"] = X_original_split[:, j]
            columnas[f"feature_{j+1}_norm"] = X_norm[:, j]
            columnas[f"feature_{j+1}_encoded"] = X_encoded[:, j]
            columnas[f"feature_{j+1}_ry_angle"] = X_angles[:, j]

        columnas["label"] = etiquetas

        columnas["class_name"] = np.where(
            etiquetas == -1,
            "versicolor",
            "virginica"
        )

        df = pd.DataFrame(columnas)

        df.to_csv(
            carpeta_fold / f"{nombre}.csv",
            index=False
        )

        return df

    guardar_split(
        "train",
        indices_train,
        X_train,
        X_train_norm,
        X_train_encoded,
        X_train_angles,
        y_train
    )

    guardar_split(
        "validation",
        indices_val,
        X_val,
        X_val_norm,
        X_val_encoded,
        X_val_angles,
        y_val
    )

    guardar_split(
        "test",
        indices_test,
        X_test,
        X_test_norm,
        X_test_encoded,
        X_test_angles,
        y_test
    )

    # ========================================================
    # 6. GUARDAR INFORMACIÓN DEL ESCALADOR
    # ========================================================

    informacion_scaler = {
        "fold": numero_fold,
        "feature_names": nombres_caracteristicas,
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist()
    }

    with open(
        carpeta_fold / "scaler.json",
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            informacion_scaler,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # 7. COMPROBACIONES
    # ========================================================

    assert len(indices_train) == 64
    assert len(indices_val) == 16
    assert len(indices_test) == 20

    assert len(set(indices_train) & set(indices_val)) == 0
    assert len(set(indices_train) & set(indices_test)) == 0
    assert len(set(indices_val) & set(indices_test)) == 0

    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Validación:    {len(X_val)} muestras")
    print(f"Prueba:        {len(X_test)} muestras")

    print(
        "Distribución entrenamiento:",
        {
            "versicolor": int(np.sum(y_train == -1)),
            "virginica": int(np.sum(y_train == 1))
        }
    )

    print(
        "Distribución validación:",
        {
            "versicolor": int(np.sum(y_val == -1)),
            "virginica": int(np.sum(y_val == 1))
        }
    )

    print(
        "Distribución prueba:",
        {
            "versicolor": int(np.sum(y_test == -1)),
            "virginica": int(np.sum(y_test == 1))
        }
    )

    print(
        "Rango codificado en entrenamiento:",
        float(X_train_encoded.min()),
        "a",
        float(X_train_encoded.max())
    )

    resumen.append({
        "fold": numero_fold,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "train_versicolor": int(np.sum(y_train == -1)),
        "train_virginica": int(np.sum(y_train == 1)),
        "validation_versicolor": int(np.sum(y_val == -1)),
        "validation_virginica": int(np.sum(y_val == 1)),
        "test_versicolor": int(np.sum(y_test == -1)),
        "test_virginica": int(np.sum(y_test == 1))
    })


# ============================================================
# 8. GUARDAR RESUMEN GENERAL
# ============================================================

df_resumen = pd.DataFrame(resumen)

df_resumen.to_csv(
    RUTA_RESULTADOS / "resumen_particiones.csv",
    index=False
)

print("\n" + "=" * 55)
print("PREPARACIÓN DE DATOS FINALIZADA")
print("=" * 55)

print(df_resumen.to_string(index=False))

print("\nArchivos guardados en:")
print(RUTA_DATOS)

print("\nResumen guardado en:")
print(RUTA_RESULTADOS / "resumen_particiones.csv")
