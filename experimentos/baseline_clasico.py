
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path(__file__).resolve().parent.parent

CARPETA_DATOS = RAIZ / "datos"
CARPETA_RESULTADOS = RAIZ / "resultados"

N_FOLDS = 5

# Configuración fija del modelo clásico.
MODELO_CONFIG = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42
}


# ============================================================
# DETECCIÓN DE COLUMNAS
# ============================================================

def detectar_columnas_caracteristicas(df):
    """
    Selecciona las características normalizadas.
    Si no existen, utiliza los ángulos RY y los convierte
    nuevamente a la escala normalizada dividiendo entre pi.
    """

    columnas_normalizadas = [
        columna for columna in df.columns
        if "normalized" in columna.lower()
        or "normalizado" in columna.lower()
    ]

    if len(columnas_normalizadas) == 4:
        return columnas_normalizadas, "normalizadas"

    columnas_angulos = [
        columna for columna in df.columns
        if "ry_angle" in columna.lower()
        or "angulo_ry" in columna.lower()
        or "ángulo_ry" in columna.lower()
    ]

    if len(columnas_angulos) == 4:
        return columnas_angulos, "angulos_ry"

    raise ValueError(
        "\nNo fue posible identificar exactamente cuatro "
        "características normalizadas o cuatro ángulos RY.\n"
        f"Columnas encontradas: {list(df.columns)}"
    )


def detectar_columna_etiqueta(df):
    posibles = [
        "label",
        "target",
        "y",
        "clase",
        "class"
    ]

    for columna in posibles:
        if columna in df.columns:
            return columna

    raise ValueError(
        "No se encontró la columna de etiquetas. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def cargar_particion(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}"
        )

    df = pd.read_csv(ruta)

    columnas, tipo = detectar_columnas_caracteristicas(df)
    columna_etiqueta = detectar_columna_etiqueta(df)

    X = df[columnas].to_numpy(dtype=float)

    # Los ángulos RY codifican los valores normalizados
    # mediante theta = pi * x_normalizado.
    if tipo == "angulos_ry":
        X = X / np.pi

    y = df[columna_etiqueta].to_numpy(dtype=int)

    if not set(np.unique(y)).issubset({-1, 1}):
        raise ValueError(
            f"Se esperaban etiquetas -1 y +1, "
            f"pero se encontraron: {np.unique(y)}"
        )

    if X.shape[1] != 4:
        raise ValueError(
            f"Se esperaban cuatro características, "
            f"pero se encontraron {X.shape[1]}."
        )

    return X, y, columnas, tipo


# ============================================================
# INTERVALO DE CONFIANZA
# ============================================================

def calcular_ic95(valores):
    valores = np.asarray(valores, dtype=float)

    n = len(valores)

    if n < 2:
        return np.nan, np.nan

    media = np.mean(valores)
    error = stats.sem(valores)

    margen = stats.t.ppf(
        0.975,
        df=n - 1
    ) * error

    return media - margen, media + margen


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

def ejecutar_baseline():

    resultados = []

    print("\nBASELINE CLÁSICO: REGRESIÓN LOGÍSTICA")
    print("=====================================")

    for fold in range(1, N_FOLDS + 1):

        print(f"\nProcesando fold {fold}/{N_FOLDS}...")

        carpeta_fold = CARPETA_DATOS / f"fold_{fold}"

        ruta_train = carpeta_fold / "train.csv"
        ruta_test = carpeta_fold / "test.csv"

        X_train, y_train, columnas, tipo = cargar_particion(
            ruta_train
        )

        X_test, y_test, _, _ = cargar_particion(
            ruta_test
        )

        if len(X_train) != 64 or len(X_test) != 20:
            raise ValueError(
                f"Fold {fold}: tamaños inesperados. "
                f"Train={len(X_train)}, Test={len(X_test)}."
            )

        modelo = LogisticRegression(**MODELO_CONFIG)

        inicio = time.perf_counter()

        modelo.fit(X_train, y_train)

        tiempo = time.perf_counter() - inicio

        predicciones = modelo.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predicciones,
            labels=[-1, 1]
        ).ravel()

        accuracy = accuracy_score(
            y_test,
            predicciones
        )

        f1 = f1_score(
            y_test,
            predicciones,
            pos_label=1,
            zero_division=0
        )

        precision = precision_score(
            y_test,
            predicciones,
            pos_label=1,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predicciones,
            pos_label=1,
            zero_division=0
        )

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        resultados.append({
            "fold": fold,
            "model": "LogisticRegression",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(columnas),
            "feature_scale": tipo,
            "C": MODELO_CONFIG["C"],
            "max_iter": MODELO_CONFIG["max_iter"],
            "iterations": int(np.max(modelo.n_iter_)),
            "training_time_seconds": tiempo,
            "test_accuracy": accuracy,
            "test_f1": f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_specificity": specificity,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        })

        print(
            f"Accuracy: {accuracy:.4f} | "
            f"F1: {f1:.4f} | "
            f"Tiempo: {tiempo:.4f} s"
        )

    return pd.DataFrame(resultados)


# ============================================================
# RESUMEN DE RESULTADOS
# ============================================================

def generar_resumen(df):

    metricas = [
        "test_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_specificity",
        "training_time_seconds"
    ]

    filas = []

    for metrica in metricas:

        valores = df[metrica].to_numpy()

        inferior, superior = calcular_ic95(valores)

        filas.append({
            "metric": metrica,
            "n_folds": len(valores),
            "mean": np.mean(valores),
            "standard_deviation": np.std(
                valores,
                ddof=1
            ),
            "IC95_lower": inferior,
            "IC95_upper": superior
        })

    return pd.DataFrame(filas)


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    df = ejecutar_baseline()

    ruta_resultados = (
        CARPETA_RESULTADOS / "baseline_logistico.csv"
    )

    df.to_csv(
        ruta_resultados,
        index=False
    )

    resumen = generar_resumen(df)

    ruta_resumen = (
        CARPETA_RESULTADOS / "resumen_baseline_logistico.csv"
    )

    resumen.to_csv(
        ruta_resumen,
        index=False
    )

    print("\nRESULTADOS POR FOLD")
    print("===================")

    print(
        df[[
            "fold",
            "test_accuracy",
            "test_f1",
            "training_time_seconds"
        ]].to_string(
            index=False,
            float_format="%.4f"
        )
    )

    print("\nRESUMEN DEL BASELINE")
    print("====================")

    print(
        resumen.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    print("\nARCHIVOS GUARDADOS")
    print("==================")
    print(ruta_resultados)
    print(ruta_resumen)

    print("\nBASELINE COMPLETADO")


if __name__ == "__main__":
    main()
