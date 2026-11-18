
import argparse
from pathlib import Path

import pandas as pd
import pennylane as qml
import pennylane.numpy as np

from modelos import (
    CONFIGURACIONES,
    inicializar_pesos,
    predecir
)


# ============================================================
# CONFIGURACIÓN EXPERIMENTAL
# ============================================================

TASA_APRENDIZAJE = 0.01
EPOCAS_MAXIMAS = 200
PACIENCIA = 20
MEJORA_MINIMA = 1e-4

SEMILLAS = [42, 123, 456, 789, 101112]

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_PROYECTO / "datos"


# ============================================================
# CARGA DE DATOS
# ============================================================

def cargar_particion(numero_fold):

    carpeta = RUTA_DATOS / f"fold_{numero_fold}"

    columnas = [
        "feature_1_ry_angle",
        "feature_2_ry_angle",
        "feature_3_ry_angle",
        "feature_4_ry_angle"
    ]

    train = pd.read_csv(carpeta / "train.csv")
    validation = pd.read_csv(carpeta / "validation.csv")

    X_train = np.array(
        train[columnas].to_numpy(),
        requires_grad=False
    )

    y_train = np.array(
        train["label"].to_numpy(),
        requires_grad=False
    )

    X_val = np.array(
        validation[columnas].to_numpy(),
        requires_grad=False
    )

    y_val = np.array(
        validation["label"].to_numpy(),
        requires_grad=False
    )

    return X_train, y_train, X_val, y_val


# ============================================================
# FUNCIÓN DE PÉRDIDA MSE
# ============================================================

def calcular_mse(arquitectura, pesos, X, y):

    circuito = CONFIGURACIONES[arquitectura]["circuito"]

    predicciones = qml.math.stack([
        circuito(x, pesos)
        for x in X
    ])

    return np.mean((predicciones - y) ** 2)


# ============================================================
# ENTRENAMIENTO DE UNA ARQUITECTURA
# ============================================================

def entrenar_modelo(
    arquitectura,
    X_train,
    y_train,
    X_val,
    y_val,
    semilla,
    max_epocas=EPOCAS_MAXIMAS
):

    pesos = inicializar_pesos(
        arquitectura,
        semilla=semilla
    )

    optimizador = qml.AdamOptimizer(
        stepsize=TASA_APRENDIZAJE
    )

    mejor_loss = float("inf")
    mejores_pesos = None
    mejor_epoca = 0
    epocas_sin_mejora = 0

    historial = []

    for epoca in range(1, max_epocas + 1):

        # Actualización de parámetros con todos los datos
        # de entrenamiento.

        pesos = optimizador.step(
            lambda p: calcular_mse(
                arquitectura,
                p,
                X_train,
                y_train
            ),
            pesos
        )

        # Evaluación después de actualizar los parámetros.

        loss_train = float(
            calcular_mse(
                arquitectura,
                pesos,
                X_train,
                y_train
            )
        )

        loss_val = float(
            calcular_mse(
                arquitectura,
                pesos,
                X_val,
                y_val
            )
        )

        historial.append({
            "epoch": epoca,
            "train_loss": loss_train,
            "validation_loss": loss_val
        })

        # Early stopping: solo se considera mejora si la
        # pérdida disminuye al menos MEJORA_MINIMA.

        if loss_val < mejor_loss - MEJORA_MINIMA:

            mejor_loss = loss_val
            mejores_pesos = np.array(
                pesos,
                requires_grad=True
            )

            mejor_epoca = epoca
            epocas_sin_mejora = 0

        else:

            epocas_sin_mejora += 1

        if epocas_sin_mejora >= PACIENCIA:
            break

    return {
        "architecture": arquitectura,
        "seed": semilla,
        "epochs_executed": epoca,
        "best_epoch": mejor_epoca,
        "best_validation_loss": mejor_loss,
        "weights": mejores_pesos,
        "history": historial
    }


# ============================================================
# PRUEBA CORTA DE ENTRENAMIENTO
# ============================================================

def prueba_entrenamiento():

    print("\n" + "=" * 65)
    print("PRUEBA CORTA DEL SISTEMA DE ENTRENAMIENTO")
    print("=" * 65)

    # Solo fold 1 y semilla 42.
    X_train, y_train, X_val, y_val = cargar_particion(1)

    print("\nDatos cargados:")
    print("Entrenamiento:", len(X_train))
    print("Validación:", len(X_val))

    # Tres épocas únicamente para comprobar el funcionamiento.
    EPOCAS_PRUEBA = 3

    for arquitectura in ["TTN", "MERA", "QCNN"]:

        print("\n" + "-" * 55)
        print("Arquitectura:", arquitectura)
        print("-" * 55)

        resultado = entrenar_modelo(
            arquitectura=arquitectura,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            semilla=42,
            max_epocas=EPOCAS_PRUEBA
        )

        print(
            "Épocas ejecutadas:",
            resultado["epochs_executed"]
        )

        print(
            "Mejor época:",
            resultado["best_epoch"]
        )

        print(
            "Mejor pérdida de validación:",
            resultado["best_validation_loss"]
        )

        print(
            "Pesos finales:",
            resultado["weights"]
        )

        assert resultado["weights"] is not None
        assert np.isfinite(
            resultado["best_validation_loss"]
        )

        print("Estado: ENTRENAMIENTO CORRECTO")

    print("\n" + "=" * 65)
    print("PRUEBA DE ENTRENAMIENTO FINALIZADA")
    print("=" * 65)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prueba",
        action="store_true",
        help="Ejecuta tres épocas de prueba por arquitectura."
    )

    argumentos = parser.parse_args()

    if argumentos.prueba:
        prueba_entrenamiento()

    else:
        print(
            "Por seguridad, este archivo solo ejecuta "
            "la prueba corta con --prueba."
        )
        print(
            "El experimento completo se ejecutará "
            "mediante un programa independiente."
        )
