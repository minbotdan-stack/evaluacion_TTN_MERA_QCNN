
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ============================================================
# CONFIGURACIÓN
# ============================================================

N_QUBITS = 4
PROBABILIDAD_RUIDO = 0.01

ARQUITECTURAS = ["TTN", "MERA", "QCNN"]
SEMILLAS = [42, 123, 456, 789, 101112]
NUM_FOLDS = 5

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_PROYECTO / "datos"
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"

RUTA_HISTORIALES = RUTA_RESULTADOS / "historiales_ideales.json"
RUTA_IDEALES = RUTA_RESULTADOS / "experimentos_ideales.csv"
RUTA_SALIDA = RUTA_RESULTADOS / "experimentos_ruidosos.csv"

dispositivo = qml.device(
    "default.mixed",
    wires=N_QUBITS
)


# ============================================================
# MODELO DE RUIDO
# ============================================================

def aplicar_ruido_dos_cubits(wires):
    """
    Aplica ruido depolarizante local a ambos cúbits
    después de una puerta de dos cúbits.
    """

    for wire in wires:
        qml.DepolarizingChannel(
            PROBABILIDAD_RUIDO,
            wires=wire
        )


# ============================================================
# BLOQUES CUÁNTICOS CON RUIDO
# ============================================================

def bloque_unitario_ruidoso(wires, theta):

    q0, q1 = wires

    qml.RY(theta[0], wires=q0)
    qml.RZ(theta[1], wires=q0)

    qml.RY(theta[2], wires=q1)
    qml.RZ(theta[3], wires=q1)

    qml.CNOT(wires=[q0, q1])
    aplicar_ruido_dos_cubits([q0, q1])

    qml.RY(theta[4], wires=q0)
    qml.RZ(theta[5], wires=q1)


def bloque_pooling_ruidoso(wires, theta):

    retenido, descartado = wires

    qml.RY(theta[0], wires=retenido)
    qml.RZ(theta[1], wires=retenido)

    qml.RY(theta[2], wires=descartado)
    qml.RZ(theta[3], wires=descartado)

    qml.CNOT(wires=[retenido, descartado])
    aplicar_ruido_dos_cubits([retenido, descartado])

    qml.CRY(theta[4], wires=[retenido, descartado])
    aplicar_ruido_dos_cubits([retenido, descartado])

    qml.CRZ(theta[5], wires=[retenido, descartado])
    aplicar_ruido_dos_cubits([retenido, descartado])


# ============================================================
# CODIFICACIÓN
# ============================================================

def codificar_datos(x):

    for qubit in range(N_QUBITS):
        qml.RY(x[qubit], wires=qubit)


# ============================================================
# CIRCUITO TTN
# ============================================================

@qml.qnode(dispositivo)
def circuito_ttn_ruidoso(x, pesos):

    codificar_datos(x)

    bloque_unitario_ruidoso([0, 1], pesos[0])
    bloque_unitario_ruidoso([2, 3], pesos[1])

    bloque_unitario_ruidoso([0, 2], pesos[2])

    return qml.expval(qml.PauliZ(0))


# ============================================================
# CIRCUITO MERA
# ============================================================

@qml.qnode(dispositivo)
def circuito_mera_ruidoso(x, pesos):

    codificar_datos(x)

    bloque_unitario_ruidoso([1, 2], pesos[0])

    bloque_unitario_ruidoso([0, 1], pesos[1])
    bloque_unitario_ruidoso([2, 3], pesos[2])

    bloque_unitario_ruidoso([0, 2], pesos[3])

    return qml.expval(qml.PauliZ(0))


# ============================================================
# CIRCUITO QCNN
# ============================================================

@qml.qnode(dispositivo)
def circuito_qcnn_ruidoso(x, pesos):

    codificar_datos(x)

    bloque_unitario_ruidoso([0, 1], pesos[0])
    bloque_unitario_ruidoso([2, 3], pesos[0])

    bloque_pooling_ruidoso([0, 1], pesos[1])
    bloque_pooling_ruidoso([2, 3], pesos[1])

    bloque_unitario_ruidoso([0, 2], pesos[2])

    bloque_pooling_ruidoso([0, 2], pesos[3])

    return qml.expval(qml.PauliZ(0))


CIRCUITOS_RUIDOSOS = {
    "TTN": circuito_ttn_ruidoso,
    "MERA": circuito_mera_ruidoso,
    "QCNN": circuito_qcnn_ruidoso
}


# ============================================================
# CARGA DE DATOS
# ============================================================

def cargar_test(fold):

    carpeta = RUTA_DATOS / f"fold_{fold}"
    archivo = carpeta / "test.csv"

    datos = pd.read_csv(archivo)

    columnas = [
        "feature_1_ry_angle",
        "feature_2_ry_angle",
        "feature_3_ry_angle",
        "feature_4_ry_angle"
    ]

    X = datos[columnas].to_numpy(dtype=float)
    y = datos["label"].to_numpy(dtype=int)

    return X, y


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluar_circuito(arquitectura, pesos, X, y):

    circuito = CIRCUITOS_RUIDOSOS[arquitectura]

    salidas = np.array([
        float(circuito(x, pesos))
        for x in X
    ])

    predicciones = np.where(
        salidas >= 0,
        1,
        -1
    )

    accuracy = accuracy_score(y, predicciones)

    f1 = f1_score(
        y,
        predicciones,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    matriz = confusion_matrix(
        y,
        predicciones,
        labels=[-1, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

    mse = float(np.mean((salidas - y) ** 2))

    return {
        "test_accuracy": float(accuracy),
        "test_f1": float(f1),
        "test_mse": mse,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }


# ============================================================
# EJECUCIÓN
# ============================================================

def ejecutar(prueba=False):

    if not RUTA_HISTORIALES.exists():
        raise FileNotFoundError(RUTA_HISTORIALES)

    with open(RUTA_HISTORIALES, "r", encoding="utf-8") as archivo:
        historiales = json.load(archivo)

    ideales = pd.read_csv(RUTA_IDEALES)

    if len(historiales) != 75 or len(ideales) != 75:
        raise ValueError(
            "Se esperaban 75 entrenamientos ideales y 75 historiales."
        )

    resultados = []

    registros = historiales[:3] if prueba else historiales

    print("=" * 65)
    print("EVALUACIÓN DE CIRCUITOS BAJO RUIDO")
    print("=" * 65)
    print("Simulador: default.mixed")
    print("Probabilidad de ruido:", PROBABILIDAD_RUIDO)
    print("Evaluaciones:", len(registros))

    for indice, registro in enumerate(registros, start=1):

        fold = int(registro["fold"])
        semilla = int(registro["seed"])
        arquitectura = registro["architecture"]

        pesos = np.asarray(
            registro["best_weights"],
            dtype=float
        )

        X_test, y_test = cargar_test(fold)

        metricas = evaluar_circuito(
            arquitectura,
            pesos,
            X_test,
            y_test
        )

        resultado_ideal = ideales[
            (ideales["fold"] == fold) &
            (ideales["seed"] == semilla) &
            (ideales["architecture"] == arquitectura)
        ]

        if len(resultado_ideal) != 1:
            raise ValueError(
                f"No se encontró un resultado ideal único: "
                f"fold={fold}, seed={semilla}, modelo={arquitectura}"
            )

        ideal = resultado_ideal.iloc[0]

        fila = {
            "fold": fold,
            "seed": semilla,
            "architecture": arquitectura,
            "noise_probability": PROBABILIDAD_RUIDO,
            **metricas,
            "ideal_accuracy": float(ideal["test_accuracy"]),
            "ideal_f1": float(ideal["test_f1"])
        }

        fila["accuracy_drop"] = (
            fila["ideal_accuracy"] - fila["test_accuracy"]
        )

        fila["f1_drop"] = (
            fila["ideal_f1"] - fila["test_f1"]
        )

        resultados.append(fila)

        print(
            f"{indice}/{len(registros)} | "
            f"Fold {fold} | Semilla {semilla} | {arquitectura} | "
            f"Accuracy: {fila['test_accuracy']:.3f} | "
            f"F1: {fila['test_f1']:.3f}"
        )

    if prueba:
        print("\nPRUEBA CORTA COMPLETADA.")
        print("Se evaluaron tres ejecuciones.")
        print("Todavía no se generó el archivo definitivo.")
        return

    df = pd.DataFrame(resultados)
    df.to_csv(RUTA_SALIDA, index=False)

    print("\n" + "=" * 65)
    print("EXPERIMENTO CON RUIDO FINALIZADO")
    print("=" * 65)
    print("Ejecuciones:", len(df))
    print("Archivo:", RUTA_SALIDA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prueba",
        action="store_true",
        help="Evalúa solamente las primeras tres ejecuciones."
    )

    argumentos = parser.parse_args()

    ejecutar(prueba=argumentos.prueba)
