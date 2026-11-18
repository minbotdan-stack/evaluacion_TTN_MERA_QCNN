
import json
import time
from pathlib import Path

import pandas as pd
import pennylane.numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from modelos import CONFIGURACIONES, contar_parametros
from entrenamiento import entrenar_modelo


# Configuración del experimento
ARQUITECTURAS = ["TTN", "MERA", "QCNN"]
SEMILLAS = [42, 123, 456, 789, 101112]
NUM_FOLDS = 5

RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RUTA_PROYECTO / "datos"
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"

RUTA_RESULTADOS.mkdir(exist_ok=True)


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
    test = pd.read_csv(carpeta / "test.csv")

    X_train = np.array(
        train[columnas].to_numpy(), requires_grad=False
    )
    y_train = np.array(
        train["label"].to_numpy(), requires_grad=False
    )

    X_val = np.array(
        validation[columnas].to_numpy(), requires_grad=False
    )
    y_val = np.array(
        validation["label"].to_numpy(), requires_grad=False
    )

    X_test = np.array(
        test[columnas].to_numpy(), requires_grad=False
    )
    y_test = np.array(
        test["label"].to_numpy(), requires_grad=False
    )

    return X_train, y_train, X_val, y_val, X_test, y_test



def evaluar_modelo(arquitectura, pesos, X_test, y_test):
    import numpy as np_std

    circuito = CONFIGURACIONES[arquitectura]["circuito"]

    salidas = [
        float(circuito(x, pesos))
        for x in X_test
    ]

    predicciones = np_std.asarray(
        [1 if salida >= 0 else -1 for salida in salidas],
        dtype=int
    )

    etiquetas_reales = np_std.asarray(y_test, dtype=int)

    accuracy = accuracy_score(etiquetas_reales, predicciones)

    f1 = f1_score(
        etiquetas_reales,
        predicciones,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    matriz = confusion_matrix(
        etiquetas_reales,
        predicciones,
        labels=[-1, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

    mse = float(np_std.mean(
        (np_std.asarray(salidas) - etiquetas_reales) ** 2
    ))

    return {
        "test_accuracy": float(accuracy),
        "test_f1": float(f1),
        "test_mse": mse,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }



def ejecutar_experimentos():
    resultados = []
    historiales = []

    ruta_csv = RUTA_RESULTADOS / "experimentos_ideales.csv"
    ruta_json = RUTA_RESULTADOS / "historiales_ideales.json"

    total = NUM_FOLDS * len(SEMILLAS) * len(ARQUITECTURAS)
    numero_ejecucion = 0

    print("=" * 70)
    print("EXPERIMENTO COMPLETO: TTN vs MERA vs QCNN")
    print("=" * 70)
    print("Particiones:", NUM_FOLDS)
    print("Semillas:", len(SEMILLAS))
    print("Arquitecturas:", len(ARQUITECTURAS))
    print("Total de entrenamientos:", total)
    print("=" * 70)

    for fold in range(1, NUM_FOLDS + 1):

        print(f"\nCARGANDO PARTICIÓN {fold}/{NUM_FOLDS}")

        (
            X_train, y_train,
            X_val, y_val,
            X_test, y_test
        ) = cargar_particion(fold)

        for semilla in SEMILLAS:

            for arquitectura in ARQUITECTURAS:

                numero_ejecucion += 1

                print("\n" + "-" * 70)
                print(
                    f"EJECUCIÓN {numero_ejecucion}/{total} | "
                    f"Fold {fold} | Semilla {semilla} | "
                    f"{arquitectura}"
                )
                print("-" * 70)

                inicio = time.perf_counter()

                resultado = entrenar_modelo(
                    arquitectura=arquitectura,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    semilla=semilla
                )

                tiempo_entrenamiento = (
                    time.perf_counter() - inicio
                )

                pesos = resultado["weights"]

                metricas = evaluar_modelo(
                    arquitectura,
                    pesos,
                    X_test,
                    y_test
                )

                numero_parametros = contar_parametros(
                    arquitectura
                )

                registro = {
                    "fold": fold,
                    "seed": semilla,
                    "architecture": arquitectura,
                    "train_samples": len(X_train),
                    "validation_samples": len(X_val),
                    "test_samples": len(X_test),
                    "epochs_executed": resultado["epochs_executed"],
                    "best_epoch": resultado["best_epoch"],
                    "best_validation_loss": resultado[
                        "best_validation_loss"
                    ],
                    "training_time_seconds": tiempo_entrenamiento,
                    "trainable_parameters": numero_parametros,
                    **metricas
                }

                resultados.append(registro)

                historiales.append({
                    "fold": fold,
                    "seed": semilla,
                    "architecture": arquitectura,
                    "history": resultado["history"],
                    "best_weights": pesos.tolist()
                })

                # Guardar después de cada entrenamiento
                pd.DataFrame(resultados).to_csv(
                    ruta_csv,
                    index=False
                )

                with open(ruta_json, "w", encoding="utf-8") as archivo:
                    json.dump(
                        historiales,
                        archivo,
                        indent=4,
                        ensure_ascii=False
                    )

                print("Épocas ejecutadas:", registro["epochs_executed"])
                print("Mejor época:", registro["best_epoch"])
                print(
                    "Pérdida de validación:",
                    round(registro["best_validation_loss"], 6)
                )
                print(
                    "Exactitud de prueba:",
                    round(registro["test_accuracy"], 4)
                )
                print(
                    "F1 de prueba:",
                    round(registro["test_f1"], 4)
                )
                print(
                    "Tiempo:",
                    round(tiempo_entrenamiento, 2),
                    "segundos"
                )
                print("Resultados guardados correctamente.")

    print("\n" + "=" * 70)
    print("TODOS LOS ENTRENAMIENTOS HAN FINALIZADO")
    print("=" * 70)

    print("\nResultados:")
    print(ruta_csv)

    print("\nHistoriales y pesos:")
    print(ruta_json)

    print("\nTotal de ejecuciones:", len(resultados))


if __name__ == "__main__":
    ejecutar_experimentos()
