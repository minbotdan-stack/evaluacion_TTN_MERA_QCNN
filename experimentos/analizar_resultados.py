
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = RAIZ / "resultados" / "experimentos_ideales.csv"

CARPETA_RESULTADOS = RAIZ / "resultados"
CARPETA_GRAFICAS = RAIZ / "graficas"

ARQUITECTURAS = ["TTN", "MERA", "QCNN"]
METRICAS = ["test_accuracy", "test_f1"]

N_FOLDS = 5
N_SEMILLAS = 5
N_EJECUCIONES_ESPERADAS = N_FOLDS * N_SEMILLAS * len(ARQUITECTURAS)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def intervalo_confianza(valores, confianza=0.95):
    """
    Intervalo de confianza del 95 % para la media,
    utilizando la distribución t de Student.
    """
    valores = np.asarray(valores, dtype=float)
    valores = valores[~np.isnan(valores)]

    n = len(valores)

    if n < 2:
        return np.nan, np.nan

    media = np.mean(valores)
    error = stats.sem(valores)
    margen = stats.t.ppf(
        (1 + confianza) / 2,
        df=n - 1
    ) * error

    return media - margen, media + margen


def resumir_grupo(grupo, columna):
    valores = grupo[columna].dropna().to_numpy()

    inferior, superior = intervalo_confianza(valores)

    return pd.Series({
        "n": len(valores),
        "media": np.mean(valores),
        "desviacion_estandar": np.std(valores, ddof=1),
        "IC95_inferior": inferior,
        "IC95_superior": superior
    })


def correccion_holm(p_values):
    """
    Corrección de Holm para comparaciones múltiples.
    Devuelve los valores p ajustados.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    orden = np.argsort(p_values)
    ajustados = np.empty(n)

    maximo = 0.0

    for rango, indice in enumerate(orden):
        valor = (n - rango) * p_values[indice]
        maximo = max(maximo, valor)
        ajustados[indice] = min(maximo, 1.0)

    return ajustados


# ============================================================
# 1. CARGA Y VALIDACIÓN
# ============================================================

def cargar_datos():
    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ARCHIVO_ENTRADA}"
        )

    df = pd.read_csv(ARCHIVO_ENTRADA)

    columnas_necesarias = [
        "fold",
        "seed",
        "architecture",
        "epochs_executed",
        "best_epoch",
        "best_validation_loss",
        "training_time_seconds",
        "trainable_parameters",
        "test_accuracy",
        "test_f1",
        "test_mse",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive"
    ]

    faltantes = [
        columna for columna in columnas_necesarias
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas en el CSV: {faltantes}"
        )

    if len(df) != N_EJECUCIONES_ESPERADAS:
        raise ValueError(
            f"Se esperaban {N_EJECUCIONES_ESPERADAS} ejecuciones, "
            f"pero se encontraron {len(df)}."
        )

    if df.duplicated(
        subset=["fold", "seed", "architecture"]
    ).any():
        raise ValueError(
            "Hay ejecuciones duplicadas para una combinación "
            "de fold, semilla y arquitectura."
        )

    conteo = df.groupby("architecture").size()

    for arquitectura in ARQUITECTURAS:
        if conteo.get(arquitectura, 0) != 25:
            raise ValueError(
                f"{arquitectura}: se esperaban 25 ejecuciones."
            )

    if set(df["architecture"].unique()) != set(ARQUITECTURAS):
        raise ValueError(
            "Las arquitecturas del CSV no coinciden "
            "con las tres arquitecturas esperadas."
        )

    if df[METRICAS].isna().any().any():
        raise ValueError(
            "Hay valores ausentes en accuracy o F1."
        )

    print("\nVALIDACIÓN DEL CONJUNTO DE RESULTADOS")
    print("------------------------------------")
    print(f"Ejecuciones encontradas: {len(df)}")
    print(f"Folds: {df['fold'].nunique()}")
    print(f"Semillas: {df['seed'].nunique()}")
    print("\nEjecuciones por arquitectura:")
    print(conteo.to_string())
    print("\nValidación completada correctamente.")

    return df


# ============================================================
# 2. ESTADÍSTICAS DESCRIPTIVAS: 25 EJECUCIONES
# ============================================================

def generar_resumen_ejecuciones(df):
    filas = []

    columnas = [
        "test_accuracy",
        "test_f1",
        "test_mse",
        "training_time_seconds",
        "epochs_executed",
        "best_validation_loss"
    ]

    for arquitectura in ARQUITECTURAS:
        grupo = df[df["architecture"] == arquitectura]

        for columna in columnas:
            resumen = resumir_grupo(grupo, columna)

            filas.append({
                "architecture": arquitectura,
                "metric": columna,
                **resumen.to_dict()
            })

    resultado = pd.DataFrame(filas)

    ruta = CARPETA_RESULTADOS / "resumen_estadistico_ejecuciones.csv"
    resultado.to_csv(ruta, index=False)

    print("\nRESUMEN DESCRIPTIVO: 25 EJECUCIONES")
    print("-----------------------------------")

    for metrica in METRICAS:
        print(f"\n{metrica}")

        tabla = resultado[
            resultado["metric"] == metrica
        ][[
            "architecture",
            "media",
            "desviacion_estandar",
            "IC95_inferior",
            "IC95_superior"
        ]]

        print(tabla.to_string(index=False, float_format="%.4f"))

    print(f"\nGuardado: {ruta}")

    return resultado


# ============================================================
# 3. RESUMEN POR FOLD
# ============================================================

def generar_resumen_folds(df):
    resumen = (
        df.groupby(["architecture", "fold"])[METRICAS]
        .mean()
        .reset_index()
    )

    ruta = CARPETA_RESULTADOS / "resumen_por_fold.csv"
    resumen.to_csv(ruta, index=False)

    print("\nMEDIA POR FOLD (PROMEDIO DE LAS 5 SEMILLAS)")
    print("-------------------------------------------")

    for metrica in METRICAS:
        tabla = resumen.pivot(
            index="fold",
            columns="architecture",
            values=metrica
        )

        print(f"\n{metrica}")
        print(tabla.to_string(float_format="%.4f"))

    print(f"\nGuardado: {ruta}")

    return resumen


# ============================================================
# 4. INTERVALOS DE CONFIANZA ENTRE FOLDS
# ============================================================

def generar_resumen_inferencial(resumen_folds):
    filas = []

    for arquitectura in ARQUITECTURAS:
        grupo = resumen_folds[
            resumen_folds["architecture"] == arquitectura
        ]

        for metrica in METRICAS:
            valores = grupo[metrica].to_numpy()
            inferior, superior = intervalo_confianza(valores)

            filas.append({
                "architecture": arquitectura,
                "metric": metrica,
                "n_folds": len(valores),
                "media_folds": np.mean(valores),
                "desviacion_estandar_folds": np.std(
                    valores, ddof=1
                ),
                "IC95_inferior": inferior,
                "IC95_superior": superior
            })

    resultado = pd.DataFrame(filas)

    ruta = CARPETA_RESULTADOS / "resumen_inferencial_por_fold.csv"
    resultado.to_csv(ruta, index=False)

    print("\nRESUMEN POR FOLD E IC DEL 95 %")
    print("--------------------------------")

    print(
        resultado.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    print(f"\nGuardado: {ruta}")

    return resultado


# ============================================================
# 5. FRIEDMAN Y WILCOXON CON HOLM
# ============================================================

def realizar_pruebas_estadisticas(resumen_folds):
    filas = []

    for metrica in METRICAS:
        tabla = resumen_folds.pivot(
            index="fold",
            columns="architecture",
            values=metrica
        )

        valores = [
            tabla[arquitectura].to_numpy()
            for arquitectura in ARQUITECTURAS
        ]

        estadistico, p_friedman = stats.friedmanchisquare(
            *valores
        )

        filas.append({
            "metric": metrica,
            "test": "Friedman",
            "comparison": "TTN vs MERA vs QCNN",
            "statistic": estadistico,
            "p_value": p_friedman,
            "p_adjusted_holm": np.nan
        })

        comparaciones = list(
            combinations(ARQUITECTURAS, 2)
        )

        p_values = []
        resultados_wilcoxon = []

        for arquitectura_a, arquitectura_b in comparaciones:
            a = tabla[arquitectura_a].to_numpy()
            b = tabla[arquitectura_b].to_numpy()

            try:
                resultado = stats.wilcoxon(
                    a,
                    b,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto"
                )
                estadistico_w = resultado.statistic
                p_value = resultado.pvalue

            except ValueError:
                estadistico_w = np.nan
                p_value = 1.0

            p_values.append(p_value)

            resultados_wilcoxon.append({
                "metric": metrica,
                "test": "Wilcoxon",
                "comparison": f"{arquitectura_a} vs {arquitectura_b}",
                "statistic": estadistico_w,
                "p_value": p_value
            })

        p_ajustados = correccion_holm(p_values)

        for fila, p_ajustado in zip(
            resultados_wilcoxon,
            p_ajustados
        ):
            fila["p_adjusted_holm"] = p_ajustado
            filas.append(fila)

    resultado = pd.DataFrame(filas)

    resultado["significant_alpha_0_05"] = (
        resultado["p_adjusted_holm"].fillna(
            resultado["p_value"]
        ) < 0.05
    )

    ruta = CARPETA_RESULTADOS / "pruebas_estadisticas.csv"
    resultado.to_csv(ruta, index=False)

    print("\nPRUEBAS ESTADÍSTICAS")
    print("--------------------")
    print(
        resultado.to_string(
            index=False,
            float_format="%.5f"
        )
    )

    print(f"\nGuardado: {ruta}")

    return resultado


# ============================================================
# 6. MATRICES DE CONFUSIÓN AGREGADAS
# ============================================================

def generar_matrices_confusion(df):
    filas = []

    fig, ejes = plt.subplots(
        1, 3,
        figsize=(12, 4)
    )

    for eje, arquitectura in zip(ejes, ARQUITECTURAS):
        grupo = df[df["architecture"] == arquitectura]

        tn = int(grupo["true_negative"].sum())
        fp = int(grupo["false_positive"].sum())
        fn = int(grupo["false_negative"].sum())
        tp = int(grupo["true_positive"].sum())

        matriz = np.array([
            [tn, fp],
            [fn, tp]
        ])

        total = matriz.sum()

        filas.append({
            "architecture": arquitectura,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "total_predictions": total
        })

        imagen = eje.imshow(matriz)

        eje.set_title(arquitectura)
        eje.set_xlabel("Predicción")
        eje.set_ylabel("Clase real")

        eje.set_xticks([0, 1])
        eje.set_yticks([0, 1])

        eje.set_xticklabels([
            "Versicolor (-1)",
            "Virginica (+1)"
        ])

        eje.set_yticklabels([
            "Versicolor (-1)",
            "Virginica (+1)"
        ])

        for i in range(2):
            for j in range(2):
                eje.text(
                    j, i,
                    str(matriz[i, j]),
                    ha="center",
                    va="center"
                )

    fig.colorbar(imagen, ax=ejes.ravel().tolist())
    fig.suptitle(
        "Matrices de confusión agregadas: clasificación ideal"
    )
    fig.tight_layout()

    ruta_grafica = CARPETA_GRAFICAS / "matrices_confusion_ideales.png"
    fig.savefig(ruta_grafica, dpi=300, bbox_inches="tight")
    plt.close(fig)

    resultado = pd.DataFrame(filas)

    ruta_csv = CARPETA_RESULTADOS / "matrices_confusion_ideales.csv"
    resultado.to_csv(ruta_csv, index=False)

    print("\nMATRICES DE CONFUSIÓN AGREGADAS")
    print("--------------------------------")
    print(resultado.to_string(index=False))

    print(f"\nGuardado: {ruta_csv}")
    print(f"Gráfica: {ruta_grafica}")

    return resultado


# ============================================================
# 7. GRÁFICAS COMPARATIVAS
# ============================================================

def generar_graficas(df, resumen_folds):
    # --------------------------------------------------------
    # Accuracy y F1: distribución de las 25 ejecuciones
    # --------------------------------------------------------

    fig, ejes = plt.subplots(1, 2, figsize=(11, 5))

    for eje, metrica, titulo in zip(
        ejes,
        METRICAS,
        ["Accuracy", "F1-score"]
    ):
        datos = [
            df[df["architecture"] == arquitectura][metrica].to_numpy()
            for arquitectura in ARQUITECTURAS
        ]

        eje.boxplot(datos, tick_labels=ARQUITECTURAS)
        eje.set_title(titulo)
        eje.set_ylabel("Puntuación")
        eje.set_ylim(0, 1.05)
        eje.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Distribución del rendimiento: 25 ejecuciones por arquitectura"
    )
    fig.tight_layout()

    ruta = CARPETA_GRAFICAS / "distribucion_accuracy_f1.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # Rendimiento promedio por fold
    # --------------------------------------------------------

    fig, ejes = plt.subplots(1, 2, figsize=(11, 5))

    for eje, metrica, titulo in zip(
        ejes,
        METRICAS,
        ["Accuracy por fold", "F1-score por fold"]
    ):
        tabla = resumen_folds.pivot(
            index="fold",
            columns="architecture",
            values=metrica
        )

        tabla[ARQUITECTURAS].plot(
            kind="line",
            marker="o",
            ax=eje
        )

        eje.set_title(titulo)
        eje.set_xlabel("Fold")
        eje.set_ylabel("Puntuación")
        eje.set_xticks(range(1, N_FOLDS + 1))
        eje.set_ylim(0, 1.05)
        eje.grid(alpha=0.3)
        eje.legend(title="Arquitectura")

    fig.tight_layout()

    ruta = CARPETA_GRAFICAS / "rendimiento_por_fold.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # Tiempo de entrenamiento
    # --------------------------------------------------------

    fig, eje = plt.subplots(figsize=(7, 5))

    datos = [
        df[df["architecture"] == arquitectura][
            "training_time_seconds"
        ].to_numpy()
        for arquitectura in ARQUITECTURAS
    ]

    eje.boxplot(datos, tick_labels=ARQUITECTURAS)
    eje.set_title("Tiempo de entrenamiento por arquitectura")
    eje.set_ylabel("Tiempo (segundos)")
    eje.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    ruta = CARPETA_GRAFICAS / "tiempo_entrenamiento.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nGRÁFICAS GENERADAS")
    print("------------------")
    print("graficas/distribucion_accuracy_f1.png")
    print("graficas/rendimiento_por_fold.png")
    print("graficas/tiempo_entrenamiento.png")


# ============================================================
# 8. EJECUCIÓN PRINCIPAL
# ============================================================

def main():
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    CARPETA_GRAFICAS.mkdir(exist_ok=True)

    print("ANÁLISIS DE EXPERIMENTOS CUÁNTICOS IDEALES")
    print("==========================================")

    df = cargar_datos()

    generar_resumen_ejecuciones(df)

    resumen_folds = generar_resumen_folds(df)

    generar_resumen_inferencial(resumen_folds)

    realizar_pruebas_estadisticas(resumen_folds)

    generar_matrices_confusion(df)

    generar_graficas(df, resumen_folds)

    print("\nANÁLISIS COMPLETADO")
    print("===================")
    print("Revisa las carpetas resultados/ y graficas/.")


if __name__ == "__main__":
    main()
