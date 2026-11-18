
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = RAIZ / "resultados"

ARCHIVO_CUANTICO = (
    CARPETA_RESULTADOS / "experimentos_ideales.csv"
)

ARCHIVO_CLASICO = (
    CARPETA_RESULTADOS / "baseline_logistico.csv"
)

ARQUITECTURAS = ["TTN", "MERA", "QCNN"]
MODELOS = [
    "TTN",
    "MERA",
    "QCNN",
    "Regresion_logistica"
]

METRICAS = [
    "test_accuracy",
    "test_f1"
]

N_FOLDS = 5


# ============================================================
# CORRECCIÓN DE HOLM
# ============================================================

def correccion_holm(p_values):

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
# CARGA Y VALIDACIÓN
# ============================================================

def cargar_resultados():

    if not ARCHIVO_CUANTICO.exists():
        raise FileNotFoundError(ARCHIVO_CUANTICO)

    if not ARCHIVO_CLASICO.exists():
        raise FileNotFoundError(ARCHIVO_CLASICO)

    cuantico = pd.read_csv(ARCHIVO_CUANTICO)
    clasico = pd.read_csv(ARCHIVO_CLASICO)

    columnas_cuanticas = {
        "fold",
        "architecture",
        "test_accuracy",
        "test_f1"
    }

    columnas_clasicas = {
        "fold",
        "test_accuracy",
        "test_f1"
    }

    if not columnas_cuanticas.issubset(cuantico.columns):
        raise ValueError(
            "El CSV cuántico no contiene todas las columnas necesarias."
        )

    if not columnas_clasicas.issubset(clasico.columns):
        raise ValueError(
            "El CSV clásico no contiene todas las columnas necesarias."
        )

    if len(cuantico) != 75:
        raise ValueError(
            f"Se esperaban 75 ejecuciones cuánticas, "
            f"pero se encontraron {len(cuantico)}."
        )

    if len(clasico) != 5:
        raise ValueError(
            f"Se esperaban 5 folds clásicos, "
            f"pero se encontraron {len(clasico)}."
        )

    if cuantico.duplicated(
        subset=["fold", "seed", "architecture"]
    ).any():
        raise ValueError("Hay ejecuciones cuánticas duplicadas.")

    if clasico["fold"].duplicated().any():
        raise ValueError("Hay folds clásicos duplicados.")

    if set(clasico["fold"]) != set(range(1, N_FOLDS + 1)):
        raise ValueError("Los folds clásicos no coinciden.")

    if set(cuantico["fold"]) != set(range(1, N_FOLDS + 1)):
        raise ValueError("Los folds cuánticos no coinciden.")

    if set(cuantico["architecture"].unique()) != set(ARQUITECTURAS):
        raise ValueError("No se encontraron las tres arquitecturas.")

    if cuantico[METRICAS].isna().any().any():
        raise ValueError("Hay métricas cuánticas ausentes.")

    if clasico[METRICAS].isna().any().any():
        raise ValueError("Hay métricas clásicas ausentes.")

    return cuantico, clasico


# ============================================================
# PROMEDIO DE LAS SEMILLAS CUÁNTICAS POR FOLD
# ============================================================

def construir_tabla_por_fold(cuantico, clasico):

    # Promediar las cinco semillas dentro de cada fold.
    quantum_fold = (
        cuantico
        .groupby(["fold", "architecture"])[METRICAS]
        .mean()
        .reset_index()
    )

    # Convertir arquitecturas a columnas.
    quantum_wide = {}

    for metrica in METRICAS:

        tabla = quantum_fold.pivot(
            index="fold",
            columns="architecture",
            values=metrica
        )

        for arquitectura in ARQUITECTURAS:

            quantum_wide[
                f"{metrica}_{arquitectura}"
            ] = tabla[arquitectura]

    resultado = pd.DataFrame(quantum_wide)

    # Incorporar los resultados clásicos.
    clasico = clasico.set_index("fold")

    for metrica in METRICAS:

        resultado[
            f"{metrica}_Regresion_logistica"
        ] = clasico[metrica]

    resultado = resultado.reset_index()

    return resultado


# ============================================================
# RESUMEN DESCRIPTIVO DE LOS CUATRO MODELOS
# ============================================================

def generar_resumen(tabla):

    filas = []

    for metrica in METRICAS:

        for modelo in MODELOS:

            columna = f"{metrica}_{modelo}"
            valores = tabla[columna].to_numpy(dtype=float)

            n = len(valores)
            media = np.mean(valores)
            desviacion = np.std(valores, ddof=1)

            error = stats.sem(valores)

            margen = (
                stats.t.ppf(0.975, df=n - 1) * error
            )

            filas.append({
                "metric": metrica,
                "model": modelo,
                "n_folds": n,
                "mean": media,
                "standard_deviation": desviacion,
                "IC95_lower": media - margen,
                "IC95_upper": media + margen
            })

    resumen = pd.DataFrame(filas)

    ruta = (
        CARPETA_RESULTADOS /
        "comparacion_descriptiva_cuatro_modelos.csv"
    )

    resumen.to_csv(ruta, index=False)

    print("\nCOMPARACIÓN DESCRIPTIVA")
    print("=======================")

    for metrica in METRICAS:

        print(f"\n{metrica}")

        print(
            resumen[resumen["metric"] == metrica][[
                "model",
                "mean",
                "standard_deviation",
                "IC95_lower",
                "IC95_upper"
            ]].to_string(
                index=False,
                float_format="%.4f"
            )
        )

    print(f"\nGuardado: {ruta}")

    return resumen


# ============================================================
# PRUEBA DE FRIEDMAN Y COMPARACIONES WILCOXON
# ============================================================

def realizar_pruebas(tabla):

    filas = []

    for metrica in METRICAS:

        datos = {
            modelo: tabla[
                f"{metrica}_{modelo}"
            ].to_numpy(dtype=float)
            for modelo in MODELOS
        }

        # Friedman: comparación global de los cuatro modelos.
        estadistico, p_friedman = stats.friedmanchisquare(
            *[datos[modelo] for modelo in MODELOS]
        )

        filas.append({
            "metric": metrica,
            "test": "Friedman",
            "comparison": "Cuatro modelos",
            "statistic": estadistico,
            "p_value": p_friedman,
            "p_adjusted_holm": np.nan
        })

        # Seis comparaciones por pares.
        comparaciones = list(combinations(MODELOS, 2))

        p_values = []
        filas_pares = []

        for modelo_a, modelo_b in comparaciones:

            a = datos[modelo_a]
            b = datos[modelo_b]

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

            filas_pares.append({
                "metric": metrica,
                "test": "Wilcoxon",
                "comparison": f"{modelo_a} vs {modelo_b}",
                "statistic": estadistico_w,
                "p_value": p_value
            })

        p_ajustados = correccion_holm(p_values)

        for fila, p_ajustado in zip(
            filas_pares,
            p_ajustados
        ):

            fila["p_adjusted_holm"] = p_ajustado
            filas.append(fila)

    resultado = pd.DataFrame(filas)

    resultado["significant_alpha_0_05"] = (
        resultado["p_adjusted_holm"]
        .fillna(resultado["p_value"]) < 0.05
    )

    ruta = (
        CARPETA_RESULTADOS /
        "comparacion_estadistica_cuatro_modelos.csv"
    )

    resultado.to_csv(ruta, index=False)

    print("\nPRUEBAS ESTADÍSTICAS")
    print("====================")

    print(
        resultado.to_string(
            index=False,
            float_format="%.5f"
        )
    )

    print(f"\nGuardado: {ruta}")

    return resultado


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def main():

    print("COMPARACIÓN: MODELOS CUÁNTICOS Y CLÁSICO")
    print("========================================")

    cuantico, clasico = cargar_resultados()

    print("\nArchivos validados:")
    print("- 75 ejecuciones cuánticas.")
    print("- 5 evaluaciones de regresión logística.")

    tabla = construir_tabla_por_fold(
        cuantico,
        clasico
    )

    ruta_folds = (
        CARPETA_RESULTADOS /
        "comparacion_por_fold_cuatro_modelos.csv"
    )

    tabla.to_csv(ruta_folds, index=False)

    print("\nRESULTADOS POR FOLD")
    print("===================")

    print(
        tabla.to_string(
            index=False,
            float_format="%.4f"
        )
    )

    generar_resumen(tabla)

    realizar_pruebas(tabla)

    print("\nCOMPARACIÓN COMPLETADA")
    print("======================")
    print("Revisa la carpeta resultados/.")


if __name__ == "__main__":
    main()
