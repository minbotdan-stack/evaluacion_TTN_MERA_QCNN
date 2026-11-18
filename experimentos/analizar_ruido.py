from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "resultados" / "experimentos_ruidosos.csv"
SALIDA = BASE / "resultados"

df = pd.read_csv(ARCHIVO)

columnas = [
    "fold", "seed", "architecture",
    "test_accuracy", "test_f1",
    "ideal_accuracy", "ideal_f1",
    "accuracy_drop", "f1_drop"
]

faltantes = [c for c in columnas if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas necesarias: {faltantes}")

if len(df) != 75:
    raise ValueError(f"Se esperaban 75 ejecuciones y se encontraron {len(df)}.")

if df.duplicated(["fold", "seed", "architecture"]).any():
    raise ValueError("Hay ejecuciones duplicadas.")

print("=" * 65)
print("ANÁLISIS DEL EXPERIMENTO CON RUIDO")
print("=" * 65)

print("\nEJECUCIONES POR ARQUITECTURA")
print(df.groupby("architecture").size().to_string())

# Resumen de las 25 ejecuciones de cada arquitectura.
metricas = [
    "test_accuracy",
    "test_f1",
    "ideal_accuracy",
    "ideal_f1",
    "accuracy_drop",
    "f1_drop"
]

resumen = df.groupby("architecture")[metricas].agg(["mean", "std"])
resumen.columns = [
    f"{metrica}_{estadistico}"
    for metrica, estadistico in resumen.columns
]
resumen = resumen.reset_index()

print("\nRESULTADOS PROMEDIO Y DESVIACIÓN ESTÁNDAR")
print("(Cada arquitectura tiene 25 ejecuciones)")
print(resumen.round(4).to_string(index=False))

# Promedios por fold: cada fold reúne las cinco semillas.
por_fold = (
    df.groupby(["architecture", "fold"])[
        ["test_accuracy", "test_f1", "accuracy_drop", "f1_drop"]
    ]
    .mean()
    .reset_index()
)

print("\nRESULTADOS PROMEDIO POR FOLD")
print(por_fold.round(4).to_string(index=False))

# Guardar los resultados.
SALIDA.mkdir(exist_ok=True)

archivo_resumen = SALIDA / "resumen_experimento_ruido.csv"
archivo_folds = SALIDA / "resumen_ruido_por_fold.csv"

resumen.to_csv(archivo_resumen, index=False)
por_fold.to_csv(archivo_folds, index=False)

print("\nARCHIVOS GENERADOS")
print(f"- {archivo_resumen}")
print(f"- {archivo_folds}")

print("\nINTERPRETACIÓN DE LAS DIFERENCIAS")
print("accuracy_drop = accuracy ideal - accuracy con ruido")
print("f1_drop       = F1 ideal - F1 con ruido")
print("Un valor positivo indica pérdida de rendimiento.")
print("Un valor cero indica que la métrica no cambió.")
print("Un valor negativo indica una métrica mayor bajo ruido.")
