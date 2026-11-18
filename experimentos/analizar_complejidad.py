
from collections import Counter
import pennylane as qml
import pennylane.numpy as np

from modelos import CONFIGURACIONES, contar_parametros


def analizar_circuito(arquitectura):
    configuracion = CONFIGURACIONES[arquitectura]
    circuito = configuracion["circuito"]

    # Datos y pesos de prueba. No se entrena ni se modifica el modelo.
    x = np.zeros(4)
    pesos = np.zeros(
        (configuracion["numero_bloques"], 6)
    )

    # Construir el circuito cuántico sin ejecutarlo.
    tape = circuito.construct((x, pesos), {})
    operaciones = tape.operations

    # Contar operaciones por tipo.
    conteo = Counter(op.name for op in operaciones)

    # Contar operaciones según el número de cúbits involucrados.
    puertas_1q = sum(
        len(op.wires) == 1 for op in operaciones
    )

    puertas_2q = sum(
        len(op.wires) == 2 for op in operaciones
    )

    puertas_mas_2q = sum(
        len(op.wires) > 2 for op in operaciones
    )

    # Calcular profundidad mediante planificación por capas.
    profundidad_por_cubit = {}

    for op in operaciones:
        cubits = list(op.wires)

        capa_anterior = max(
            (profundidad_por_cubit.get(c, 0) for c in cubits),
            default=0
        )

        capa_actual = capa_anterior + 1

        for c in cubits:
            profundidad_por_cubit[c] = capa_actual

    profundidad = max(
        profundidad_por_cubit.values(),
        default=0
    )

    return {
        "Arquitectura": arquitectura,
        "Parametros entrenables": contar_parametros(arquitectura),
        "Puertas de 1 cubit": puertas_1q,
        "Puertas de 2 cubits": puertas_2q,
        "Puertas de mas de 2 cubits": puertas_mas_2q,
        "Operaciones totales": len(operaciones),
        "Profundidad": profundidad,
        "Desglose": dict(conteo)
    }


def main():
    print("COMPLEJIDAD DE LOS CIRCUITOS CUANTICOS")
    print("=" * 65)

    resultados = []

    for arquitectura in ["TTN", "MERA", "QCNN"]:
        resultado = analizar_circuito(arquitectura)
        resultados.append(resultado)

        print(f"\n{arquitectura}")
        print("-" * 40)

        for clave, valor in resultado.items():
            if clave != "Desglose":
                print(f"{clave}: {valor}")

        print("Desglose de operaciones:")
        for nombre, cantidad in sorted(
            resultado["Desglose"].items()
        ):
            print(f"  {nombre}: {cantidad}")

    print("\nRESUMEN COMPARATIVO")
    print("=" * 65)

    print(
        f"{'Modelo':<12}"
        f"{'Parámetros':<14}"
        f"{'Puertas 2q':<14}"
        f"{'Operaciones':<14}"
        f"{'Profundidad':<12}"
    )

    for r in resultados:
        print(
            f"{r['Arquitectura']:<12}"
            f"{r['Parametros entrenables']:<14}"
            f"{r['Puertas de 2 cubits']:<14}"
            f"{r['Operaciones totales']:<14}"
            f"{r['Profundidad']:<12}"
        )


if __name__ == "__main__":
    main()
