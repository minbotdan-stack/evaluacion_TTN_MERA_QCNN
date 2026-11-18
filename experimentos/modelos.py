
import pennylane as qml
import pennylane.numpy as np


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

N_QUBITS = 4

dispositivo = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# BLOQUE UNITARIO DE DOS CÚBITS
# ============================================================

def bloque_unitario(wires, theta):
    """
    Bloque parametrizado de dos cúbits.

    Utilizado en las conexiones jerárquicas de TTN y MERA,
    y en las capas convolucionales de QCNN.

    Tiene seis parámetros entrenables.
    """

    q0, q1 = wires

    qml.RY(theta[0], wires=q0)
    qml.RZ(theta[1], wires=q0)

    qml.RY(theta[2], wires=q1)
    qml.RZ(theta[3], wires=q1)

    qml.CNOT(wires=[q0, q1])

    qml.RY(theta[4], wires=q0)
    qml.RZ(theta[5], wires=q1)


# ============================================================
# BLOQUE DE POOLING CUÁNTICO
# ============================================================

def bloque_pooling(wires, theta):
    """
    Operación unitaria de pooling sobre dos cúbits.

    El primer cúbit es el que se conserva.
    El segundo queda inactivo después del pooling.

    La operación utiliza:
    - Rotaciones RY y RZ.
    - CNOT.
    - Rotaciones controladas CRY y CRZ.

    Tiene seis parámetros entrenables.
    """

    retenido, descartado = wires

    qml.RY(theta[0], wires=retenido)
    qml.RZ(theta[1], wires=retenido)

    qml.RY(theta[2], wires=descartado)
    qml.RZ(theta[3], wires=descartado)

    qml.CNOT(wires=[retenido, descartado])

    qml.CRY(theta[4], wires=[retenido, descartado])
    qml.CRZ(theta[5], wires=[retenido, descartado])


# ============================================================
# CODIFICACIÓN DE LOS DATOS
# ============================================================

def codificar_datos(x):
    """
    Codifica las cuatro características mediante RY.

    x contiene los ángulos finales preparados en preparar_datos.py.
    No se vuelve a multiplicar por dos.
    """

    for qubit in range(N_QUBITS):
        qml.RY(x[qubit], wires=qubit)


# ============================================================
# ARQUITECTURA TTN
# ============================================================

@qml.qnode(dispositivo, interface="autograd")
def circuito_ttn(x, pesos):

    codificar_datos(x)

    # Primer nivel:
    # 4 cúbits -> dos ramas.
    bloque_unitario([0, 1], pesos[0])
    bloque_unitario([2, 3], pesos[1])

    # Se conservan los cúbits 0 y 2.
    # Los cúbits 1 y 3 no vuelven a participar.

    # Segundo nivel:
    # Dos ramas -> una salida.
    bloque_unitario([0, 2], pesos[2])

    return qml.expval(qml.PauliZ(0))


# ============================================================
# ARQUITECTURA MERA
# ============================================================

@qml.qnode(dispositivo, interface="autograd")
def circuito_mera(x, pesos):

    codificar_datos(x)

    # Disentangler entre las ramas centrales.
    bloque_unitario([1, 2], pesos[0])

    # Isometrías de coarse-graining.
    bloque_unitario([0, 1], pesos[1])
    bloque_unitario([2, 3], pesos[2])

    # Se conservan los cúbits 0 y 2.
    # Los cúbits 1 y 3 quedan inactivos.

    # Coarse-graining final.
    bloque_unitario([0, 2], pesos[3])

    return qml.expval(qml.PauliZ(0))


# ============================================================
# ARQUITECTURA QCNN
# ============================================================

@qml.qnode(dispositivo, interface="autograd")
def circuito_qcnn(x, pesos):

    codificar_datos(x)

    # --------------------------------------------------------
    # Primera capa convolucional.
    # Los mismos parámetros se comparten entre ambos pares.
    # --------------------------------------------------------

    bloque_unitario([0, 1], pesos[0])
    bloque_unitario([2, 3], pesos[0])

    # --------------------------------------------------------
    # Primera capa de pooling.
    # También comparte parámetros entre los dos pares.
    # Se conservan 0 y 2.
    # --------------------------------------------------------

    bloque_pooling([0, 1], pesos[1])
    bloque_pooling([2, 3], pesos[1])

    # Los cúbits 1 y 3 quedan inactivos.

    # --------------------------------------------------------
    # Segunda capa convolucional.
    # Opera sobre los cúbits conservados.
    # --------------------------------------------------------

    bloque_unitario([0, 2], pesos[2])

    # --------------------------------------------------------
    # Segunda capa de pooling.
    # Se conserva el cúbit 0.
    # --------------------------------------------------------

    bloque_pooling([0, 2], pesos[3])

    return qml.expval(qml.PauliZ(0))


# ============================================================
# CONFIGURACIÓN DE LAS ARQUITECTURAS
# ============================================================

CONFIGURACIONES = {

    "TTN": {
        "circuito": circuito_ttn,
        "numero_bloques": 3,
        "numero_parametros": 18,
        "descripcion": "Red jerárquica en forma de árbol."
    },

    "MERA": {
        "circuito": circuito_mera,
        "numero_bloques": 4,
        "numero_parametros": 24,
        "descripcion": "Red jerárquica con disentangler y coarse-graining."
    },

    "QCNN": {
        "circuito": circuito_qcnn,
        "numero_bloques": 4,
        "numero_parametros": 24,
        "descripcion": "Red convolucional con pooling y parámetros compartidos."
    }
}


# ============================================================
# INICIALIZACIÓN DE PARÁMETROS
# ============================================================

def inicializar_pesos(arquitectura, semilla=42):

    if arquitectura not in CONFIGURACIONES:
        raise ValueError(
            f"Arquitectura desconocida: {arquitectura}"
        )

    numero_bloques = CONFIGURACIONES[
        arquitectura
    ]["numero_bloques"]

    generador = np.random.default_rng(semilla)

    pesos = generador.uniform(
        low=-0.1,
        high=0.1,
        size=(numero_bloques, 6)
    )

    return np.array(pesos, requires_grad=True)


# ============================================================
# PREDICCIÓN
# ============================================================

def predecir(arquitectura, x, pesos):

    if arquitectura not in CONFIGURACIONES:
        raise ValueError(
            f"Arquitectura desconocida: {arquitectura}"
        )

    circuito = CONFIGURACIONES[
        arquitectura
    ]["circuito"]

    return circuito(x, pesos)


# ============================================================
# CONTEO DE PARÁMETROS
# ============================================================

def contar_parametros(arquitectura):

    if arquitectura not in CONFIGURACIONES:
        raise ValueError(
            f"Arquitectura desconocida: {arquitectura}"
        )

    return CONFIGURACIONES[
        arquitectura
    ]["numero_parametros"]


# ============================================================
# PRUEBA DE FUNCIONAMIENTO Y GRADIENTES
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE ARQUITECTURAS CUÁNTICAS")
    print("=" * 60)

    muestra = np.array([
        0.73,
        0.87,
        0.72,
        0.21
    ])

    arquitecturas = ["TTN", "MERA", "QCNN"]

    for nombre in arquitecturas:

        print(f"\nArquitectura: {nombre}")

        pesos = inicializar_pesos(
            nombre,
            semilla=42
        )

        resultado = predecir(
            nombre,
            muestra,
            pesos
        )

        # Verificar que los gradientes pueden calcularse.
        gradiente = qml.grad(
            lambda p: predecir(nombre, muestra, p)
        )(pesos)

        cantidad_parametros = contar_parametros(nombre)

        print(
            "Descripción:",
            CONFIGURACIONES[nombre]["descripcion"]
        )

        print("Cúbits:", N_QUBITS)
        print("Parámetros entrenables:", cantidad_parametros)
        print("Expectativa Z:", float(resultado))
        print("Gradientes finitos:", bool(np.all(np.isfinite(gradiente))))

        assert np.isfinite(float(resultado))
        assert -1.0 <= float(resultado) <= 1.0
        assert np.all(np.isfinite(gradiente))

        print("Estado: CORRECTO")

    print("\n" + "=" * 60)
    print("VERIFICACIÓN FINALIZADA")
    print("=" * 60)
