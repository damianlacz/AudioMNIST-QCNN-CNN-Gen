import pennylane as qml
import matplotlib.pyplot as plt

N_QUBITS = 8
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(device=dev, interface="torch")
def generator_circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    layers = weights.shape[0]
    for l in range(layers):
        for i in range(N_QUBITS):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

@qml.qnode(device=dev, interface="torch")
def discriminator_circuit(inputs, weights):

    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    layers = weights.shape[0]
    for l in range(layers):
        for i in range(N_QUBITS):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    return qml.expval(qml.PauliZ(0))

def visualize_circuits(*circuits):
    return

    fig, axes = plt.subplots(1, len(circuits), figsize=(5 * len(circuits), 5))
    if len(circuits) == 1:
        axes = [axes]
    for ax, circuit in zip(axes, circuits):
        qml.draw(circuit)(qml.numpy.zeros(N_QUBITS), qml.numpy.zeros((2, N_QUBITS, 3)))
        ax.set_title(circuit.__name__)
    plt.tight_layout()
    plt.show()