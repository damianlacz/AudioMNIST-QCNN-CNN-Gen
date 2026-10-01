import torch as t
import pennylane as qml
import matplotlib.pyplot as plt

N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def q_recon_circuit(x_feat, recon_feat):
    for i in range(N_QUBITS):
        qml.RY(x_feat[i], wires=i)
    for i in range(N_QUBITS):
        qml.RY(-recon_feat[i], wires=i)
    return qml.probs(wires=range(N_QUBITS))

# 2. Quantum KL Divergence Circuit
@qml.qnode(dev, interface="torch", diff_method="backprop")
def qkl_circuit(mu_q, std_q, mu_p, std_p):
    for i in range(N_QUBITS):
        qml.RY(mu_q[i], wires=i)
        qml.RZ(std_q[i], wires=i)
    for i in range(N_QUBITS):
        qml.RZ(-std_p[i], wires=i)
        qml.RY(-mu_p[i], wires=i)
    return qml.probs(wires=range(N_QUBITS))

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

@t.no_grad()
def visualize_circuits(circuits, style="mpl", *params, **kwargs):
    for name, circuit in circuits.items():
        match style:
            case "mpl":
                import matplotlib.pyplot as plt
                qml.draw_mpl(circuit, **kwargs)(*params)
                plt.title(f"Circuit: {name}")
                plt.tight_layout()
                plt.show()
            case "text":
                text = qml.draw(circuit, **kwargs)(*params)
                print(f"Circuit: {name}")
                print(text)
                print("-" * 50)
            case _:
                raise ValueError("Style should be one of the items [`mpl`, `text`]")
