import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Activación sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    return x * (1 - x)

# Inicializar pesos
def init_pesos(seed=42):
    np.random.seed(seed)
    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# Visualiza la red neuronal como diagrama
def plot_red(W1, b1, W2, b2):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.set_title("🧠 Estructura de la Red Neuronal", fontsize=14)

    # Posiciones de neuronas
    input_pos = [(0.1, 0.8), (0.1, 0.2)]
    hidden_pos = [(0.5, y) for y in np.linspace(0.1, 0.9, 4)]
    output_pos = [(0.9, 0.5)]

    # Dibuja neuronas
    def draw_neurons(positions, label):
        for i, (x, y) in enumerate(positions):
            ax.add_patch(plt.Circle((x, y), 0.03, color="black", fill=False))
            ax.text(x, y, f"{label}{i+1}", fontsize=10, ha="center", va="center")

    draw_neurons(input_pos, "I")
    draw_neurons(hidden_pos, "H")
    draw_neurons(output_pos, "O")

    # Dibuja conexiones con colores y grosor según peso
    def draw_connections(p1, p2, pesos, layer_name):
        for i, (x1, y1) in enumerate(p1):
            for j, (x2, y2) in enumerate(p2):
                peso = pesos[i, j] if layer_name == "W1" else pesos[j, i]
                color = "red" if peso >= 0 else "blue"
                lw = min(5, max(0.5, abs(peso) * 2))
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.8)

    draw_connections(input_pos, hidden_pos, W1, "W1")
    draw_connections(hidden_pos, output_pos, W2.T, "W2")

    return fig

# Visualiza la frontera de decisión
def plot_decision(W1, b1, W2, b2, X, y):
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 300), np.linspace(-0.2, 1.2, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    a1 = sigmoid(grid @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    Z = a2.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.4, colors=["blue", "orange"])
    ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k', cmap=plt.cm.RdYlBu, s=100)
    ax.set_title("📊 Frontera de decisión aprendida")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True)
    return fig

# Entrena la red y devuelve resultados
def entrenar_red(X, y, lr, epochs, W1, b1, W2, b2):
    for _ in range(epochs):
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)

        error = y - a2
        d_a2 = error * sigmoid_deriv(a2)
        d_W2 = a1.T @ d_a2
        d_b2 = np.sum(d_a2, axis=0, keepdims=True)

        d_a1 = d_a2 @ W2.T * sigmoid_deriv(a1)
        d_W1 = X.T @ d_a1
        d_b1 = np.sum(d_a1, axis=0, keepdims=True)

        W2 += lr * d_W2
        b2 += lr * d_b2
        W1 += lr * d_W1
        b1 += lr * d_b1
    return W1, b1, W2, b2

# Interfaz Streamlit
st.set_page_config(page_title="Red Neuronal Visual", layout="centered")
st.title("🔍 Visualizador de Red Neuronal Simple")
st.markdown("Este ejemplo muestra cómo una red aprende la función lógica **OR**.")

# Entradas
epochs = st.slider("Épocas de entrenamiento", 100, 10000, step=100, value=2000)
lr = st.slider("Tasa de aprendizaje", 0.01, 1.0, step=0.01, value=0.1)
entrenar = st.button("🔁 Entrenar red")

# Datos
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [1]])

# Inicialización
W1, b1, W2, b2 = init_pesos()

if entrenar:
    W1, b1, W2, b2 = entrenar_red(X, y, lr, epochs, W1, b1, W2, b2)

    st.subheader("🎯 Predicciones de la red")
    for xi in X:
        a1 = sigmoid(xi @ W1 + b1)
        a2 = sigmoid(a1 @ W2 + b2)
        st.write(f"Entrada: {xi} → Salida: {a2[0][0]:.4f}")

    st.subheader("📌 Visualización de estructura de la red")
    st.pyplot(plot_red(W1, b1, W2, b2))

    st.subheader("📈 Frontera de decisión")
    st.pyplot(plot_decision(W1, b1, W2, b2, X, y))
