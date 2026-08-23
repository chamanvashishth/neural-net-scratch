import streamlit as st

st.set_page_config(page_title="Neural Network From Scratch", page_icon="🧠", layout="wide")

st.title("Neural Network From Scratch")
st.caption("A NumPy-only implementation of an MNIST classifier — no PyTorch or TensorFlow.")

st.divider()

left, right, third = st.columns(3)
left.metric("Framework", "NumPy")
right.metric("Dataset", "MNIST")
third.metric("Optimizer", "Adam")

st.header("Architecture")
st.code("Input (784) → Dense → BatchNorm → ReLU → Dropout → Dense → BatchNorm → ReLU → Dropout → Dense (10) → Softmax", language="text")

st.header("What is implemented")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    - Dense layers
    - ReLU activation
    - Batch normalization
    - Dropout
    - Softmax
    """)
with c2:
    st.markdown("""
    - Cross-entropy loss
    - Backpropagation
    - Adam optimization
    - MNIST data loading
    - Precision, recall and F1 metrics
    """)

st.header("How the training pipeline works")
st.markdown("**MNIST → normalization → train/validation split → forward pass → loss → backpropagation → Adam update → checkpoint**")

with st.expander("Why this is a code explorer instead of live inference"):
    st.write(
        "The repository currently does not contain the trained checkpoint files or results/history.json. "
        "The original visualize.py script therefore cannot run as a deployed application by itself. "
        "This Streamlit entry point is intentionally separated from the training script and documents the actual implementation without fabricating model predictions."
    )

st.divider()
st.caption("Source code: https://github.com/chamanvashishth/neural-net-scratch")
