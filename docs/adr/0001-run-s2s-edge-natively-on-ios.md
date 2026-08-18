---
status: accepted
---

# Run S2S Edge natively on iOS

The iOS demo will run S2S Edge entirely on-device through a native MLX Swift implementation rather than calling the Python runtime or a network inference service. This makes offline edge inference the capability being proved, while accepting the cost of a separate native implementation and tighter artifact, memory, and performance constraints.
