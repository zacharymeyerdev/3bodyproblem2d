import numpy as np
import matplotlib.pyplot as plt

positions = np.array([
    [1.0, 0.5],
    [-0.5, 0.5],
    [0.25, -1.0],
])

plt.figure()
plt.scatter(positions[:, 0], positions[:, 1], s=100)

for i, (x, y) in enumerate(positions, start=1):
    plt.text(x + 0.05, y + 0.05, f"Body {i}")

plt.title("Positions of Bodies")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()