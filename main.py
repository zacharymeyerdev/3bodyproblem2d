'''
To-do:
add full trails to the bodies
    make a data model for each body, store previous locations in a list
    test around for max trail length
use numba for speed (?)
    njit fastmath
add constants
    G, DT, and SOFTEN (config file?)
choose default masses + initial positions + velocities
maybe let the user choose at the start
    parser + args
acceleration
    velocity verlet + array of shapes
add forces

astropy would be nice
    astropy.units with .value
more?
    collision detection
    better rendering
    tree code? (i gotta look into this)
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

positions = np.array([
    [1.0, 0.5],
    [-0.5, 0.5],
    [0.25, -1.0],
])

fig, ax = plt.subplots()
colors = ["#ff0000", "#0000ff", "#00ff00"]
scatter = ax.scatter(positions[:, 0], positions[:, 1], s=100, c=colors)

plt.figure()
plt.scatter(positions[:, 0], positions[:, 1], s=100, c=colors)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect('equal')

ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('3 bodies animation')

def get_positions(k: int) -> np.ndarray:
    x1 = 1.0 + 0.02 * k # moving right
    y1 = 0.5

    x2 = -0.5
    y2 = 0.5 + 0.4 *np.sin(0.08 * k) # sin wave

    angle = 0.05 * k # circular motion
    x3 = 0.25 + np.cos(angle)
    y3 = -1 + np.sin(angle)

    return np.array([
        [x1, y1],
        [x2, y2],
        [x3, y3]
    ])

def update(frame):
    new_positions = get_positions(frame)
    scatter.set_offsets(new_positions)
    return scatter,

animation = animation.FuncAnimation(
    fig, update, frames=300, interval=40, blit=True
)

plt.show()