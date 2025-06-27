'''
To-do:
add full trails to the bodies
    make a data model for each body, store previous locations in a list
    test around for max trail length
use numba for speed (?)
    njit fastmath
add constants
    G, DT, and SOFTEN (config file?)
choose default masses + initial POSITIONS + velocities
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
    REARRANGE CODE
    lots and lots of comments
'''
import config
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
SOFTEN = config.SOFTEN
G = config.G
SECONDS_PER_STEP = config.SECONDS_PER_STEP

@dataclass
class Body:
    pos: np.ndarray
    vel: np.ndarray
    mass: float
    trail: list

MASS = np.array([
    5.97e24,  # Earth
    7.35e22,  # Moon
    1.989e30, # Sun
])

POSITIONS = np.array([
    [1.5e11, 0.0],   # Earth
    [1.5e11, 4.0e8], # Moon
    [0.0, 0.0],      # Sun
])

VELOCITIES = np.array([
    [0.0, 29780.0],  # Earth
    [1022.0, 0.0],   # Moon
    [0.0, 0.0],      # Sun
])

bodies = [Body(pos=POSITIONS[i].copy(),
            vel=VELOCITIES[i].copy(),
            mass=MASS[i],
            trail=[])
        for i in range(len(MASS))]

def acceleration(bodies):
    N = len(bodies)
    acc = np.zeros((N, 2))
    for i in range(N):
        for j in range(i+1, N):
            diff = bodies[i].pos - bodies[j].pos
            dist2 = (diff @ diff) + SOFTEN*SOFTEN
            inv_d3 = dist2 ** (-1.5)
            f_vec = -G * bodies[i].mass * bodies[j].mass * inv_d3 * diff
            acc[i] += f_vec / bodies[i].mass
            acc[j] += f_vec / bodies[j].mass
    return acc

def step(bodies, dt=SECONDS_PER_STEP):
    acc = acceleration(bodies)
    for i, body in enumerate(bodies):
        body.vel += acc[i] * dt
        body.pos += body.vel * dt
        body.trail.append(body.pos.copy())
        if len(body.trail) > config.MAX_TRAIL:
            body.trail.pop(0)
    return np.array([b.pos for b in bodies])

fig, ax = plt.subplots()
colors = ["#ff0000", "#0000ff", "#00ff00"]
scatter = ax.scatter(POSITIONS[:, 0], POSITIONS[:, 1], s=100, c=colors)

plt.figure()
plt.scatter(POSITIONS[:, 0], POSITIONS[:, 1], s=100, c=colors)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect('equal')

ax.set_aspect('equal')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('3 bodies animation')

def get_POSITIONS(k: int) -> np.ndarray:
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
    new_POSITIONS = step(bodies)
    scatter.set_offsets(new_POSITIONS)
    return scatter,

animation = animation.FuncAnimation(
    fig, update, frames=300, interval=40, blit=True
)

plt.show()