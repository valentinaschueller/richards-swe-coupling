import numpy as np
import proplot as pplt


class River:
    def __init__(self) -> None:
        pass

    def step(self, dt: float, h: float, flux: float) -> float:
        return h + dt * flux


t = 0
t_end = 1
N = 20
dt = (t_end - t) / N
h_0 = 1

h = h_0

h_values = [h_0]
t_values = [t]

river = River()

while t < t_end:
    flux = -1 + 2 * np.random.rand()
    h = river.step(dt, h, flux)
    t += dt

    h_values.append(h)
    t_values.append(t)

fig, ax = pplt.subplots()
ax.plot(t_values, h_values)
pplt.show()
