import numpy as np
import proplot as pplt


class River:
    def __init__(self, h_0: float, t_0: float) -> None:
        self.h = h_0
        self.t = t_0
        self.result = [h_0]
        self.time = [t_0]

    def step(self, dt: float, flux: float) -> float:
        self.h += dt * flux
        self.t += dt

    def end_time_step(self) -> None:
        self.result.append(self.h)
        self.time.append(self.t)


t = 0
t_end = 1
N = 20
dt = (t_end - t) / N
h_0 = 1

river = River(h_0, t)

while river.t < t_end:
    flux = -1 + 2 * np.random.rand()
    river.step(dt, flux)
    river.end_time_step()

_, ax = pplt.subplots()
ax.plot(river.time, river.result)
pplt.show()
