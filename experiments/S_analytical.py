from pathlib import Path

import numpy as np
import proplot as pplt

from coupling.analysis import get_a, get_alpha, get_b, get_S

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)

K = 1.0
c = 1.0
L = 1

dts = 10.0 ** np.arange(-9, 0, 1)
dzs = 10.0 ** np.arange(-6, 0, 1)
S = np.zeros((len(dts), len(dzs)))
for i in range(len(dts)):
    for j in range(len(dzs)):
        dt = dts[i]
        dz = dzs[j]
        M = round(L / dz)
        a = get_a(K, c, dt, dz)
        b = get_b(K, c, dt, dz)
        alpha = get_alpha(a, b, L, M, dz)
        S[i, j] = get_S(a, b, alpha, dz)

fig, ax = pplt.subplots()
fig.suptitle(f"K = {K}, c = {c}, L = {L}")
for i in range(len(dts)):
    ax.plot(dzs, S[i], label=rf"$\Delta t=$ {dts[i]}", cycle="tab20b", marker=".")
ax.format(
    # title=rf"$S$ for $\Delta t$ decreasing, $\Delta z$ = {dz}",
    xlabel=r"Grid size $\Delta z$",
    ylabel="S",
    xformatter="sci",
    xscale="log",
)
fig.legend(ncols=1, frame=False)
fig.savefig(plot_dir / "S_dz_decreasing.pdf")

fig, ax = pplt.subplots()
fig.suptitle(f"K = {K}, c = {c}, L = {L}")

for i in range(len(dzs)):
    ax.semilogx(
        dts, S[:, i], label=rf"$\Delta z=$ {dzs[i]}", cycle="tab20b", marker="."
    )
ax.format(
    # title=rf"$S$ for $\Delta t$ decreasing, $\Delta z$ = {dz}",
    xlabel=r"Time step $\Delta t$",
    ylabel="S",
    xformatter="sci",
    xscale="log",
)
fig.legend(ncols=1, frame=False)
fig.savefig(plot_dir / "S_dt_decreasing.pdf")
