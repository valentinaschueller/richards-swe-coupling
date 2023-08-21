from pathlib import Path

import numpy as np
import proplot as pplt

from coupling.analysis import compute_coupling_behavior, get_sigma

plot_dir = Path("plots")
plot_dir.mkdir(exist_ok=True)

c = 1.0
K = 1.0
dt = 1e-1
L = 1
M = 10

dz = L / M

omegas = np.linspace(0, 2, 100)
S = compute_coupling_behavior(K, c, L, dt, M)[1]
sigmas = get_sigma(omegas, S)
fig_sigma, ax_sigma = pplt.subplots(refwidth="25em", refheight="20em")
ax_sigma.plot(omegas, 0 * omegas, color="gray")
ax_sigma.plot(omegas, sigmas, color="k")
ax_sigma.format(
    xlabel=r"Relaxation parameter $\omega$",
    ylabel=r"$\Sigma(\omega)$",
    title=rf"K = {K}, c = {c}, L = {L}, $\Delta t$={dt}, $\Delta z$={dz}",
)
fig_sigma.suptitle(r"Iteration matrix $\Sigma(\omega)$")
fig_sigma.savefig(plot_dir / "sigma.pdf")

fig_omega, axs_omega = pplt.subplots(ncols=2, refwidth="20em", sharex=False)
fig_omega.suptitle(f"K = {K}, c = {c}, L = {L}")

fig_S, axs_S = pplt.subplots(ncols=2, refwidth="20em", sharex=False)
fig_S.suptitle(f"K = {K}, c = {c}, L = {L}")

dts = 1 / (10 ** np.arange(8, -1, -1))
dz = 1e-1
S_values = np.array([compute_coupling_behavior(K, c, L, dt, M)[1] for dt in dts])
omega_opts = 1 / (1 - S_values)

ax_S = axs_S[0]
ax_S.semilogx(dts, S_values, color="k", marker=".")
ax_S.format(
    title=rf"S for $\Delta t$ decreasing, $\Delta z$ = {dz}",
    xlabel=r"Time step size $\Delta t$",
    ylabel="S",
    xformatter="sci",
)

ax_omega = axs_omega[0]
ax_omega.semilogx(dts, omega_opts, color="k", marker=".")
ax_omega.format(
    title=rf"$\omega_\mathrm{{opt}}$ for $\Delta t$ decreasing, $\Delta z$ = {dz}",
    xlabel=r"Time step size $\Delta t$",
    ylabel=r"$\omega_\mathrm{opt}$",
    xformatter="sci",
    ylim=[0, 1],
)

Ms = 10 ** np.arange(8, -1, -1)
dzs = 1 / Ms
dt = 1e-1
S_values = np.array([compute_coupling_behavior(K, c, L, dt, M)[1] for M in Ms])
omega_opts = 1 / (1 - S_values)

ax_S = axs_S[1]
ax_S.semilogx(dts, S_values, color="k", marker=".")
ax_S.format(
    title=rf"S for $\Delta z$ decreasing, $\Delta t$ = {dt}",
    xlabel=r"Grid size $\Delta z$",
    ylabel="S",
    xformatter="sci",
)
fig_S.savefig(plot_dir / "compute_S.pdf")

ax_omega = axs_omega[1]
ax_omega.semilogx(dts, omega_opts, color="k", marker=".")
ax_omega.format(
    title=rf"$\omega_\mathrm{{opt}}$ for $\Delta z$ decreasing, $\Delta t$ = {dt}",
    xlabel=r"Grid size $\Delta z$",
    ylabel=r"$\omega_\mathrm{opt}$",
    xformatter="sci",
)
fig_omega.savefig(plot_dir / "omega_opt.pdf")
