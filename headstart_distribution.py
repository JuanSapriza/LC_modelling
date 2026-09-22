#In[]:
# Imports

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#In[]:
# Configuration

N_HS = 5
HS_VALUES = np.linspace(0.0, 1.0, N_HS)

T = np.linspace(0.0, 1.6, 500)

T_EXIT = 0.58
T_WALL = 1.90

GREEN = "#8DBA91"
RED = "#C98282"
DARK_RED = "#6B1014"
BLACK = "#111111"
WALL = "#E8E1DE"


#In[]:
# Helper

def gaussian(t, mu, sigma):
    p = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    return p / np.max(p)


#In[]:
# Figure

fig = plt.figure(figsize=(9.0, 7.4))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

propagated_area = []

for hs in HS_VALUES:
    mu = 0.82 - 0.98 * hs
    sigma = 0.18
    P = gaussian(T, mu, sigma)

    mask_cross = T <= T_EXIT
    mask_prop = T >= T_EXIT

    verts_cross = [(hs, T[0], 0.0)] + [(hs, t, p) for t, p in zip(T[mask_cross], P[mask_cross])] + [(hs, T[mask_cross][-1], 0.0)]
    verts_prop = [(hs, T[mask_prop][0], 0.0)] + [(hs, t, p) for t, p in zip(T[mask_prop], P[mask_prop])] + [(hs, T[-1], 0.0)]

    z = 100 + 100 * hs

    ax.add_collection3d(Poly3DCollection([verts_cross], facecolor=GREEN, edgecolor="none", alpha=0.9, zorder=z))
    ax.add_collection3d(Poly3DCollection([verts_prop], facecolor=RED, edgecolor="none", alpha=0.9, zorder=z))

    ax.plot(np.full_like(T, hs), T, P, color=BLACK, linewidth=2.3, zorder=z + 1)
    ax.plot([hs, hs], [T[0], T[0]], [0.0, P[0]], color=BLACK, linewidth=2.3, zorder=z + 1)
    ax.plot([hs, hs], [T[-1], T[-1]], [0.0, P[-1]], color=BLACK, linewidth=2.3, zorder=z + 1)
    ax.plot([hs, hs], [T[0], T[-1]], [0.0, 0.0], color=BLACK, linewidth=0.8, alpha=0.8, zorder=z + 1)
    ax.plot([hs, hs], [T_EXIT, T_EXIT], [0.0, np.interp(T_EXIT, T, P)], color=DARK_RED, linewidth=1.0, linestyle="--", zorder=z + 2)

    propagated_area.append(np.trapezoid(P[mask_prop], T[mask_prop]))


#In[]:
# Wall with accumulated propagated probability

propagated_area = np.asarray(propagated_area)
bar_height = 0.88 * propagated_area / propagated_area.max()

wall_vertices = [(-0.08, T_WALL, 0.0), (1.10, T_WALL, 0.0), (1.10, T_WALL, 1.02), (-0.08, T_WALL, 1.02)]
ax.add_collection3d(Poly3DCollection([wall_vertices], facecolor=WALL, edgecolor=BLACK, linewidth=1.2, alpha=0.35, zorder=20))

bar_width = 0.14

for hs, height in zip(HS_VALUES, bar_height):
    x0 = hs - bar_width / 2
    x1 = hs + bar_width / 2
    bar_vertices = [(x0, T_WALL - 0.002, 0.0), (x1, T_WALL - 0.002, 0.0), (x1, T_WALL - 0.002, height), (x0, T_WALL - 0.002, height)]
    ax.add_collection3d(Poly3DCollection([bar_vertices], facecolor=RED, edgecolor=BLACK, linewidth=1.6, alpha=0.9, zorder=500 + 100 * hs))

ax.text(0.52, T_WALL + 0.015, 1.08, r"$\int_{t_{\mathrm{exit}}}^{\infty} P(t_{\mathrm{x}}\mid h)\,dt_{\mathrm{x}}$", fontsize=11, color=BLACK, ha="center", va="bottom", zorder=1200)


#In[]:
# Axes

ax.set_xlim(-0.08, 1.18)
ax.set_ylim(0.0, 2.02)
ax.set_zlim(0.0, 1.18)

ax.set_box_aspect((1.0, 1.55, 0.9))

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

ax.grid(False)

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_alpha(0.0)
    axis.line.set_color((0, 0, 0, 0))


#In[]:
# Axis arrows and labels

axis_kw = dict(color=BLACK, linewidth=1.8, arrow_length_ratio=0.08)

ax.quiver(0.0, 0.0, 0.0, 1.10, 0.0, 0.0, zorder=1000, **axis_kw)
ax.quiver(0.0, 0.0, 0.0, 0.0, 1.62, 0.0, zorder=1000, **axis_kw)
ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, 1.10, zorder=1000, **axis_kw)

ax.text(1.15, 0.0, 0.0, r"$h$", fontsize=14, color=BLACK, ha="left", va="center", zorder=1100)
ax.text(0.0, 1.68, 0.0, r"$t_{\mathrm{x}}$", fontsize=14, color=BLACK, ha="center", va="center", zorder=1100)
ax.text(0.0, 0.0, 1.15, r"$P(t_{\mathrm{x}}\mid h)$", fontsize=14, color=BLACK, ha="center", va="bottom", zorder=1100)


#In[]:
# Crossing / propagation boundary

ax.plot([0.0, 1.02], [T_EXIT, T_EXIT], [0.0, 0.0], color=DARK_RED, linewidth=1.2, linestyle="--", zorder=1000)
ax.text(1.04, T_EXIT, 0.0, r"$t_{\mathrm{exit}}$", fontsize=10, color=DARK_RED, ha="left", va="center", zorder=1100)

ax.text(1.08, 0.28, 0.02, "crossing", fontsize=10, color="#3E6E43", ha="center", zorder=1100)
ax.text(1.08, 1.10, 0.02, "propagate", fontsize=10, color=DARK_RED, ha="center", zorder=1100)


#In[]:
# View

ax.view_init(elev=23, azim=-20)

ax.set_title(r"(d) Headstart-conditioned crossing-time distributions", pad=10, fontsize=13, weight="bold")

plt.tight_layout()

# Optional:
# plt.savefig("headstart_transition.pdf", bbox_inches="tight")
# plt.savefig("headstart_transition.png", dpi=300, bbox_inches="tight")

plt.show()