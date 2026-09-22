#In[]:
# Dynamic state-space visualization for the second-order model

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


#In[]:
# Configuration

N = 10
MID = N / 2

DARK_RED = "#6B1014"

# Coordinate convention:
# +V   -> +x
# +V'  -> -y
# +V'' -> +z
#
# Dynamics:
# dV/dt  = V'
# dV'/dt = V''

quad_colors = {
    (+1, +1): (0.93, 0.72, 0.70, 1.0),
    (+1, -1): (0.86, 0.61, 0.60, 1.0),
    (-1, +1): (0.96, 0.81, 0.79, 1.0),
    (-1, -1): (0.78, 0.52, 0.53, 1.0),
}


#In[]:
# Build voxel colors

filled = np.ones((N, N, N), dtype=bool)
facecolors = np.empty((N, N, N, 4), dtype=float)

for ix in range(N):
    for iy in range(N):
        for iz in range(N):

            # +V' points toward decreasing plotted y
            sign_vdot = +1 if iy < MID else -1

            # +V'' points upward
            sign_vddot = +1 if iz >= MID else -1

            facecolors[ix, iy, iz] = quad_colors[(sign_vdot, sign_vddot)]


#In[]:
# Figure

fig = plt.figure(figsize=(8.8, 8.2))
ax = fig.add_subplot(111, projection="3d")

ax.voxels(
    filled,
    facecolors=facecolors,
    edgecolor=(0.30, 0.22, 0.22, 0.42),
    linewidth=0.35,
    shade=False,
)

ax.set_xlim(0, N + 2.8)
ax.set_ylim(-2.8, N)
ax.set_zlim(0, N + 2.8)

ax.set_box_aspect((1, 1, 1))

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.grid(False)

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_alpha(0.0)
    axis.line.set_color((0, 0, 0, 0))

ax.view_init(
    elev=25,
    azim=-53,
)

ax.set_title(
    r"(a) Dynamic state space",
    pad=12,
    fontsize=13,
    weight="bold",
)

fig.canvas.draw()


#In[]:
# Projection helpers
# Drawing the arrows as 2D overlays prevents them from being hidden by voxels.

def project_point(p):
    x2, y2, _ = proj3d.proj_transform(
        p[0],
        p[1],
        p[2],
        ax.get_proj(),
    )

    return x2, y2


def overlay_arrow(
    p0,
    p1,
    color="black",
    lw=2.8,
    ms=18,
    alpha=1.0,
    zorder=1000,
):

    a = project_point(p0)
    b = project_point(p1)

    ax.annotate(
        "",
        xy=b,
        xytext=a,
        xycoords="data",
        textcoords="data",
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            mutation_scale=ms,
            alpha=alpha,
        ),
        zorder=zorder,
        annotation_clip=False,
    )

    return a, b


def overlay_line(
    p0,
    p1,
    color="black",
    lw=2.8,
    alpha=1.0,
    zorder=995,
):

    a = project_point(p0)
    b = project_point(p1)

    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        transform=ax.transData,
        color=color,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
        clip_on=False,
    )

    return a, b


#In[]:
# Main coordinate axes
# The portions inside the cube are faded; the external portions are solid.

center = (
    MID,
    MID,
    MID,
)

# Inside cube

overlay_line(
    center,
    (N, MID, MID),
    color="black",
    lw=3.0,
    alpha=0.20,
)

overlay_line(
    center,
    (MID, 0, MID),
    color="black",
    lw=3.0,
    alpha=0.20,
)

overlay_line(
    center,
    (MID, MID, N),
    color="black",
    lw=3.0,
    alpha=0.20,
)

# Outside cube

_, V_tip = overlay_arrow(
    (N, MID, MID),
    (N + 2.0, MID, MID),
    color="black",
    lw=3.0,
    ms=20,
)

_, Vd_tip = overlay_arrow(
    (MID, 0, MID),
    (MID, -2.0, MID),
    color="black",
    lw=3.0,
    ms=20,
)

_, Vdd_tip = overlay_arrow(
    (MID, MID, N),
    (MID, MID, N + 2.0),
    color="black",
    lw=3.0,
    ms=20,
)

# Labels

ax.text2D(
    V_tip[0] + 0.006,
    V_tip[1],
    r"$V$",
    transform=ax.transData,
    fontsize=15,
    color="black",
    zorder=1200,
    clip_on=False,
)

ax.text2D(
    Vd_tip[0] - 0.004,
    Vd_tip[1] - 0.010,
    r"$\dot V$",
    transform=ax.transData,
    fontsize=15,
    color="black",
    zorder=1200,
    clip_on=False,
)

ax.text2D(
    Vdd_tip[0],
    Vdd_tip[1] + 0.010,
    r"$\ddot V$",
    transform=ax.transData,
    fontsize=15,
    color="black",
    zorder=1200,
    clip_on=False,
    ha="center",
)


#In[]:
# Motion arrows
#
# One arrow is shown on every visible face of the eight sub-cubes obtained
# by splitting the space at:
#
# V = 0
# V' = 0
# V'' = 0
#
# Rule:
#
# V' > 0  -> V increases
# V' < 0  -> V decreases
#
# V'' > 0 -> V' increases
# V'' < 0 -> V' decreases

L = 2.7
eps = 0.16


#In[]:
# Top face
#
# Parallel to V and V'.
#
# We show V motion.
# Direction depends on sign(V').

z = N + eps

# V < 0, V' > 0

overlay_arrow(
    (1.2, 2.4, z),
    (1.2 + L, 2.4, z),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V > 0, V' > 0

overlay_arrow(
    (6.0, 2.4, z),
    (6.0 + L, 2.4, z),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V < 0, V' < 0

overlay_arrow(
    (3.8, 7.6, z),
    (3.8 - L, 7.6, z),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V > 0, V' < 0

overlay_arrow(
    (8.8, 7.6, z),
    (8.8 - L, 7.6, z),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)


#In[]:
# Front face
#
# Parallel to V and V''.
#
# This face corresponds to V' > 0,
# therefore V always increases.

y = -eps

# V < 0, V'' < 0

overlay_arrow(
    (1.1, y, 2.4),
    (1.1 + L, y, 2.4),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V > 0, V'' < 0

overlay_arrow(
    (6.1, y, 2.4),
    (6.1 + L, y, 2.4),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V < 0, V'' > 0

overlay_arrow(
    (1.1, y, 7.6),
    (1.1 + L, y, 7.6),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V > 0, V'' > 0

overlay_arrow(
    (6.1, y, 7.6),
    (6.1 + L, y, 7.6),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)


#In[]:
# Right face
#
# Parallel to V' and V''.
#
# We show V' motion.
# Direction depends on sign(V'').

x = N + eps

# V' > 0, V'' < 0

overlay_arrow(
    (x, 2.0, 2.4),
    (x, 2.0 + L, 2.4),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V' < 0, V'' < 0

overlay_arrow(
    (x, 6.2, 2.4),
    (x, 6.2 + L, 2.4),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V' > 0, V'' > 0

overlay_arrow(
    (x, 4.0, 7.6),
    (x, 4.0 - L, 7.6),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)

# V' < 0, V'' > 0

overlay_arrow(
    (x, 9.0, 7.6),
    (x, 9.0 - L, 7.6),
    color=DARK_RED,
    lw=3.0,
    ms=18,
)


#In[]:
# Final layout

plt.tight_layout()

# Optional:
# plt.savefig("dynamic_state_space.pdf", bbox_inches="tight")
# plt.savefig("dynamic_state_space.png", dpi=300, bbox_inches="tight")

plt.show()