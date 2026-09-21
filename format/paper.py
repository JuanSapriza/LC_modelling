from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Width used while generating the figures. The manuscript will scale these
# down to the IEEE single/double-column width, bringing the 12 pt plot text
# close to the manuscript's ~7--8 pt figure text.
SINGLE_COLUMN_WIDTH_IN = 6.0
DOUBLE_COLUMN_WIDTH_IN = 12.0
BASE_FONT_SIZE_PT = 12.0

BLACK = "#000000"
DARK_GRAY = "#3f3f3f"
MID_GRAY = "#888888"
LIGHT_GRAY = "#cfcfcf"
VERY_LIGHT_GRAY = "#eeeeee"

BLOOD_RED = "#c92f35"
DARK_RED = "#7f1d24"
LIGHT_RED = "#df777b"
PALE_RED = "#f1d1d2"
PALE_YELLOW = "#f5e6b8"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_heatmap",
    ["#ffffff", PALE_YELLOW, LIGHT_RED, BLOOD_RED, DARK_RED],
    N=256,
)


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": BASE_FONT_SIZE_PT,
        "axes.titlesize": BASE_FONT_SIZE_PT,
        "axes.labelsize": BASE_FONT_SIZE_PT,
        "legend.fontsize": BASE_FONT_SIZE_PT - 1,
        "xtick.labelsize": BASE_FONT_SIZE_PT - 1,
        "ytick.labelsize": BASE_FONT_SIZE_PT - 1,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "legend.edgecolor": BLACK,
        "figure.facecolor": "white",
        "axes.facecolor": "none",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure(columns=1, height_in=None, **kwargs):
    width_in = SINGLE_COLUMN_WIDTH_IN if columns == 1 else DOUBLE_COLUMN_WIDTH_IN
    if height_in is None:
        height_in = 0.62 * width_in
    return plt.figure(figsize=(width_in, height_in), **kwargs)


def subplots(nrows=1, ncols=1, columns=1, height_in=None, **kwargs):
    width_in = SINGLE_COLUMN_WIDTH_IN if columns == 1 else DOUBLE_COLUMN_WIDTH_IN
    if height_in is None:
        height_in = 0.62 * width_in
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(width_in, height_in), **kwargs)


def style_legend(legend):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("white")
    frame.set_alpha(1.0)
    frame.set_boxstyle("square", pad=0.15)


def savefig(fig, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "pad_inches": 0.01}
    options.update(kwargs)
    fig.savefig(path, **options)


apply()
