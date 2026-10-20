"""Generate orthographic CAD inspection sheets (conversion tools only)."""
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
def sheet(name, meshes, labels=True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = plt.get_cmap("tab20")
    for ax, (a, b, direction) in zip(axes, [(0,1,"XY"),(1,2,"YZ"),(0,2,"XZ")]):
        for i, v, f in meshes:
            color = colors(i % 20)
            ax.add_collection(PolyCollection(v[f][:,:,[a,b]], facecolor=color, edgecolor="none", alpha=.85))
            if labels:
                center = (v.max(0) + v.min(0)) / 2
                ax.text(center[a], center[b], str(i), fontsize=8, color="black", bbox=dict(facecolor="white",alpha=.8,pad=1,edgecolor="none"))
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(direction)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / (name + ".png"), dpi=140)
    plt.close(fig)

if __name__ == "__main__":
    for name in ["SO101_Assembly", "soarm_soft_gripper"]:
        raw = np.load(ROOT / "assets" / (name + "_raw.npz"))
        indices = range(10) if name == "SO101_Assembly" else range(5)
        sheet(name + "_inspection", [(i, raw[f"v{i}"], raw[f"f{i}"]) for i in indices])
    mesh = trimesh.load(ROOT / "assets/cad/IMXmountSO101.stl")
    print("STL bounds", mesh.bounds, "center", mesh.centroid)
    sheet("IMXmountSO101_inspection", [(0, mesh.vertices, mesh.faces)], False)
