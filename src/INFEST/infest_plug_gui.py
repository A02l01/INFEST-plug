#!/usr/bin/env python3
"""
infest_plug_gui.py — Interface graphique de calibration et visualisation
pour la détection du plug d'inoculation (module INFEST.plug).

Usage :
    python infest_plug_gui.py [image.jpg ...]

Fonctionne avec Tkinter + Matplotlib (déjà dans les dépendances INFEST).
Ouvrez une image de feuille (idéalement le bounding box d'une feuille,
cf. infest-check-layout), ajustez les seuils, contrôlez visuellement le
masque du plug (magenta), de la lésion (rouge) et de la feuille (vert),
puis exportez les paramètres retenus dans un fichier YAML réutilisable
par le pipeline (argument PlugParams(**yaml)).

Aucune dépendance supplémentaire requise en dehors de scikit-image,
scipy, matplotlib, numpy (déjà installés avec INFEST).
"""

from __future__ import annotations

import sys
from os.path import abspath

import numpy as np
import yaml

from INFEST.plug import PlugParams, plug_mask

# ---------------------------------------------------------------------------
# Rendu d'un overlay : feuille (vert), lésion (rouge), plug (magenta)
# ---------------------------------------------------------------------------


def leaf_pixels(img: np.ndarray, f: float = 1.1) -> np.ndarray:
    return (f * img[:, :, 0]) < img[:, :, 1]


def lesion_pixels(img: np.ndarray, f: float = 1.1) -> np.ndarray:
    return (f * img[:, :, 0]) > img[:, :, 1]


def render_overlay(img: np.ndarray, params: PlugParams) -> tuple[np.ndarray, dict]:
    """Retourne (image RGB avec overlays semi-transparents, stats)."""
    overlay = img.astype(float).copy()

    leaf = leaf_pixels(img)
    lesion = lesion_pixels(img)
    plug = plug_mask(img, params)

    overlay[leaf] = 0.65 * overlay[leaf] + 0.35 * np.array([40, 200, 40])
    overlay[lesion] = 0.55 * overlay[lesion] + 0.45 * np.array([220, 30, 30])
    overlay[plug] = 0.40 * overlay[plug] + 0.60 * np.array([255, 0, 255])

    stats = {
        "plug_area_px": int(plug.sum()),
        "leaf_area_px": int(leaf.sum()),
        "lesion_area_px": int(lesion.sum()),
    }
    return np.clip(overlay, 0, 255).astype(np.uint8), stats


# ---------------------------------------------------------------------------
# L'application Tkinter
# ---------------------------------------------------------------------------


class PlugGUI:
    SLIDERS = [
        # (attribut, label, min, max, resolution)
        ("sat_max", "Saturation max (blanc)", 0.0, 1.0, 0.01),
        ("val_min", "Valeur min (blanc)", 0.0, 1.0, 0.01),
        ("area_min", "Surface min (px)", 10, 50000, 10),
        ("area_max", "Surface max (px)", 100, 200000, 100),
        ("circularity_min", "Circularité min", 0.0, 1.0, 0.01),
        ("ring_width", "Largeur anneau (px)", 1, 50, 1),
        ("leaf_ring_fraction", "Fraction feuille anneau", 0.0, 1.0, 0.01),
    ]

    def __init__(self, root, paths: list[str]):
        import tkinter as tk
        from tkinter import filedialog, ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk = tk
        self.filedialog = filedialog

        self.root = root
        self.root.title("INFEST — calibration du plug d'inoculation")

        self.paths = list(paths)
        self.ipath = 0
        self.params = PlugParams()

        # --- Figure matplotlib -------------------------------------------------
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.fig.add_subplot()
        self.ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=3, sticky="nsew")

        # --- Panneau de contrôles ----------------------------------------------
        controls = ttk.Frame(root, padding=8)
        controls.grid(row=0, column=1, sticky="ne")

        self.vars: dict[str, tk.DoubleVar] = {}
        defaults = vars(self.params)

        for i, (attr, label, lo, hi, res) in enumerate(self.SLIDERS):
            ttk.Label(controls, text=label).grid(row=i, column=0, sticky="w")
            var = tk.DoubleVar(value=float(defaults[attr]))
            self.vars[attr] = var
            scale = ttk.Scale(
                controls, from_=lo, to=hi, orient="horizontal", length=220,
                command=lambda _v, a=attr: self._on_change(a),
                variable=var,
            )
            scale.grid(row=i, column=1, sticky="ew", padx=4)
            lab = ttk.Label(controls, text=f"{defaults[attr]:g}", width=8)
            lab.grid(row=i, column=2)
            setattr(self, f"_lab_{attr}", lab)

        # --- Boutons ------------------------------------------------------------
        btns = ttk.Frame(root, padding=8)
        btns.grid(row=1, column=1, sticky="ne")
        ttk.Button(btns, text="Ouvrir image(s)...", command=self.open_images).pack(fill="x")
        ttk.Button(btns, text="Image précédente", command=lambda: self.step(-1)).pack(fill="x")
        ttk.Button(btns, text="Image suivante", command=lambda: self.step(+1)).pack(fill="x")
        ttk.Button(btns, text="Sauver params YAML", command=self.save_yaml).pack(fill="x")

        self.status = ttk.Label(root, text="", padding=8, wraplength=260)
        self.status.grid(row=2, column=1, sticky="news")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        if self.paths:
            self.show()

    # -------------------------------------------------------------------------

    def _on_change(self, attr: str):
        setattr(self.params, attr, self.vars[attr].get())
        getattr(self, f"_lab_{attr}").config(text=f"{self.params.__dict__[attr]:g}")
        if self.paths:
            self.show()

    def open_images(self):
        files = self.filedialog.askopenfilenames(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), ("Tout", "*.*")]
        )
        if files:
            self.paths = list(files)
            self.ipath = 0
            self.show()

    def step(self, d: int):
        if not self.paths:
            return
        self.ipath = (self.ipath + d) % len(self.paths)
        self.show()

    def show(self):
        from skimage import io

        path = self.paths[self.ipath]
        img = io.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.dtype != np.uint8:
            img = (np.clip(img.astype(float) / img.max(), 0, 1) * 255).astype(np.uint8)

        overlay, stats = render_overlay(img, self.params)

        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.imshow(overlay)
        self.ax.set_title(f"{path.split('/')[-1]}", fontsize=9)
        self.canvas.draw()

        total_leaf = stats["leaf_area_px"] + stats["lesion_area_px"] + stats["plug_area_px"]
        msg = (
            f"[{self.ipath + 1}/{len(self.paths)}] {abspath(path)}\n"
            f"plug   : {stats['plug_area_px']} px\n"
            f"lésion : {stats['lesion_area_px']} px\n"
            f"feuille+plug (total tissu) : {total_leaf} px"
        )
        self.status.config(text=msg)

    def save_yaml(self):
        out = self.filedialog.asksaveasfilename(
            defaultextension=".yml",
            filetypes=[("YAML", "*.yml *.yaml")],
            initialfile="plug_params.yml",
        )
        if not out:
            return
        with open(out, "w") as h:
            yaml.safe_dump(
                {k: getattr(self.params, k) for k, _, _, _, _ in self.SLIDERS},
                h, sort_keys=False,
            )
        self.status.config(text=f"Params sauvegardés :\n{out}")


def main():
    import tkinter as tk

    paths = sys.argv[1:]
    root = tk.Tk()
    PlugGUI(root, paths)
    root.mainloop()
    return


if __name__ == "__main__":
    main()