# =========================================================================
# plug.py — Détection du plug d'inoculation blanc pour INFEST
# =========================================================================
#
# Problème :
#   Dans INFEST (darcyabjones/INFEST), Leaf.lesion() = (1.1*R > G) et
#   Leaf.leaf() = (R > 1.1*G). Le plug blanc (R ≈ G ≈ B) ne satisfait
#   aucune des deux conditions -> il est exclu comme le fond. Pendant la
#   phase où la lésion est masquée par le plug, lesion_area ≈ 0 alors que
#   la lésion grandit dessous, ce qui déplace la fenêtre d'observation de
#   la sigmoïde selon la taille de la feuille (biais d'identifiabilité
#   du paramètre k).
#
# Solution :
#   Détecter le plug comme une tache quasi-blanche, circulaire, de surface
#   connue, ENTOURÉE de tissu foliaire (pour la distinguer du fond).
#   On ajoute plug_area aux stats ; en aval : lesion_total = lesion + plug.
#
# Usage :
#   from INFEST.plug import find_plug
#   plug_mask = find_plug(img_rgb)  # image d'une feuille (bounding box)
# =========================================================================

import numpy as np


__names__ = ["find_plug", "PlugParams", "plug_mask"]


class PlugParams:
    """Paramètres de détection. À calibrer sur qq images annotées à la main.

    Les seuils par défaut supposent une image RGB uint8 [0-255] issue du
    pipeline INFEST (après correction couleur ou non : la définition de
    "blanc" en HSV est assez robuste aux deux).
    """

    def __init__(
        self,
        sat_max: float = 0.25,       # saturation max d'un pixel "blanc"
        val_min: float = 0.45,       # luminosité min d'un pixel "blanc"
        area_min: int = 200,         # surface min du plug en pixels (à calibrer!)
        area_max: int = 20000,       # surface max du plug en pixels
        circularity_min: float = 0.55,  # 4*pi*A/P^2 ; un disque = 1.0
        ring_width: int = 7,         # largeur de l'anneau testé autour du blob
        leaf_ring_fraction: float = 0.4,  # fraction min de l'anneau = tissu foliaire
        min_object_size: int = 50,
    ):
        self.sat_max = sat_max
        self.val_min = val_min
        self.area_min = area_min
        self.area_max = area_max
        self.circularity_min = circularity_min
        self.ring_width = ring_width
        self.leaf_ring_fraction = leaf_ring_fraction
        self.min_object_size = min_object_size
        return


def _leaf_pixels(img: np.ndarray, f: float = 1.1) -> np.ndarray:
    """Même définition du tissu foliaire que Leaf.leaf() : vert > f*rouge."""
    return (f * img[:, :, 0]) < img[:, :, 1]


def plug_mask(img: np.ndarray, params: PlugParams | None = None) -> np.ndarray:
    """Retourne un masque booléen (H, W) des pixels couverts par le plug.

    img : image RGB d'UNE feuille (le bounding box du layout, cf. Panel.get()).
    """
    from skimage.color import rgb2hsv
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects, binary_dilation, disk
    from scipy import ndimage as ndi

    p = params if params is not None else PlugParams()

    img01 = img.astype(float) / 255.0
    hsv = rgb2hsv(np.clip(img01, 0, 1))

    # --- 1. Candidats "blancs" : faible saturation, forte valeur ---------
    white = (hsv[:, :, 1] <= p.sat_max) & (hsv[:, :, 2] >= p.val_min)

    # --- 2. Nettoyage : petits objets et trous ---------------------------
    white = ndi.binary_fill_holes(white)
    white = remove_small_objects(white, p.min_object_size)

    # --- 3. Filtrage des blobs : taille, circularité, entourage ----------
    leaf = _leaf_pixels(img)
    ring_struct = disk(p.ring_width)

    out = np.zeros(img.shape[:-1], dtype=bool)

    labels = label(white)
    for region in regionprops(labels):
        area = region.area
        if not (p.area_min <= area <= p.area_max):
            continue

        perim = region.perimeter
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)
        if circularity < p.circularity_min:
            continue

        # Le plug est sur la feuille : son anneau externe doit contenir
        # du tissu foliaire. Ça le distingue du fond (anneau vide) et
        # d'un reflet (anneau non foliaire).
        blob = labels == region.label
        ring = binary_dilation(blob, ring_struct) & ~blob
        if ring.sum() == 0:
            continue
        frac_leaf = leaf[ring].mean()
        if frac_leaf < p.leaf_ring_fraction:
            continue

        out |= blob

    return out


# =========================================================================
# Patch d'intégration dans src/INFEST/leaf.py
# =========================================================================
#
# 1) Ajouter dans LeafStats (NamedTuple) les champs :
#
#       plug_area: int = 0
#
#    et dans LeafStats.from_images() :
#
#       plug_area = np.sum(plug)  # + paramètre `plug` en argument
#
#    (penser à mettre "plug_area" dans header() et as_dict()).
#
# 2) Ajouter cette méthode à la classe Leaf :
#
#       def plug(self, params: "PlugParams | None" = None):
#           """Masque du plug d'inoculation (blanc), sinon confondu avec le fond."""
#           from INFEST.plug import plug_mask, PlugParams
#           m = plug_mask(self.img_original, params)
#           if self.exclude_mask is not None:
#               m[self.exclude_mask] = False
#           return m.astype(int)
#
# 3) Dans Leaf.stats(), remplacer :
#
#       return LeafStats.from_images(
#           id=self.id,
#           lesion=lesion_,
#           leaf=leaf_,
#           ...
#
#    par :
#
#       plug_ = self.plug()
#       return LeafStats.from_images(
#           id=self.id,
#           lesion=lesion_,
#           leaf=leaf_,
#           plug=plug_,
#           ...
#
# 4) Dans LeafStats.from_images, la feuille CORRIGÉE vaut :
#
#       leaf_area = np.sum(leaf) + np.sum(lesion) + np.sum(plug)
#
#    (le plug recouvre du tissu vivant).
#    NE PAS ajouter le plug à lesion_area dans le module : gardez la colonne
#    plug_area séparée dans analysis.tsv et faites l'addition en aval.
#    Raison : une fois que la lésion a dépassé le plug, les pixels du plug
#    deviennent nécrotiques et sont comptés deux fois (lesion + plug).
#
# =========================================================================
# Transition plug -> lésion : éviter le double comptage en aval
# =========================================================================
#
#   lesion_total(t) = lesion_area(t) + plug_area(t) * visible_fraction(t)
#
#   Deux stratégies simples :
#
#   a) Décroissance du plug : si plug_area(t) < plug_area(t0) * 0.8, la
#      lésion a commencé à envahir le plug ; on estime la fraction visible
#      comme plug_area(t) / plug_area(t0) (le plug se referme au fur et
#      à mesure qu'il est colonisé/nécrosé).
#
#   b) Conservative (recommandée pour le fit) : compter
#      lesion_total = lesion + plug tant que plug_area > seuil, et
#      lesion_total = lesion dès que le plug disparaît ; au point de
#      transition, les deux définitions coïncident approximativement
#      (la lésion recouvre au moins la zone du plug quand elle est
#      démasquée). Vérifier sur qq feuilles que le saut est faible.
#
#   Dans les deux cas, le CONTRAINT de votre fit logistique/Gompertz
#   devient : lesion_total(t=0) >= surface du plug, ce qui ré-identifie
#   proprement k et t0 indépendamment de la taille de la feuille.
# =========================================================================