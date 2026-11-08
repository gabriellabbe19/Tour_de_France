#!/usr/bin/env python3
"""
gpx2tdf.py - Profil altimétrique "style Tour de France" à partir d'un GPX
(RideWithGPS, BRouter-web, Komoot, Strava, Garmin...).

Fonctions :
  - lecture GPX (trk/trkpt, sinon rte/rtept), waypoints utilisés pour nommer les cols
  - nettoyage + lissage de l'altitude (filtre médian puis Savitzky-Golay)
  - détection automatique des cols et côtes (proéminence + rognage du départ)
  - catégorisation 4 / 3 / 2 / 1 / HC (score = longueur(m) x pente moyenne(%))
  - annotations personnalisées : localités, sprints intermédiaires, points libres
    (fichier JSON éditable ou options --marker en ligne de commande)
  - rendu PNG / SVG / PDF : profil, montées colorées, drapeaux avec infos, bandeau
    de pente, bandeau titre, et chaque élément affichable/masquable à volonté
  - modes de remplissage du profil : montées colorées / bandes de pente /
    dégradé d'altitude / uni
  - export CSV et tableau texte (pratique pour coller dans un message de forum)
  - API propre (build_figure) réutilisée par l'application Streamlit (app.py)

Dépendances : numpy, scipy, matplotlib   (pip install numpy scipy matplotlib)

Exemples :
  python gpx2tdf.py etape.gpx --title "Étape 14 - Pau > Luz Ardiden" --out etape14
  python gpx2tdf.py etape.gpx --names "Col d'Aspin;;Tourmalet" --km-to-go
  python gpx2tdf.py etape.gpx --dump-annotations annos.json        # gabarit à compléter
  python gpx2tdf.py etape.gpx --annotations annos.json --marker 86.5:sprint:Lourdes
  python gpx2tdf.py etape.gpx --fill altitude --no-ribbon --dpi 200
"""
import argparse
import csv
import json
import math
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, medfilt, savgol_filter

# --------------------------------------------------------------------------
# Paramètres par défaut (tout est modifiable en ligne de commande)
# --------------------------------------------------------------------------
STEP_M = 10.0  # pas de rééchantillonnage (m)

# (catégorie, score minimal). Score = longueur (m) x pente moyenne (%).
# Barème de type Strava ; à ajuster selon ce que fait votre concours.
CATEGORIES = [("HC", 80000), ("1", 64000), ("2", 32000), ("3", 16000), ("4", 8000)]

CAT_COLORS = {  # du plus facile au plus dur
    "4": "#8FBF6A",
    "3": "#E7C547",
    "2": "#EE9A3A",
    "1": "#DB4B3C",
    "HC": "#7A1530",
}
GRADE_BINS = [3, 6, 9, 12]  # bornes du bandeau de pente (%) ; < 3 % inclut plat et descente
GRADE_COLORS = ["#B8D4A0", "#F1CC47", "#EE8F3A", "#D9453A", "#5B2A86"]

# dégradé vertical pour --fill altitude (du bas vers le haut)
ALT_RAMP = ["#86BB6F", "#E9D16C", "#F2994A", "#E15759", "#B13A63"]

SPRINT_GREEN = "#1B7B34"
YELLOW = "#F7D117"
INK = "#1B1B1B"
CREAM = "#FBF7EA"


# --------------------------------------------------------------------------
# Lecture du GPX
# --------------------------------------------------------------------------
def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _child_text(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c.text
    return None


def parse_gpx(path):
    root = ET.parse(path).getroot()
    pts = [e for e in root.iter() if _local(e.tag) == "trkpt"]
    if not pts:  # export "route" (rte) au lieu de "track"
        pts = [e for e in root.iter() if _local(e.tag) == "rtept"]
    if not pts:
        sys.exit("Aucun point de trace (trkpt/rtept) trouvé dans le GPX.")

    lat, lon, ele = [], [], []
    for p in pts:
        lat.append(float(p.get("lat")))
        lon.append(float(p.get("lon")))
        t = _child_text(p, "ele")
        ele.append(float(t) if t not in (None, "") else np.nan)
    lat, lon, ele = map(np.array, (lat, lon, ele))

    ok = ~np.isnan(ele)
    if ok.sum() < 2:
        sys.exit(
            "Le GPX ne contient pas d'altitude. Réexportez-le depuis RideWithGPS / "
            "BRouter avec l'altitude, ou ajoutez-la via un MNT (SRTM, IGN RGE ALTI)."
        )
    if (~ok).any():
        idx = np.arange(len(ele))
        ele = np.interp(idx, idx[ok], ele[ok])

    wpts = []
    for e in root.iter():
        if _local(e.tag) == "wpt":
            name = (_child_text(e, "name") or "").strip()
            if name:
                wpts.append((float(e.get("lat")), float(e.get("lon")), name))
    return lat, lon, ele, wpts


def haversine(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------
# Pré-traitement
# --------------------------------------------------------------------------
@dataclass
class Profile:
    d: np.ndarray  # distance (m), pas constant
    alt_raw: np.ndarray  # altitude rééchantillonnée, non lissée
    alt: np.ndarray  # altitude lissée
    lat: np.ndarray
    lon: np.ndarray


def build_profile(lat, lon, ele, smooth_m, step=STEP_M):
    # anti-pics sur les points BRUTS (avant rééchantillonnage, sinon un point
    # aberrant isolé devient un triangle de plusieurs dizaines de mètres)
    if len(ele) >= 5:
        ele = medfilt(ele, 5)
        ele[:2], ele[-2:] = ele[2], ele[-3]  # medfilt met des 0 aux bords
    seg = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    d = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 0.01])  # supprime les doublons
    d, lat, lon, ele = d[keep], lat[keep], lon[keep], ele[keep]

    grid = np.arange(0, d[-1], step)
    if grid[-1] < d[-1]:
        grid = np.append(grid, d[-1])
    alt_raw = np.interp(grid, d, ele)
    la = np.interp(grid, d, lat)
    lo = np.interp(grid, d, lon)

    # 1) anti-pics (ponts, tunnels, artefacts MNT)  2) lissage Savitzky-Golay
    alt = medfilt(alt_raw, 5)
    win = int(round(smooth_m / step)) | 1  # impair
    win = max(5, min(win, (len(alt) // 2) * 2 - 1))
    alt = savgol_filter(alt, win, 2, mode="interp")
    return Profile(grid, alt_raw, alt, la, lo)


def elevation_gain(alt, threshold=2.0):
    gain, ref = 0.0, alt[0]
    for a in alt[1:]:
        if a - ref >= threshold:
            gain += a - ref
            ref = a
        elif a < ref:
            ref = a
    return gain


def alt_at(prof, km):
    """Altitude lissée interpolée au kilomètre donné."""
    return float(np.interp(km * 1000.0, prof.d, prof.alt))


# --------------------------------------------------------------------------
# Détection des montées
# --------------------------------------------------------------------------
@dataclass
class Climb:
    start_i: int
    end_i: int
    km_start: float
    km_end: float
    length_m: float
    gain_m: float
    avg_grade: float
    max_grade: float
    alt_summit: float
    score: float
    category: str  # "" si non classée
    name: str = ""


def categorize(score, cats=CATEGORIES):
    for name, thr in cats:
        if score >= thr:
            return name
    return ""


def max_grade_100m(alt, i0, i1, step=STEP_M, win_m=100.0):
    k = max(1, int(round(win_m / step)))
    if i1 - i0 < k:
        return (alt[i1] - alt[i0]) / max(1e-9, (i1 - i0) * step) * 100
    seg = alt[i0 : i1 + 1]
    return float(np.max((seg[k:] - seg[:-k]) / (k * step)) * 100)


def detect_climbs(prof, min_prominence=25.0, min_length=500.0, min_grade=3.0,
                  min_gain=30.0, trim_grade=2.0, trim_window=500.0):
    alt, d = prof.alt, prof.d
    n = len(alt)
    step = d[1] - d[0] if n > 1 else STEP_M

    # Sommets : maxima avec proéminence suffisante (le dernier point compte
    # comme sommet possible : arrivée au sommet).
    low = alt.min() - 1000.0
    padded = np.concatenate([[low], alt, [low]])
    peaks, _ = find_peaks(padded, prominence=min_prominence)
    peaks = peaks - 1
    peaks = peaks[(peaks >= 0) & (peaks < n)]

    climbs = []
    prev = 0
    k = max(1, int(round(trim_window / step)))
    for p in peaks:
        if p <= prev:
            continue
        s = prev + int(np.argmin(alt[prev : p + 1]))
        # rognage : on avance le départ tant que les 500 m suivants sont quasi plats
        # (fenêtre longue = insensible au bruit d'altitude)
        while s < p - k and (alt[s + k] - alt[s]) / (k * step) * 100 < trim_grade:
            s += 1
        prev = p
        length = d[p] - d[s]
        gain = alt[p] - alt[s]
        if length < min_length or gain < min_gain:
            continue
        avg = gain / length * 100
        if avg < min_grade:
            continue
        score = length * avg
        climbs.append(Climb(
            start_i=int(s), end_i=int(p),
            km_start=d[s] / 1000, km_end=d[p] / 1000,
            length_m=float(length), gain_m=float(gain),
            avg_grade=float(avg),
            max_grade=max_grade_100m(alt, s, p, step),
            alt_summit=float(alt[p]), score=float(score),
            category=categorize(score),
        ))
    return climbs


# --------------------------------------------------------------------------
# Noms des cols
# --------------------------------------------------------------------------
def names_from_waypoints(prof, climbs, wpts, max_off_track=250.0, max_along=500.0):
    """Associe à chaque col le waypoint nommé le plus proche du sommet."""
    if not wpts:
        return
    for c in climbs:
        best, best_dist = None, 1e18
        for la, lo, name in wpts:
            dist = haversine(prof.lat, prof.lon, la, lo)
            j = int(np.argmin(dist))
            if dist[j] > max_off_track:
                continue
            along = abs(prof.d[j] - prof.d[c.end_i])
            if along <= max_along and along < best_dist:
                best, best_dist = name, along
        if best and not c.name:
            c.name = best


def names_from_overpass(prof, climbs, radius=400):
    """Optionnel (--overpass) : cherche un col / sommet nommé dans OpenStreetMap.
    Nécessite un accès Internet ; renvoie silencieusement si ça échoue."""
    for c in climbs:
        if c.name:
            continue
        lat, lon = prof.lat[c.end_i], prof.lon[c.end_i]
        q = (f'[out:json][timeout:20];(node["mountain_pass"="yes"](around:{radius},{lat},{lon});'
             f'node["natural"="saddle"](around:{radius},{lat},{lon});'
             f'node["natural"="peak"](around:{radius},{lat},{lon}););out tags center;')
        try:
            req = urllib.request.Request(
                "https://overpass-api.de/api/interpreter",
                data=urllib.parse.urlencode({"data": q}).encode(),
                headers={"User-Agent": "gpx2tdf/1.0"},
            )
            with urllib.request.urlopen(req, timeout=25) as r:
                els = json.load(r).get("elements", [])
            els = [e for e in els if e.get("tags", {}).get("name")]
            if els:
                els.sort(key=lambda e: haversine(lat, lon, e["lat"], e["lon"]))
                c.name = els[0]["tags"]["name"]
        except Exception as exc:  # réseau, quota...
            print(f"[overpass] {exc}", file=sys.stderr)
            return


# --------------------------------------------------------------------------
# Annotations : localités, sprints intermédiaires, points libres
# --------------------------------------------------------------------------
@dataclass
class Marker:
    km: float
    label: str
    kind: str  # "localite" | "sprint" | "point"


@dataclass
class RenderOptions:
    flags: bool = True       # drapeaux des montées
    ribbon: bool = True      # bandeau de pente sous le profil
    grid: bool = True        # grille discrète
    legend: bool = True      # légende (pente + catégories)
    header: bool = True     # bandeau titre jaune
    markers: bool = True    # annotations localités / sprints / points
    fill: str = "climbs"    # climbs | slope | altitude | plain


def _marker_entries(raw, kind):
    """Normalise une liste d'entrées {"km":..., "nom":...} en [Marker]."""
    out = []
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        km = e.get("km")
        label = e.get("nom", e.get("label", "") or "")
        if km is None or not str(label).strip():
            continue
        out.append(Marker(km=float(km), label=str(label).strip(), kind=kind))
    out.sort(key=lambda m: m.km)
    return out


def load_annotations(path):
    """Lit un fichier d'annotations JSON.

    Format (toutes les clés sont optionnelles) :
    {
      "cols":      [{"km": 108.3, "nom": "Col du Grand Vent", "categorie": "HC"}],
      "localites": [{"km": 0.0,   "nom": "Pau"}],
      "sprints":   [{"km": 86.5,  "nom": "Lourdes"}],
      "points":    [{"km": 45.0,  "nom": "Ravitaillement"}]
    }
    "cols" renomme / recatégorise la montée détectée au km du sommet (tolérance 2 km).
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"Fichier d'annotations introuvable : {path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"Annotations JSON invalides : {exc}")
    return {
        "cols": data.get("cols", []) or [],
        "localites": data.get("localites", []) or [],
        "sprints": data.get("sprints", []) or [],
        "points": data.get("points", []) or [],
    }


def save_annotations(path, climbs, markers):
    """Écrit un gabarit d'annotations : cols détectés à renommer + annotations en place."""
    data = {
        "cols": [
            {"km": round(c.km_end, 1), "nom": c.name, "categorie": c.category}
            for c in climbs
        ],
        "localites": [
            {"km": m.km, "nom": m.label} for m in markers if m.kind == "localite"
        ],
        "sprints": [
            {"km": m.km, "nom": m.label} for m in markers if m.kind == "sprint"
        ],
        "points": [
            {"km": m.km, "nom": m.label} for m in markers if m.kind == "point"
        ],
    }
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def apply_col_overrides(climbs, entries, tol_km=2.0):
    """Applique les renommages / recatégorisations de "cols" sur les montées détectées."""
    matched = 0
    for e in entries:
        if not isinstance(e, dict) or "km" not in e:
            continue
        km = float(e["km"])
        best, best_d = None, tol_km
        for c in climbs:
            d = abs(c.km_end - km)
            if d <= best_d:
                best, best_d = c, d
        if best is None:
            print(f"[annotations] aucune montée au km {km} (±{tol_km} km) : entrée ignorée",
                  file=sys.stderr)
            continue
        matched += 1
        if str(e.get("nom", "") or "").strip():
            best.name = str(e["nom"]).strip()
        cat = str(e.get("categorie", "") or "").strip()
        if cat and (cat in CAT_COLORS or cat.upper() in CAT_COLORS):
            best.category = cat.upper() if cat.upper() == "HC" else cat
    return matched


def markers_from_annotations(ann):
    out = []
    out += _marker_entries(ann.get("localites"), "localite")
    out += _marker_entries(ann.get("sprints"), "sprint")
    out += _marker_entries(ann.get("points"), "point")
    out.sort(key=lambda m: (m.km, m.kind))
    return out


def parse_marker_cli(spec):
    """--marker "KM:TYPE:NOM" -> Marker (types : localite, sprint, point)."""
    parts = spec.split(":", 2)
    if len(parts) != 3:
        sys.exit(f"--marker mal formé : « {spec} » (attendu KM:TYPE:NOM)")
    km_s, kind, label = (p.strip() for p in parts)
    aliases = {"localite": "localite", "ville": "localite", "localité": "localite",
               "sprint": "sprint", "point": "point", "ravitaillement": "point"}
    kind = aliases.get(kind.lower())
    if kind is None:
        sys.exit(f"--marker type inconnu : « {spec} » (localite | sprint | point)")
    try:
        km = float(km_s.replace(",", "."))
    except ValueError:
        sys.exit(f"--marker km invalide : « {spec} »")
    return Marker(km=km, label=label, kind=kind)


# --------------------------------------------------------------------------
# Rendu
# --------------------------------------------------------------------------
def fmt(x, nd=0):
    s = f"{x:,.{nd}f}".replace(",", " ")
    return s.replace(".", ",")


def pick_font():
    from matplotlib import font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    for f in ("Roboto Condensed", "Arial Narrow", "Liberation Sans Narrow",
              "Liberation Sans", "Arial", "DejaVu Sans"):
        if f in avail:
            return f
    return "DejaVu Sans"


def build_figure(prof, climbs, title, subtitle, km_to_go=False, show_all=False,
                vexag=1.0, opts=None, markers=None):
    """Construit la figure matplotlib (sans l'écrire sur disque).

    Rendu disponible pour l'application Streamlit ; utiliser render() pour
    écrire directement des fichiers PNG / SVG / PDF.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon as MplPolygon

    opts = opts or RenderOptions()
    markers = markers or []
    locs = [m for m in markers if m.kind == "localite"]
    sprints = [m for m in markers if m.kind == "sprint"]
    points = [m for m in markers if m.kind == "point"]

    plt.rcParams["font.family"] = pick_font()
    x = prof.d / 1000
    alt = prof.alt
    total = x[-1]
    a_min, a_max = float(alt.min()), float(alt.max())
    rng = max(a_max - a_min, 50.0)

    flagged = [c for c in climbs if c.category or show_all]

    fig = plt.figure(figsize=(16, 7.6), facecolor=CREAM)
    ax = fig.add_axes([0.055, 0.225, 0.925, 0.64], facecolor=CREAM)
    axg = fig.add_axes([0.055, 0.18, 0.925, 0.032], facecolor=CREAM)

    # placement des drapeaux et sprints (avant de fixer l'axe y : il faut la
    # place pour les étiquettes)
    items = []
    if opts.flags:
        items += [("col", c) for c in flagged]
    if opts.markers:
        items += [("sprint", m) for m in sprints]

    def item_km(item):
        return item[1].km_end if item[0] == "col" else item[1].km

    items.sort(key=item_km)
    xmax = total * 1.03
    width_km = max(0.15 * total, 7.0)
    occupied = [[], [], []]
    placed = []
    for kind, payload in items:
        km_i = item_km((kind, payload))
        right_side = km_i + width_km <= xmax
        span = ((km_i - 0.4, km_i + width_km) if right_side
                else (km_i - width_km, km_i + 0.4))
        lvl = next((li for li, occ in enumerate(occupied)
                    if all(span[1] < o0 or span[0] > o1 for o0, o1 in occ)), None)
        if lvl is None:
            lvl = min(range(3), key=lambda li: len(occupied[li]))
        occupied[lvl].append(span)
        placed.append((kind, payload, right_side, lvl))
    max_lvl = max([p[3] for p in placed], default=0)
    vx = max(vexag, 0.2)

    def level_y(lvl):
        return a_max + (0.14 + 0.28 * lvl) * rng / vx

    y0 = max(0.0, math.floor((a_min - 0.18 * rng) / 50.0) * 50.0)
    top = a_max + (0.14 + 0.28 * max_lvl + 0.24) * rng / vx
    if locs and opts.markers:
        top += 0.22 * rng / vx  # bandeau pour les noms de localités
    ax.set_xlim(0, xmax)
    ax.set_ylim(y0, top)

    # base : surface sous la courbe + remplissage choisi
    if opts.fill in ("climbs", "plain", "slope"):
        ax.fill_between(x, y0, alt, color="#E4DCC0", lw=0, zorder=2)

    if opts.fill == "climbs":
        for c in climbs:
            col = CAT_COLORS.get(c.category, "#CFC8AE")
            sl = slice(c.start_i, c.end_i + 1)
            ax.fill_between(x[sl], y0, alt[sl], color=col, lw=0, zorder=3)
    elif opts.fill == "slope":
        # bandes de pente sur tout le profil (fenêtre centrée de 200 m)
        stp = prof.d[1] - prof.d[0]
        k = max(1, int(round(100 / stp)))
        n = len(alt)
        g = np.zeros(n)
        g[k:-k] = (alt[2 * k:] - alt[:-2 * k]) / (2 * k * stp) * 100
        if n > 2 * k:
            g[:k], g[-k:] = g[k], g[-k - 1]
        bounds = [-np.inf] + GRADE_BINS + [np.inf]
        for i, col in enumerate(GRADE_COLORS):
            m = (g >= bounds[i]) & (g < bounds[i + 1])
            if m.any():
                ax.fill_between(x, y0, alt, where=m, color=col, lw=0,
                                zorder=3, interpolate=False)
    elif opts.fill == "altitude":
        # dégradé vertical découpé exactement sur le profil
        cmap = LinearSegmentedColormap.from_list("alt", ALT_RAMP)
        grad = np.linspace(0, 1, 256).reshape(256, 1)
        im = ax.imshow(grad, extent=(0, xmax, y0, a_max), origin="lower",
                       aspect="auto", cmap=cmap, zorder=2, interpolation="bicubic")
        verts = np.column_stack([x, alt])
        verts = np.vstack([verts, [[x[-1], y0], [x[0], y0]]])
        clip = MplPolygon(verts, closed=True, transform=ax.transData)
        im.set_clip_path(clip)

    ax.plot(x, alt, color=INK, lw=1.8, solid_capstyle="round", zorder=5)

    # grille discrète
    kstep = 5 if total <= 60 else 10 if total <= 150 else 20
    tick_vals = np.arange(0, total + 0.01, kstep)  # valeurs affichées
    ticks = (total - tick_vals) if km_to_go else tick_vals  # positions sur l'axe
    if opts.grid:
        for t in ticks:
            ax.axvline(t, color="#D8D0B4", lw=0.7, zorder=1)
        ystep = 250 if rng < 1200 else 500
        for yv in np.arange(math.ceil(y0 / ystep) * ystep, top, ystep):
            ax.axhline(yv, color="#D8D0B4", lw=0.7, zorder=1)
    ystep = 250 if rng < 1200 else 500
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(math.ceil(y0 / ystep) * ystep, a_max + 0.2 * rng, ystep))
    ax.tick_params(axis="y", labelsize=10, colors="#555", length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("altitude (m)", fontsize=10, color="#555")
    for s in ax.spines.values():
        s.set_visible(False)

    if opts.ribbon:
        # les graduations en km se lisent sous le bandeau de pente
        axg.set_xticks(ticks)
        axg.set_xticklabels([f"{v:.0f}" for v in tick_vals])
        axg.tick_params(axis="x", length=0, pad=4, labelsize=10, colors="#555")
        axg.set_xlabel("km à l'arrivée" if km_to_go else "km", fontsize=10,
                       color="#555", labelpad=2, loc="right")
    else:
        # sans bandeau : les km se lisent directement sur l'axe du profil
        ax.set_xticklabels([f"{v:.0f}" for v in tick_vals])
        ax.tick_params(axis="x", labelsize=10, colors="#555", pad=4)
        axg.set_visible(False)
    for s in axg.spines.values():
        s.set_visible(False)

    # drapeaux des montées + sprints intermédiaires
    n_col = 0
    for kind, payload, right_side, lvl in placed:
        if kind == "col":
            c = payload
            n_col += 1
            xs, ys = c.km_end, c.alt_summit
            yl = level_y(lvl)
            col = CAT_COLORS.get(c.category, "#CFC8AE")
            ax.plot([xs, xs], [ys, yl], color=INK, lw=1.2, zorder=6)
            ax.scatter([xs], [yl], s=720, color=col, edgecolor=INK, linewidth=1.6,
                       zorder=7, clip_on=False)
            ax.text(xs, yl, c.category or "·", ha="center", va="center",
                    fontsize=10.5 if c.category == "HC" else 12,
                    fontweight="bold", color="white" if c.category in ("1", "HC") else INK,
                    zorder=8)
            name = c.name or ("Côte" if c.category in ("4", "3", "") else "Col") + f" {n_col}"
            info = (f"{fmt(c.alt_summit)} m  ·  km {fmt(c.km_end, 1)}\n"
                    f"{fmt(c.length_m / 1000, 1)} km à {fmt(c.avg_grade, 1)} %")
            dx = 12 if right_side else -12
            ax.annotate(
                f"{name}\n{info}", (xs, yl), xytext=(dx, 0), textcoords="offset points",
                ha="left" if right_side else "right", va="center", fontsize=9.5,
                color=INK, zorder=8, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc="white",
                          ec=INK, lw=1.0),
            )
        else:
            m = payload
            xs = m.km
            ys = alt_at(prof, xs)
            yl = level_y(lvl)
            ax.plot([xs, xs], [ys, yl], color=SPRINT_GREEN, lw=1.2,
                    ls=(0, (4, 2)), zorder=6)
            ax.scatter([xs], [yl], s=300, color=SPRINT_GREEN, edgecolor=INK,
                       linewidth=1.4, zorder=7, clip_on=False)
            ax.text(xs, yl, "S", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=8)
            info = f"Sprint intermédiaire\n{m.label}  ·  km {fmt(xs, 1)}"
            dx = 12 if right_side else -12
            ax.annotate(
                info, (xs, yl), xytext=(dx, 0), textcoords="offset points",
                ha="left" if right_side else "right", va="center", fontsize=9.5,
                color=INK, zorder=8, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.25", fc="white",
                          ec=SPRINT_GREEN, lw=1.0),
            )

    # points libres (ravitaillements, zones clés...)
    if opts.markers:
        for m in points:
            ys = alt_at(prof, m.km)
            ax.scatter([m.km], [ys], s=70, color=INK, edgecolor="white",
                       linewidth=1.2, zorder=7)
            ax.annotate(m.label, (m.km, ys), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=INK, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.2",
                                  fc="white", ec=INK, lw=0.7))

    # localités : nom en haut du profil, filet vertical jusqu'à l'altitude
    if locs and opts.markers:
        y_row1 = top - 0.055 * rng / vx
        y_row2 = top - 0.215 * rng / vx
        last_km, row = -1e9, 0
        for m in locs:
            ys = alt_at(prof, m.km)
            yl = y_row1 if row == 0 else y_row2
            ax.plot([m.km, m.km], [ys, yl], color="#777", lw=1.0, zorder=6)
            if m.km - last_km < max(1.5, total * 0.03):
                row = 1 - row  # localités trop proches : on alterne
                yl = y_row1 if row == 0 else y_row2
            last_km = m.km
            ax.text(m.km, yl, m.label.upper(), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=INK, zorder=8)

    # bandeau de pente (par pas de 100 m)
    if opts.ribbon:
        # une cellule tous les 100 m, pente calculée sur 200 m centrés (moins de bruit)
        stp = prof.d[1] - prof.d[0]
        k = int(round(100 / stp))
        gx = np.arange(k, len(alt) - k, k)
        grades = (alt[gx + k] - alt[gx - k]) / (2 * k * stp) * 100
        cmap = ListedColormap(GRADE_COLORS)
        norm = BoundaryNorm(GRADE_BINS, cmap.N, extend="both")
        edges = np.append(x[gx] - 0.05, x[gx[-1]] + 0.05)
        axg.pcolormesh(edges, [0, 1], grades[None, :], cmap=cmap, norm=norm,
                       shading="flat")
        axg.set_xlim(0, xmax)
        axg.set_yticks([])

    # légende
    if opts.legend:
        lx = 0.055
        xx = lx
        if opts.ribbon:
            fig.text(lx, 0.075, "Pente :", fontsize=9.5, color="#555", va="center")
            xx = lx + 0.045
            labels = ["< 3 %", "3-6 %", "6-9 %", "9-12 %", "> 12 %"]
            for col, lab in zip(GRADE_COLORS, labels):
                fig.patches.append(plt.Rectangle((xx, 0.066), 0.012, 0.018,
                                                 transform=fig.transFigure,
                                                 fc=col, ec="none"))
                fig.text(xx + 0.016, 0.075, lab, fontsize=9.5, color="#555", va="center")
                xx += 0.065
            xx += 0.02
        fig.text(xx, 0.075, "Catégories :", fontsize=9.5, color="#555", va="center")
        xx += 0.095
        for cat in ("4", "3", "2", "1", "HC"):
            fig.add_artist(Line2D([xx + 0.007], [0.075], marker="o", ms=11, ls="",
                                  mfc=CAT_COLORS[cat], mec=INK, mew=1.2,
                                  transform=fig.transFigure))
            fig.text(xx + 0.019, 0.075, cat, fontsize=9.5, color="#555", va="center")
            xx += 0.045
        fig.text(0.98, 0.03, "Score = longueur (m) × pente moyenne (%)  |  "
                 "4 ≥ 8 000, 3 ≥ 16 000, 2 ≥ 32 000, 1 ≥ 64 000, HC ≥ 80 000",
                 fontsize=8.5, color="#888", ha="right", va="center")

    # en-tête
    if opts.header:
        fig.patches.append(plt.Rectangle((0, 0.915), 1, 0.085,
                                          transform=fig.transFigure,
                                          fc=YELLOW, ec="none", zorder=0))
        fig.text(0.02, 0.958, title, fontsize=24, fontweight="bold", color=INK,
                 va="center")
        fig.text(0.98, 0.958, subtitle, fontsize=13, color=INK, ha="right", va="center")

    return fig


def render(prof, climbs, title, subtitle, out_base, formats, dpi, km_to_go,
           show_all, vexag, opts=None, markers=None):
    """Construit la figure puis écrit les fichiers demandés."""
    fig = build_figure(prof, climbs, title, subtitle, km_to_go, show_all, vexag,
                       opts=opts, markers=markers)
    outs = []
    for f in formats:
        p = Path(f"{out_base}.{f}")
        fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor())
        outs.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return outs


# --------------------------------------------------------------------------
# Sorties texte
# --------------------------------------------------------------------------
def print_table(climbs):
    print(f"{'#':>2} {'Nom':<28} {'Cat':>3} {'km départ':>9} {'km sommet':>9} "
          f"{'Long.':>7} {'Pente':>6} {'Max':>6} {'Alt.':>6}")
    for i, c in enumerate(climbs, 1):
        print(f"{i:>2} {(c.name or '-'):<28} {(c.category or '-'):>3} "
              f"{c.km_start:>9.1f} {c.km_end:>9.1f} {c.length_m / 1000:>6.1f}k "
              f"{c.avg_grade:>5.1f}% {c.max_grade:>5.1f}% {c.alt_summit:>5.0f}m")


def write_csv(path, climbs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["nom", "categorie", "km_depart", "km_sommet", "longueur_km",
                    "pente_moy_pct", "pente_max_100m_pct", "alt_sommet_m", "denivele_m", "score"])
        for c in climbs:
            w.writerow([c.name, c.category, f"{c.km_start:.1f}", f"{c.km_end:.1f}",
                        f"{c.length_m / 1000:.2f}", f"{c.avg_grade:.1f}", f"{c.max_grade:.1f}",
                        f"{c.alt_summit:.0f}", f"{c.gain_m:.0f}", f"{c.score:.0f}"])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Profil altimétrique style Tour de France")
    ap.add_argument("gpx")
    ap.add_argument("--title", help="titre (défaut : nom du fichier)")
    ap.add_argument("--subtitle", help="sous-titre (défaut : distance | D+ | alt. max)")
    ap.add_argument("--out", help="préfixe de sortie (défaut : nom du GPX)")
    ap.add_argument("--formats", default="png", help="png,svg,pdf (défaut : png)")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--smooth", type=float, default=200, help="fenêtre de lissage en m (200)")
    ap.add_argument("--min-prominence", type=float, default=25, help="dénivelé mini entre cols (m)")
    ap.add_argument("--min-length", type=float, default=500, help="longueur mini d'une côte (m)")
    ap.add_argument("--min-grade", type=float, default=3.0, help="pente moyenne mini (%%)")
    ap.add_argument("--min-gain", type=float, default=30, help="dénivelé mini d'une côte (m)")
    ap.add_argument("--names", help="noms des montées classées dans l'ordre, séparés par ';' "
                                    "(laisser vide pour sauter : 'Col A;;Col C')")
    ap.add_argument("--overpass", action="store_true", help="chercher les noms dans OpenStreetMap")
    ap.add_argument("--km-to-go", action="store_true", help="axe en km restants (style TDF)")
    ap.add_argument("--all", action="store_true", help="afficher aussi les montées non classées")
    ap.add_argument("--vexag", type=float, default=1.0,
                    help="> 1 : profil plus plat dans l'image, < 1 : plus dramatique (défaut 1)")
    # --- annotations ---
    ap.add_argument("--annotations", metavar="JSON",
                    help="fichier d'annotations (cols, localités, sprints, points)")
    ap.add_argument("--marker", action="append", default=[], metavar="KM:TYPE:NOM",
                    help="annotation rapide, répétable, ex : --marker 0:localite:Pau "
                         "(types : localite, sprint, point)")
    ap.add_argument("--dump-annotations", metavar="JSON",
                    help="écrire un gabarit d'annotations à compléter")
    # --- rendu ---
    ap.add_argument("--fill", choices=["climbs", "slope", "altitude", "plain"],
                    default="climbs",
                    help="remplissage du profil : montées colorées (défaut), bandes "
                         "de pente, dégradé d'altitude, uni")
    ap.add_argument("--no-flags", action="store_true", help="masquer les drapeaux")
    ap.add_argument("--no-ribbon", action="store_true", help="masquer le bandeau de pente")
    ap.add_argument("--no-grid", action="store_true", help="masquer la grille")
    ap.add_argument("--no-legend", action="store_true", help="masquer la légende")
    ap.add_argument("--no-header", action="store_true", help="masquer le bandeau titre")
    ap.add_argument("--no-markers", action="store_true",
                    help="masquer les annotations localités/sprints/points")
    a = ap.parse_args()

    lat, lon, ele, wpts = parse_gpx(a.gpx)
    prof = build_profile(lat, lon, ele, a.smooth)
    climbs = detect_climbs(prof, a.min_prominence, a.min_length, a.min_grade, a.min_gain)

    names_from_waypoints(prof, climbs, wpts)
    if a.overpass:
        names_from_overpass(prof, [c for c in climbs if c.category])
    if a.names:
        given = [s.strip() for s in a.names.split(";")]
        ranked = [c for c in climbs if c.category or a.all]
        for c, nm in zip(ranked, given):
            if nm:
                c.name = nm

    # annotations
    ann = {"cols": [], "localites": [], "sprints": [], "points": []}
    if a.annotations:
        ann = load_annotations(a.annotations)
        apply_col_overrides(climbs, ann["cols"])
    markers = markers_from_annotations(ann) + [parse_marker_cli(s) for s in a.marker]

    if a.dump_annotations:
        save_annotations(a.dump_annotations, climbs, markers)
        print(f"Gabarit d'annotations écrit : {a.dump_annotations}")

    opts = RenderOptions(flags=not a.no_flags, ribbon=not a.no_ribbon,
                         grid=not a.no_grid, legend=not a.no_legend,
                         header=not a.no_header, markers=not a.no_markers,
                         fill=a.fill)

    dplus = elevation_gain(prof.alt)
    title = a.title or Path(a.gpx).stem.replace("_", " ")
    subtitle = a.subtitle or (f"{fmt(prof.d[-1] / 1000, 1)} km   |   D+ {fmt(dplus)} m   |   "
                              f"alt. max {fmt(prof.alt.max())} m")
    out_base = a.out or Path(a.gpx).with_suffix("")

    print(f"Distance {prof.d[-1] / 1000:.1f} km | D+ {dplus:.0f} m | alt max "
          f"{prof.alt.max():.0f} m | {len(climbs)} montées détectées "
          f"| {len(markers)} annotations\n")
    print_table(climbs)
    write_csv(f"{out_base}_cols.csv", climbs)
    outs = render(prof, climbs, title, subtitle, out_base,
                 [f.strip() for f in a.formats.split(",")], a.dpi, a.km_to_go,
                 a.all, a.vexag, opts=opts, markers=markers)
    print("\nFichiers :", ", ".join(str(p) for p in outs), f"+ {out_base}_cols.csv")


if __name__ == "__main__":
    main()
