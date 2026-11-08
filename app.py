#!/usr/bin/env python3
"""
app.py - Interface Streamlit pour gpx2tdf.py

Lancement :
    pip install numpy scipy matplotlib streamlit pandas
    streamlit run app.py

Fonctionnalités :
  - chargement d'un GPX (glisser-déposer), paramètres de lissage et de
    détection réglables en direct
  - renommage des montées détectées directement dans le tableau
  - annotations éditables : localités, sprints intermédiaires, points libres
    (ajout / suppression / réordonnancement libre)
  - chaque élément du rendu affichable ou masquable (drapeaux, bandeau de
    pente, grille, légende, titre, annotations)
  - export PNG et SVG téléchargeables
"""
import io

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

import gpx2tdf as g

st.set_page_config(page_title="Profil altimétrique — TDF", page_icon="🚴", layout="wide")
st.title("🚴 Profil altimétrique — style Tour de France")
st.caption(
    "Chargez un GPX, ajustez le profil, renommez les montées, ajoutez localités et "
    "sprints, puis exportez en PNG ou SVG. Tout est calculé localement."
)

FILL_LABELS = {
    "climbs": "Montées colorées par catégorie",
    "slope": "Bandes de pente sur tout le profil",
    "altitude": "Dégradé d'altitude",
    "plain": "Uni (beige)",
}


# --------------------------------------------------------------------------
# Calcul (mis en cache) : parsing + lissage + détection
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Analyse du GPX…")
def compute(gpx_bytes, smooth, min_prominence, min_length, min_grade):
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".gpx", delete=False) as f:
        f.write(gpx_bytes)
        path = f.name
    try:
        lat, lon, ele, wpts = g.parse_gpx(path)
        prof = g.build_profile(lat, lon, ele, smooth)
        climbs = g.detect_climbs(prof, min_prominence, min_length, min_grade)
        return prof, climbs
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# Barre latérale
# --------------------------------------------------------------------------
with st.sidebar:
    up = st.file_uploader("Fichier GPX", type=["gpx"], key="gpx_up")
    st.divider()
    st.subheader("Profil")
    smooth = st.slider("Lissage (m)", 0, 600, 200, 20,
                       help="Fenêtre du filtre Savitzky-Golay, en mètres.")
    vexag = st.slider("Exagération verticale", 0.5, 2.0, 1.0, 0.1,
                      help="< 1 : profil plus escarpé ; > 1 : aplati.")
    fill = st.selectbox("Remplissage", list(FILL_LABELS), format_func=FILL_LABELS.get)
    km_to_go = st.checkbox("Axe en km restants (style TDF)")
    show_all = st.checkbox("Afficher aussi les montées non classées")
    st.divider()
    st.subheader("Détection des montées")
    min_prom = st.slider("Proéminence mini (m)", 10, 150, 25, 5)
    min_len = st.slider("Longueur mini (m)", 200, 3000, 500, 100)
    min_grade = st.slider("Pente moyenne mini (%)", 2.0, 8.0, 3.0, 0.5)
    st.divider()
    st.subheader("Éléments affichés")
    flags = st.checkbox("Drapeaux des montées", True)
    ribbon = st.checkbox("Bandeau de pente", True)
    grid = st.checkbox("Grille", True)
    legend = st.checkbox("Légende", True)
    header = st.checkbox("Bandeau titre", True)
    markers_on = st.checkbox("Annotations (localités, sprints, points)", True)
    dpi = st.slider("Résolution d'export (DPI)", 96, 300, 130, 10)

if up is None:
    st.info(
        "Déposez un fichier .gpx pour commencer.\n\n"
        "Pas de trace sous la main ? Générez l'étape de démonstration du dépôt :\n"
        "`python examples/make_demo_gpx.py examples/demo_etape.gpx`"
    )
    st.stop()

try:
    prof, climbs = compute(up.getvalue(), smooth, min_prom, min_len, min_grade)
except SystemExit as exc:
    st.error(str(exc))
    st.stop()

# --------------------------------------------------------------------------
# Titres
# --------------------------------------------------------------------------
dplus = g.elevation_gain(prof.alt)
default_title = up.name.replace(".gpx", "").replace("_", " ").strip() or "Profil"
default_sub = (f"{g.fmt(prof.d[-1] / 1000, 1)} km   |   D+ {g.fmt(dplus)} m   |   "
               f"alt. max {g.fmt(prof.alt.max())} m")

c1, c2 = st.columns([2, 3])
with c1:
    title = st.text_input("Titre", value=default_title)
with c2:
    subtitle = st.text_input("Sous-titre", value=default_sub)

# --------------------------------------------------------------------------
# Statistiques clés
# --------------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Distance", f"{g.fmt(prof.d[-1] / 1000, 1)} km")
m2.metric("D+", f"{g.fmt(dplus)} m")
m3.metric("Alt. max", f"{g.fmt(prof.alt.max())} m")
m4.metric("Montées", len(climbs))
m5.metric("Sommet final", "oui" if (len(climbs) and
           abs(climbs[-1].km_end - prof.d[-1] / 1000) < 1.0) else "—")

# --------------------------------------------------------------------------
# Montées détectées : renommage dans le tableau
# --------------------------------------------------------------------------
if climbs:
    df = pd.DataFrame({
        "nom": [c.name or "" for c in climbs],
        "categorie": [c.category or "—" for c in climbs],
        "km sommet": [round(c.km_end, 1) for c in climbs],
        "longueur (km)": [round(c.length_m / 1000, 1) for c in climbs],
        "pente moy (%)": [round(c.avg_grade, 1) for c in climbs],
        "D+ (m)": [round(c.gain_m) for c in climbs],
    })
    edited = st.data_editor(
        df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "nom": st.column_config.TextColumn("Nom (modifiable)", width="large"),
        },
        disabled=["categorie", "km sommet", "longueur (km)", "pente moy (%)", "D+ (m)"],
        key="climb_editor",
    )
    for c, nm in zip(climbs, edited["nom"].tolist()):
        nm = str(nm).strip()
        if nm:
            c.name = nm

# --------------------------------------------------------------------------
# Annotations : localités, sprints, points (ajout / suppression libre)
# --------------------------------------------------------------------------
st.subheader("Annotations")
st.caption("Ajoutez ou supprimez des lignes : elles apparaissent sur le profil au fil de l'eau. "
           "Les localités sont affichées en haut du profil, les sprints avec un point vert, "
           "les points libres avec une pastille sur la courbe.")

tab_loc, tab_spr, tab_pts = st.tabs(["🏘️ Localités", "🟢 Sprints intermédiaires", "📍 Points libres"])


def _editor(tab, key, placeholder_km, placeholder_nom):
    if key not in st.session_state:
        st.session_state[key] = []
    df = pd.DataFrame(st.session_state[key], columns=["km", "nom"])
    if df.empty:
        df = pd.DataFrame([{"km": None, "nom": None}], columns=["km", "nom"])
    ed = tab.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "km": st.column_config.NumberColumn(placeholder_km, min_value=0.0, step=0.1),
            "nom": st.column_config.TextColumn(placeholder_nom),
        },
        key=f"{key}_editor",
    )
    records = []
    for _, row in ed.iterrows():
        if pd.isna(row["km"]) or pd.isna(row["nom"]) or not str(row["nom"]).strip():
            continue
        records.append({"km": float(row["km"]), "nom": str(row["nom"]).strip()})
    st.session_state[key] = records
    return records


loc_records = _editor(tab_loc, "annot_localites", "km (ex. 42.0)", "nom (ex. Pau)")
spr_records = _editor(tab_spr, "annot_sprints", "km (ex. 86.5)", "nom (ex. Lourdes)")
pts_records = _editor(tab_pts, "annot_points", "km (ex. 45.0)", "nom (ex. Ravitaillement)")

markers = (
    [g.Marker(r["km"], r["nom"], "localite") for r in loc_records]
    + [g.Marker(r["km"], r["nom"], "sprint") for r in spr_records]
    + [g.Marker(r["km"], r["nom"], "point") for r in pts_records]
)
markers.sort(key=lambda m: m.km)

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
opts = g.RenderOptions(flags=flags, ribbon=ribbon, grid=grid, legend=legend,
                       header=header, markers=markers_on, fill=fill)
fig = g.build_figure(prof, climbs, title, subtitle, km_to_go=km_to_go,
                     show_all=show_all, vexag=vexag, opts=opts, markers=markers)
st.pyplot(fig)

# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------
buf_png = io.BytesIO()
fig.savefig(buf_png, format="png", dpi=dpi, facecolor=fig.get_facecolor())
buf_svg = io.BytesIO()
fig.savefig(buf_svg, format="svg", facecolor=fig.get_facecolor())
import matplotlib.pyplot as plt
plt.close(fig)

stem = up.name.rsplit(".", 1)[0]
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("⬇️ Exporter en PNG", buf_png.getvalue(),
                       f"{stem}_profil.png", "image/png")
with d2:
    st.download_button("⬇️ Exporter en SVG", buf_svg.getvalue().decode("utf-8"),
                       f"{stem}_profil.svg", "image/svg+xml")
with d3:
    ann = {
        "cols": [{"km": round(c.km_end, 1), "nom": c.name, "categorie": c.category}
                 for c in climbs],
        "localites": loc_records,
        "sprints": spr_records,
        "points": pts_records,
    }
    st.download_button("⬇️ Annotations (JSON)",
                       __import__("json").dumps(ann, ensure_ascii=False, indent=2),
                       f"{stem}_annotations.json", "application/json",
                       help="Réutilisable avec : python gpx2tdf.py trace.gpx --annotations …")
