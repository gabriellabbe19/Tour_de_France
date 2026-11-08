# Tour_de_France 🚴

Assistant de conception d'un Tour de France réaliste : tracé de parcours,
profils altimétriques, cols et contraintes géographiques.

## Outil : profil altimétrique style Tour de France

`gpx2tdf.py` transforme un fichier GPX (RideWithGPS, BRouter-web, Komoot,
Strava, Garmin…) en profil altimétrique « style Tour de France » : montées
détectées et catégorisées, drapeaux, bandeau de pente, et annotations
personnalisées.

### Installation

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib (+ streamlit, pandas pour l'app)
```

### Utilisation en ligne de commande

```bash
# Profil simple : PNG + tableau + CSV
python gpx2tdf.py etape.gpx --title "Étape 14 - Pau > Luz Ardiden"

# Modes de remplissage : montées colorées (défaut), bandes de pente, dégradé d'altitude, uni
python gpx2tdf.py etape.gpx --fill slope

# Axe en km restants (style TDF), montées non classées incluses
python gpx2tdf.py etape.gpx --km-to-go --all

# Gabarit d'annotations à compléter, puis rendu annoté
python gpx2tdf.py etape.gpx --dump-annotations annos.json
python gpx2tdf.py etape.gpx --annotations annos.json

# Annotations rapides en une commande
python gpx2tdf.py etape.gpx --marker 0:localite:Pau --marker 86.5:sprint:Lourdes

# Masquer des éléments un par un
python gpx2tdf.py etape.gpx --no-ribbon --no-legend --dpi 200 --formats png,svg
```

Options principales : `--smooth` (lissage en m), `--min-prominence`,
`--min-length`, `--min-grade`, `--min-gain` (détection), `--names` (noms dans
l'ordre, `;` pour sauter), `--overpass` (noms via OpenStreetMap), `--vexag`
(exagération verticale), `--no-flags / --no-ribbon / --no-grid / --no-legend /
--no-header / --no-markers` (éléments masquables), `--formats png,svg,pdf`.

### Annotations (fichier JSON)

Toutes les clés sont optionnelles ; chaque liste s'édite librement :

```json
{
  "cols":      [{ "km": 108.3, "nom": "Col du Grand Vent", "categorie": "HC" }],
  "localites": [{ "km": 0.0,   "nom": "Pau" }],
  "sprints":   [{ "km": 86.5,  "nom": "Lourdes" }],
  "points":    [{ "km": 45.0,  "nom": "Ravitaillement" }]
}
```

- **cols** : renomme / recatégorise la montée détectée au km du sommet
  (tolérance ± 2 km) ;
- **localites** : nom affiché en haut du profil, filet vertical ;
- **sprints** : point vert « S » avec étiquette ;
- **points** : pastille libre sur la courbe (ravitaillement, zone clé…).

`--dump-annotations` écrit ce fichier pré-rempli avec les montées détectées.

### Application Streamlit interactive

```bash
streamlit run app.py
```

- glisser-déposer d'un GPX, lissage et seuils de détection réglés en direct ;
- renommage des montées dans le tableau ;
- onglets **Localités / Sprints / Points** : ajout et suppression de lignes
  libre, profil mis à jour à chaque saisie ;
- chaque élément affichable/masqué (drapeaux, bandeau de pente, grille,
  légende, titre, annotations) ;
- exports **PNG / SVG** téléchargeables + fichier d'annotations JSON
  réutilisable en CLI.

### Catégorisation

Score = longueur (m) × pente moyenne (%), barème type Strava :

| Catégorie | Score minimal |
|-----------|---------------|
| HC        | 80 000        |
| 1         | 64 000        |
| 2         | 32 000        |
| 3         | 16 000        |
| 4         | 8 000         |

Le barème est modifiable en tête de `gpx2tdf.py` (`CATEGORIES`).

## Structure du dépôt

```
gpx2tdf.py                     # CLI + bibliothèque (parsing, détection, rendu)
app.py                         # application Streamlit
tests/test_gpx2tdf.py          # tests unitaires
examples/make_demo_gpx.py      # générateur de l'étape de démonstration
examples/annotations_demo.json # annotations complètes de la démo
requirements.txt
```

Étape de démonstration (119 km fictifs, 2 HC, arrivée au sommet) :

```bash
python examples/make_demo_gpx.py examples/demo_etape.gpx
python gpx2tdf.py examples/demo_etape.gpx --annotations examples/annotations_demo.json
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- un GPX sans balises `<ele>` est refusé : réexportez avec l'altitude, ou
  complétez-la via un MNT (SRTM, IGN RGE ALTI) ;
- le D+ est calculé sur le profil lissé (seuil de 2 m par pas de référence)
  pour éviter l'inflation due au bruit GPS.
