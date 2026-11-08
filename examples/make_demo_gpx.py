#!/usr/bin/env python3
"""Génère examples/demo_etape.gpx : étape de montagne fictive (~118 km)."""
import sys
from pathlib import Path

import numpy as np

SEGMENTS = [  # (longueur_km, pente_%)
    (38.0, 0.4),    # vallée ondulante
    (4.0, 4.5),     # Côte des Trois Ponts (cat. 3)
    (8.0, -2.0),    # descente douce
    (13.0, 7.0),    # Col de la Serrat (HC)
    (6.0, -6.0),    # descente raide
    (27.0, 0.3),    # vallée plate
    (4.0, 5.5),     # Côte de Peyraque (cat. 3)
    (5.0, -4.0),    # descente
    (14.0, 7.6),    # Montée de l'Aiguille (HC) — arrivée au sommet
]

rng = np.random.default_rng(7)
lat, lon, d, alt = 43.10, 0.70, 0.0, 300.0
rows = []
for length, grade in SEGMENTS:
    n = int(round(length / 0.05))
    for _ in range(n):
        noise = 1.2 * np.sin(d * 7.3) + 0.8 * rng.standard_normal()
        rows.append((lat, lon, alt + noise))
        alt += grade * 0.05 * 10.0
        d += 0.05
        lon += 0.05 / 82.0
        lat += 0.00007
rows.append((lat, lon, alt))

lines = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<gpx version="1.1" creator="demo" xmlns="http://www.topografix.com/GPX/1/1">',
         "<trk><name>Étape de démonstration — Val de Luc &gt; Aiguille de la Serrat</name><trkseg>"]
for la, lo, e in rows:
    lines.append(f'<trkpt lat="{la:.6f}" lon="{lo:.6f}"><ele>{e:.2f}</ele></trkpt>')
lines.append("</trkseg></trk></gpx>")

out = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/demo_etape.gpx")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{out} : {len(rows)} points, {d:.1f} km, alt. finale {alt:.0f} m")
