"""Tests unitaires de gpx2tdf.py — python -m unittest discover -s tests -v"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpx2tdf as g

STEP_KM = 0.1


def make_gpx(segments, base_alt=300.0, noise=0.0, seed=42):
    """Génère un GPX synthétique. segments = [(longueur_km, pente_%), ...]"""
    rng = np.random.default_rng(seed)
    pts = []
    d, alt, lat, lon = 0.0, base_alt, 43.10, 0.70
    for length, grade in segments:
        n = int(round(length / STEP_KM))
        for _ in range(n):
            a = alt + (noise * rng.standard_normal() if noise else 0.0)
            pts.append((lat, lon, d, a))
            alt += grade * STEP_KM * 10.0
            d += STEP_KM
            lon += STEP_KM / 82.0
    pts.append((lat, lon, d, alt))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="test">',
             "<trk><name>Trace test</name><trkseg>"]
    for la, lo, _d, e in pts:
        lines.append(f'<trkpt lat="{la:.6f}" lon="{lo:.6f}"><ele>{e:.2f}</ele></trkpt>')
    lines.append("</trkseg></trk></gpx>")
    return "\n".join(lines), d, alt


def build_from_segments(segments, **kw):
    xml, total_km, final_alt = make_gpx(segments, **kw)
    with tempfile.NamedTemporaryFile("w", suffix=".gpx", delete=False,
                                      encoding="utf-8") as f:
        f.write(xml)
        path = f.name
    lat, lon, ele, wpts = g.parse_gpx(path)
    prof = g.build_profile(lat, lon, ele, 100.0)
    climbs = g.detect_climbs(prof)
    return prof, climbs, total_km, path


class TestParsing(unittest.TestCase):
    def test_distance_and_altitude(self):
        prof, climbs, total_km, _ = build_from_segments(
            [(5.0, 0.0), (5.0, 6.0), (2.0, -3.0), (3.0, 8.0)]
        )
        self.assertAlmostEqual(prof.d[-1] / 1000, 15.0, delta=0.15)
        self.assertAlmostEqual(prof.alt[-1], 300 + 300 - 60 + 240, delta=20)
        self.assertGreaterEqual(len(climbs), 2)

    def test_missing_elevation_exits(self):
        xml = ('<?xml version="1.0"?><gpx><trk><trkseg>'
               '<trkpt lat="1" lon="1"></trkpt>'
               '<trkpt lat="1.001" lon="1"></trkpt>'
               "</trkseg></trk></gpx>")
        with tempfile.NamedTemporaryFile("w", suffix=".gpx", delete=False,
                                         encoding="utf-8") as f:
            f.write(xml)
            path = f.name
        with self.assertRaises(SystemExit):
            g.parse_gpx(path)

    def test_interpolation_of_missing_ele(self):
        xml = ('<?xml version="1.0"?><gpx><trk><trkseg>'
               '<trkpt lat="0" lon="0"><ele>100</ele></trkpt>'
               '<trkpt lat="0.001" lon="0"></trkpt>'
               '<trkpt lat="0.002" lon="0"><ele>200</ele></trkpt>'
               "</trkseg></trk></gpx>")
        with tempfile.NamedTemporaryFile("w", suffix=".gpx", delete=False,
                                         encoding="utf-8") as f:
            f.write(xml)
            path = f.name
        _, _, ele, _ = g.parse_gpx(path)
        self.assertAlmostEqual(ele[1], 150.0, delta=1e-6)


class TestClimbs(unittest.TestCase):
    def test_detection_and_categories(self):
        prof, climbs, _, _ = build_from_segments(
            [(5.0, 0.0), (5.0, 6.0), (2.0, -3.0), (3.0, 8.0)]
        )
        self.assertGreaterEqual(len(climbs), 2)
        first = min(climbs, key=lambda c: c.km_start)
        self.assertAlmostEqual(first.avg_grade, 6.0, delta=1.0)
        self.assertAlmostEqual(first.km_start, 5.0, delta=0.6)
        # 5 km à 6 % -> score 30000 -> catégorie 3 ou 2
        self.assertIn(first.category, ("2", "3"))
        # 3 km à 8 % -> score 24000 -> catégorie 3
        second = [c for c in climbs if c is not first][0]
        self.assertEqual(second.category, "3")

    def test_categorize_thresholds(self):
        self.assertEqual(g.categorize(85000), "HC")
        self.assertEqual(g.categorize(80000), "HC")
        self.assertEqual(g.categorize(79999), "1")
        self.assertEqual(g.categorize(40000), "2")
        self.assertEqual(g.categorize(20000), "3")
        self.assertEqual(g.categorize(9000), "4")
        self.assertEqual(g.categorize(100), "")

    def test_summit_finish_detected(self):
        prof, climbs, _, _ = build_from_segments([(5.0, 0.0), (8.0, 7.0)])
        self.assertTrue(climbs)
        last = max(climbs, key=lambda c: c.km_end)
        self.assertAlmostEqual(last.km_end, 13.0, delta=0.6)

    def test_elevation_gain(self):
        prof, _, _, _ = build_from_segments(
            [(5.0, 0.0), (5.0, 6.0), (2.0, -3.0), (3.0, 8.0)]
        )
        dplus = g.elevation_gain(prof.alt)
        self.assertAlmostEqual(dplus, 300 + 240, delta=30)


class TestAnnotations(unittest.TestCase):
    def test_parse_marker_cli(self):
        m = g.parse_marker_cli("12,0:localite:Pau: centre")
        self.assertEqual(m.km, 12.0)
        self.assertEqual(m.kind, "localite")
        self.assertEqual(m.label, "Pau: centre")
        m = g.parse_marker_cli("42:Sprint:Lourdes")
        self.assertEqual(m.kind, "sprint")
        m = g.parse_marker_cli("42:ville: Nay")
        self.assertEqual(m.kind, "localite")

    def test_parse_marker_cli_invalid(self):
        for bad in ("12", "12:patate:X"):
            with self.assertRaises(SystemExit):
                g.parse_marker_cli(bad)

    def test_load_and_dump_annotations(self):
        prof, climbs, _, _ = build_from_segments(
            [(5.0, 0.0), (5.0, 6.0), (2.0, -3.0), (3.0, 8.0)]
        )
        markers = [g.Marker(0.0, "Val de Luc", "localite"),
                   g.Marker(7.5, "Sprint de mi-course", "sprint")]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        g.save_annotations(path, climbs, markers)
        ann = g.load_annotations(path)
        self.assertEqual(len(ann["cols"]), len(climbs))
        self.assertEqual(ann["localites"][0]["nom"], "Val de Luc")
        self.assertEqual(ann["sprints"][0]["nom"], "Sprint de mi-course")

        ms = g.markers_from_annotations(ann)
        self.assertEqual(len(ms), 2)
        self.assertEqual(ms[0].kind, "localite")
        self.assertEqual(ms[1].kind, "sprint")

    def test_apply_col_overrides(self):
        prof, climbs, _, _ = build_from_segments(
            [(5.0, 0.0), (5.0, 6.0), (2.0, -3.0), (3.0, 8.0)]
        )
        target = min(climbs, key=lambda c: c.km_start)
        matched = g.apply_col_overrides(
            climbs, [{"km": target.km_end, "nom": "Col de Test", "categorie": "2"}]
        )
        self.assertEqual(matched, 1)
        self.assertEqual(target.name, "Col de Test")
        self.assertEqual(target.category, "2")

    def test_alt_at(self):
        prof, _, _, _ = build_from_segments([(5.0, 0.0), (5.0, 6.0)])
        # tolérance large : le lissage Savitzky-Golay « arrondit » l'angle plat/montée
        self.assertAlmostEqual(g.alt_at(prof, 7.5), 300 + 2.5 * 60, delta=10)


class TestRender(unittest.TestCase):
    def _render_all_modes(self, markers):
        prof, climbs, _, gpx_path = build_from_segments(
            [(5.0, 0.0), (5.0, 6.0), (2.0, -3.0), (3.0, 8.0)]
        )
        with tempfile.TemporaryDirectory() as td:
            for fill in ("climbs", "slope", "altitude", "plain"):
                out = Path(td) / f"out_{fill}"
                g.render(prof, climbs, "Titre", "Sous-titre", out, ["png"], 80,
                         False, False, 1.0,
                         opts=g.RenderOptions(fill=fill), markers=markers)
                p = Path(str(out) + ".png")
                self.assertTrue(p.exists() and p.stat().st_size > 10000,
                                f"PNG manquant pour fill={fill}")

    def test_render_modes_without_markers(self):
        self._render_all_modes([])

    def test_render_modes_with_markers_and_no_elements(self):
        markers = [g.Marker(0.0, "Val de Luc", "localite"),
                   g.Marker(7.5, "Sprint", "sprint"),
                   g.Marker(4.0, "Ravitaillement", "point")]
        prof, climbs, _, _ = build_from_segments(
            [(5.0, 0.0), (5.0, 6.0), (2.0, -3.0), (3.0, 8.0)]
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out_elements"
            g.render(prof, climbs, "Titre", "Sous-titre", out, ["png", "svg"], 80,
                     True, True, 0.8,
                     opts=g.RenderOptions(flags=False, ribbon=False, grid=False,
                                          legend=False, header=False, markers=True,
                                          fill="slope"),
                     markers=markers)
            self.assertTrue((Path(str(out) + ".png")).exists())
            svg = (Path(str(out) + ".svg")).read_text(encoding="utf-8")
            self.assertIn("Val de Luc".upper(), svg.upper())

    def test_build_figure_returns_figure(self):
        prof, climbs, _, _ = build_from_segments([(5.0, 0.0), (5.0, 6.0)])
        fig = g.build_figure(prof, climbs, "T", "S")
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
