"""Tests for the analytical-figures skill.

The skill is a package of library modules (`scripts/*.py`, relative imports) plus two argparse
entry points (`scripts/cli.py`, `scripts/bundle.py`). The suite puts the SKILL ROOT on sys.path
so `from scripts import ...` resolves exactly as the skill's own docs say, then checks: the
structure the docs promise, that every module imports, the band-integration primitive against a
hand-built construction (on both axis orders, since raw .spc exports run 4000 -> 650 cm-1), the
calibration statistics against closed-form values, that `bundle.py` turns the shipped template
into a self-contained script, and the `--help` contract for both CLIs.

Needs numpy + matplotlib (+ scipy for the exact t-multipliers); skips without them.

    uv run --with pytest python -m pytest tests/analytical-figures -q
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
import skill_contract

SKILL_ROOT = Path(__file__).resolve().parents[2] / "skills" / "analytical-figures"
SCRIPTS_DIR = SKILL_ROOT / "scripts"

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from scripts import calibration, spectra  # noqa: E402
from scripts.config import Config  # noqa: E402

CliHelpTests = skill_contract.cli.help_test_case(SKILL_ROOT)

ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"}


# ---------------------------------------------------------------- structure


class TestSkillStructure(unittest.TestCase):
    REFERENCES = ["figure_selection", "cocrystal_id", "databases", "verification", "spectra",
                  "calibration", "chemometrics", "charts", "crystal", "adding_a_family"]
    MODULES = ["config", "style", "verify", "spectra", "calibration", "chemometrics", "charts",
               "report", "doe", "crystal_engine", "crystal_pxrd", "crystal_view", "cocrystal",
               "pxrd_realism", "sources"]

    def test_skill_md_sections(self):
        text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        for heading in ("## Output contract", "## Workflow", "## Hard principles", "## Files"):
            self.assertIn(heading, text)

    def test_reference_docs_exist(self):
        for name in self.REFERENCES:
            self.assertTrue((SKILL_ROOT / "references" / f"{name}.md").is_file(), name)

    def test_modules_and_entry_points_exist(self):
        for name in self.MODULES + ["bundle", "cli"]:
            self.assertTrue((SCRIPTS_DIR / f"{name}.py").is_file(), name)
        self.assertTrue((SKILL_ROOT / "assets" / "templates" / "analysis_template.py").is_file())

    def test_journal_style_presets_exist(self):
        for journal in ("nature", "acs", "ieee", "general"):
            preset = SKILL_ROOT / "assets" / "styles" / f"analytical-{journal}.mplstyle"
            self.assertTrue(preset.is_file(), preset.name)


class TestImports(unittest.TestCase):
    CORE = ["config", "style", "verify", "spectra", "calibration", "chemometrics", "charts",
            "report", "doe", "cocrystal", "pxrd_realism", "sources"]
    CRYSTAL = ["crystal_engine", "crystal_pxrd", "crystal_view"]

    def test_core_modules_import(self):
        for name in self.CORE:
            with self.subTest(module=name):
                importlib.import_module(f"scripts.{name}")

    def test_crystal_modules_import_or_name_their_optional_dependency(self):
        for name in self.CRYSTAL:
            with self.subTest(module=name):
                try:
                    importlib.import_module(f"scripts.{name}")
                except ImportError as error:  # gemmi / Dans_Diffraction are lazy and optional
                    self.assertTrue(str(error), "ImportError without a message")


# ---------------------------------------------------------------- band integration


def _trace(seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(650, 4000, 1798)
    y = (0.02 + 1e-5 * (x - 650) + 0.8 * np.exp(-0.5 * ((x - 1748) / 8.0) ** 2)
         + 0.5 * np.exp(-0.5 * ((x - 1640) / 20.0) ** 2) + 0.004 * rng.normal(size=x.size))
    return x, y


class TestBandIntegration(unittest.TestCase):
    def test_three_callers_agree_on_both_axis_orders(self):
        x, y = _trace()
        lo, hi = 1713.0, 1789.0
        cfg = Config(strict=False, integration_windows=[(lo, hi, "ester")])
        for xx, yy in ((x, y), (x[::-1], y[::-1])):
            a = spectra.band_metric(xx, yy, "area", anchors=(lo, hi))
            b = spectra.integrate_bands(xx, yy, cfg)[0]["area"]
            c = spectra.band_area(xx, yy, lo, hi)["area"]
            self.assertGreater(a, 0)
            self.assertLess(abs(a - b), 1e-12 * a)
            self.assertLess(abs(a - c), 1e-12 * a)

    def test_shared_envelope_equals_hand_built_line(self):
        x, y = _trace(1)
        cfg = Config(strict=False, integration_baseline="shared",
                     integration_windows=[(1500.0, 1700.0, "carbox"), (1700.0, 1800.0, "ester")])
        got = {r["name"]: r["area"] for r in spectra.integrate_bands(x, y, cfg)}
        y_lo, y_hi = spectra._anchor_val(x, y, 1500.0), spectra._anchor_val(x, y, 1800.0)
        for a, b, name in cfg.integration_windows:
            line = tuple(np.interp([a, b], [1500.0, 1800.0], [y_lo, y_hi]))
            m = (x >= a) & (x <= b)
            corr = np.clip(y[m] - np.interp(x[m], [a, b], line), 0, None)
            self.assertLess(abs(got[name] - np.trapezoid(corr, x[m])), 1e-12 * got[name])

    def test_area_normalisation_is_positive_on_a_descending_axis(self):
        x = np.linspace(650, 4000, 1798)
        y = 0.01 + 0.8 * np.exp(-0.5 * ((x - 1748) / 8.0) ** 2)
        cfg = Config(strict=False, normalize="area")
        yn = spectra.normalize(x[::-1], y[::-1], cfg)
        self.assertTrue(np.all(yn >= 0))
        self.assertLess(abs(abs(np.trapezoid(yn, x[::-1])) - 1.0), 1e-9)

    def test_empty_window_is_nan(self):
        x, y = _trace(2)
        self.assertTrue(np.isnan(spectra.band_area(x, y, 5000, 5001)["area"]))


# ---------------------------------------------------------------- calibration


class TestCalibration(unittest.TestCase):
    def test_exact_line_is_recovered(self):
        x = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        y = 0.5 + 0.02 * x
        model = calibration.fit(x, y, Config(strict=False))
        self.assertAlmostEqual(model["slope"], 0.02, places=12)
        self.assertAlmostEqual(model["intercept"], 0.5, places=12)
        self.assertAlmostEqual(model["r2"], 1.0, places=12)

    def test_lod_loq_are_residual_sd_over_slope(self):
        rng = np.random.default_rng(3)
        x = np.repeat([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], 3)
        y = 0.1 + 0.05 * x + rng.normal(0, 0.02, x.size)
        cfg = Config(strict=False)
        model = calibration.fit(x, y, cfg)
        limits = calibration.lod_loq(model, cfg)
        self.assertAlmostEqual(limits["lod"], 3.3 * model["s_resid"] / model["slope"], places=9)
        self.assertAlmostEqual(limits["loq"], 10.0 * model["s_resid"] / model["slope"], places=9)
        self.assertIn("residual_sd", limits["method"])


# ---------------------------------------------------------------- bundle + cli


class TestBundle(unittest.TestCase):
    def test_template_bundles_to_a_self_contained_script(self):
        body = SKILL_ROOT / "assets" / "templates" / "analysis_template.py"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "standalone.py"
            result = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "bundle.py"), str(body), "-o", str(out)],
                capture_output=True, text=True, cwd=str(SKILL_ROOT), env=ENV,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-1500:])
            src = out.read_text(encoding="utf-8")
        self.assertNotIn("from scripts", src)
        self.assertNotIn("from . import", src)
        self.assertIn("def apply_style", src)
        self.assertIn("def save_fig", src)
        compile(src, "standalone.py", "exec")


class TestCliSubcommands(unittest.TestCase):
    def test_subcommand_help(self):
        for sub in ("spc-rsd", "calibrate"):
            with self.subTest(subcommand=sub):
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS_DIR / "cli.py"), sub, "--help"],
                    capture_output=True, text=True, env=ENV,
                )
                self.assertEqual(result.returncode, 0, result.stderr[-800:])
                self.assertIn("usage", result.stdout.lower())
