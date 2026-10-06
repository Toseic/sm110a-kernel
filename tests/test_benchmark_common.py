import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import benchmarks.common as common
from benchmarks.common import (
    B300Sim,
    BenchmarkBase,
    SolReport,
    TablePrinter,
    print_sol_report,
)


class TuneOptionsTest(unittest.TestCase):
    def make_args(self, **overrides):
        values = dict(
            warmup=2,
            iters=3,
            samples=4,
            cache_path="cache.json",
            force_sweep=False,
            force_sweep_ours=False,
            force_sweep_baseline=False,
            debug=False,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_force_sweep_can_target_each_side_independently(self):
        args = self.make_args(force_sweep_ours=True)
        self.assertTrue(BenchmarkBase.make_tune_options(args, role="ours").force_sweep)
        self.assertFalse(
            BenchmarkBase.make_tune_options(args, role="baseline").force_sweep
        )

        args = self.make_args(force_sweep_baseline=True)
        self.assertFalse(BenchmarkBase.make_tune_options(args, role="ours").force_sweep)
        self.assertTrue(
            BenchmarkBase.make_tune_options(args, role="baseline").force_sweep
        )

    def test_legacy_force_sweep_still_targets_both_sides(self):
        args = self.make_args(force_sweep=True)
        self.assertTrue(BenchmarkBase.make_tune_options(args, role="ours").force_sweep)
        self.assertTrue(
            BenchmarkBase.make_tune_options(args, role="baseline").force_sweep
        )

    def test_b300_sim_uses_a_separate_autotune_cache(self):
        args = self.make_args(
            cache_path="profiles/custom.json", b300_sim=True, b300_sim_mode="bf16"
        )
        options = BenchmarkBase.make_tune_options(args)
        self.assertEqual(str(options.cache_path), "profiles/custom.b300-sim-bf16.json")

    def test_cache_arguments_expose_independent_sweep_controls(self):
        parser = argparse.ArgumentParser()
        BenchmarkBase().add_cache_args(parser)
        args = parser.parse_args(["--force-sweep-ours"])
        self.assertTrue(args.force_sweep_ours)
        self.assertFalse(args.force_sweep_baseline)


class OutputFormatTest(unittest.TestCase):
    def test_table_printer_uses_one_record_per_line_in_debug_mode(self):
        output = io.StringIO()
        with redirect_stdout(output):
            table = TablePrinter((("M", 4), ("Time", 8)), debug=True)
            table.print_header()
            table.print_row(128, "1.25")
        self.assertEqual(output.getvalue(), "M=128 Time=1.25\n")

    def test_sol_report_is_a_table_and_debug_is_line_oriented(self):
        bounds = {
            "mma_ms": 1.0,
            "hbm_ms": 2.0,
            "gmem_ms": 3.0,
            "pure_l2_ms": 4.0,
            "best_l2_ms": 5.0,
            "sol_ms": 4.0,
            "tile_read_bytes": 6 * 1000**2,
        }
        normal = io.StringIO()
        with redirect_stdout(normal):
            print_sol_report(bounds, 8.0, label="GEMM")
        self.assertIn("SOL", normal.getvalue().splitlines()[0])
        self.assertEqual(len(normal.getvalue().splitlines()), 2)

        debug = io.StringIO()
        with redirect_stdout(debug):
            print_sol_report(bounds, 8.0, label="GEMM", debug=True)
        self.assertEqual(len(debug.getvalue().splitlines()), 1)
        self.assertIn("SOL=GEMM", debug.getvalue())

    def test_sol_report_collector_emits_one_table_for_a_sweep(self):
        bounds = {
            "mma_ms": 1.0,
            "hbm_ms": 2.0,
            "gmem_ms": 3.0,
            "pure_l2_ms": 4.0,
            "best_l2_ms": 5.0,
            "sol_ms": 4.0,
            "tile_read_bytes": 6 * 1000**2,
        }
        output = io.StringIO()
        with redirect_stdout(output):
            report = SolReport()
            report.add(bounds, 8.0, label="FC1 kernel (tokens=2048, routing excluded)")
            report.add(bounds, 10.0, label="FC1 kernel (tokens=4096, routing excluded)")
            report.print()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("SOL mma(ms)", lines[0])
        self.assertIn("tokens=2048", lines[1])
        self.assertIn("tokens=4096", lines[2])


class B300SimTest(unittest.TestCase):
    def test_locks_nearest_gpc_and_restores_all_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu = root / "gpu"
            emc = root / "emc"
            for path, frequencies, minimum, maximum in (
                (gpu, "315000000 450000000 468000000 702000000", 315000000, 1575000000),
                (emc, "665600000 4266000000", 665600000, 4266000000),
            ):
                path.mkdir()
                (path / "available_frequencies").write_text(frequencies)
                (path / "min_freq").write_text(f"{minimum}\n")
                (path / "max_freq").write_text(f"{maximum}\n")

            original_gpu, original_emc = B300Sim.GPU_PATH, B300Sim.EMC_PATH
            B300Sim.GPU_PATH, B300Sim.EMC_PATH = gpu, emc
            try:
                args = argparse.Namespace(b300_sim=True, b300_sim_mode="bf16")
                with B300Sim(args):
                    self.assertEqual(
                        (gpu / "min_freq").read_text().strip(), "468000000"
                    )
                    self.assertEqual(
                        (gpu / "max_freq").read_text().strip(), "468000000"
                    )
                    self.assertEqual(
                        (emc / "min_freq").read_text().strip(), "4266000000"
                    )
                    bounds = common.sol_bounds(250 * 1000**3, 0, 0, 0)
                    self.assertAlmostEqual(bounds["mma_ms"], 1575 / 468, places=6)
                    self.assertAlmostEqual(common.MMA_PEAK_SCALE, 468 / 1575, places=12)
                self.assertEqual((gpu / "min_freq").read_text().strip(), "315000000")
                self.assertEqual((gpu / "max_freq").read_text().strip(), "1575000000")
                self.assertEqual((emc / "min_freq").read_text().strip(), "665600000")
                self.assertEqual((emc / "max_freq").read_text().strip(), "4266000000")
                self.assertEqual(common.MMA_PEAK_SCALE, 1.0)

                nvfp4_args = argparse.Namespace(b300_sim=True, b300_sim_mode="nvfp4")
                with B300Sim(nvfp4_args):
                    self.assertEqual(
                        (gpu / "min_freq").read_text().strip(), "702000000"
                    )
                    self.assertAlmostEqual(
                        nvfp4_args._sol_mma_peak_tflops,
                        4 * 250 * 702 / 1575,
                        places=6,
                    )

                with B300Sim(argparse.Namespace(b300_sim=False, no_lock_clocks=False)):
                    self.assertEqual(
                        (gpu / "min_freq").read_text().strip(), "1575000000"
                    )
                    self.assertEqual(
                        (gpu / "max_freq").read_text().strip(), "1575000000"
                    )
                    self.assertEqual(
                        (emc / "min_freq").read_text().strip(), "4266000000"
                    )
                    self.assertEqual(
                        (emc / "max_freq").read_text().strip(), "4266000000"
                    )
            finally:
                B300Sim.GPU_PATH, B300Sim.EMC_PATH = original_gpu, original_emc


if __name__ == "__main__":
    unittest.main()
