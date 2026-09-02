import argparse
import io
import unittest
from contextlib import redirect_stdout

from benchmarks.common import BenchmarkBase, TablePrinter, print_sol_report


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
        self.assertFalse(BenchmarkBase.make_tune_options(args, role="baseline").force_sweep)

        args = self.make_args(force_sweep_baseline=True)
        self.assertFalse(BenchmarkBase.make_tune_options(args, role="ours").force_sweep)
        self.assertTrue(BenchmarkBase.make_tune_options(args, role="baseline").force_sweep)

    def test_legacy_force_sweep_still_targets_both_sides(self):
        args = self.make_args(force_sweep=True)
        self.assertTrue(BenchmarkBase.make_tune_options(args, role="ours").force_sweep)
        self.assertTrue(BenchmarkBase.make_tune_options(args, role="baseline").force_sweep)

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
            "mma_ms": 1.0, "hbm_ms": 2.0, "gmem_ms": 3.0,
            "pure_l2_ms": 4.0, "best_l2_ms": 5.0, "sol_ms": 4.0,
            "tile_read_bytes": 6 * 1000 ** 2,
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


if __name__ == "__main__":
    unittest.main()
