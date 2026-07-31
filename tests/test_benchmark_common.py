import argparse
import unittest

from benchmarks.common import BenchmarkBase


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


if __name__ == "__main__":
    unittest.main()
