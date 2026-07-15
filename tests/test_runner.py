import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sm110a_kernel.runner import AutotuneRunner, PreparedKernel, TuneOptions


@dataclasses.dataclass(frozen=True)
class Config:
    value: int


class RunnerCacheTest(unittest.TestCase):
    def test_cache_hit_skips_second_sweep(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "sm110a_kernel.runner.environment_fingerprint", return_value="test-device"
        ):
            path = Path(directory) / "cache.json"
            calls = []

            def prepare(config):
                return PreparedKernel(config, lambda: None)

            def benchmark(prepared):
                calls.append(prepared.candidate.value)
                return {"min_ms": float(prepared.candidate.value), "median_ms": float(prepared.candidate.value)}

            options = TuneOptions(cache_path=path)
            first = AutotuneRunner(options).resolve(
                (Config(2), Config(1)), namespace="unit", key=(1,), prepare=prepare,
                benchmark=benchmark, probe=None, format_candidate=str,
            )
            self.assertEqual(first.candidate, Config(1))
            self.assertEqual(calls, [2, 1])

            calls.clear()
            second = AutotuneRunner(options).resolve(
                (Config(2), Config(1)), namespace="unit", key=(1,), prepare=prepare,
                benchmark=benchmark, probe=None, format_candidate=str,
            )
            self.assertTrue(second.cache_hit)
            self.assertEqual(calls, [1])

    def test_legacy_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "sm110a_kernel.runner.environment_fingerprint", return_value="test-device"
        ):
            path = Path(directory) / "cache.json"
            path.write_text('{"unit": {}}')
            runner = AutotuneRunner(TuneOptions(cache_path=path))
            self.assertEqual(runner._load()["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
