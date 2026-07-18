import tempfile
import unittest

try:
    import torch
    from test_ddfsd import load_checkpoint
except ModuleNotFoundError:
    torch = None
    load_checkpoint = None


@unittest.skipUnless(torch is not None, "torch is unavailable in this Python environment")
class CheckpointLoadingTest(unittest.TestCase):
    def test_load_checkpoint_without_runtime_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/valid.pth"
            torch.save({"model": {"dummy": torch.tensor([1.0])}, "step": 15000}, path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["step"], 15000)
            self.assertIn("model", loaded)

    def test_load_checkpoint_missing_model_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/invalid.pth"
            torch.save({"step": 15000}, path)
            with self.assertRaisesRegex(KeyError, "no 'model' key"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
