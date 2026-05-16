import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AssetScriptTests(unittest.TestCase):
    def test_markdown_asset_generation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            cmd = [
                sys.executable,
                str(root / "scripts" / "generate_assets.py"),
                "--results",
                str(root / "docs" / "sample_results.json"),
                "--output",
                str(out),
                "--formats",
                "md",
            ]
            subprocess.run(cmd, cwd=root, check=True)
            self.assertTrue((out / "food_memory_case_study.md").exists())


if __name__ == "__main__":
    unittest.main()

