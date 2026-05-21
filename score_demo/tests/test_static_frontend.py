import re
import unittest
from pathlib import Path


class StaticFrontendTests(unittest.TestCase):
    def test_frontend_is_english_only_and_contains_scoring_controls(self):
        html = Path("app/web/index.html").read_text(encoding="utf-8")
        script = Path("app/web/app.js").read_text(encoding="utf-8")

        visible_text = html + "\n" + script

        self.assertIn("Essay Scoring", html)
        self.assertIn("Score Essay", html)
        self.assertIn("Paste an English essay", html)
        self.assertNotRegex(visible_text, re.compile(r"[\u4e00-\u9fff]"))


if __name__ == "__main__":
    unittest.main()
