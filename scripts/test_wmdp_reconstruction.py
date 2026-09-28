"""Offline regression checks; never calls the rewrite API."""

from pathlib import Path
import unittest

from rewrite_abstract import MODEL, PROMPT_PREFIX, build_payload, build_prompt, read_exact
from audit_wmdp_matches import field_text, matching_indices, normalize

ROOT = Path(__file__).resolve().parents[1]


class ReconstructionTests(unittest.TestCase):
    def test_every_saved_prompt_is_verbatim(self):
        prompts = sorted((ROOT / "rewrites").glob("*/prompt.md"))
        self.assertEqual(len(prompts), 5)
        for path in prompts:
            with self.subTest(path=path):
                saved = read_exact(path)
                self.assertTrue(saved.startswith(PROMPT_PREFIX))
                self.assertEqual(build_prompt(read_exact(path.with_name("original.md"))), saved)
                payload = build_payload(saved)
                self.assertEqual(payload["model"], "moonshotai/kimi-k2.5")
                self.assertEqual(payload["messages"], [{"role": "user", "content": saved}])
                self.assertEqual(MODEL, payload["model"])

    def test_match_is_all_terms_and_literal_without_boundaries(self):
        queries = [[normalize(t) for t in ["SLAM", "CD46"]]]
        self.assertEqual(matching_indices("slam only", queries), [])
        self.assertEqual(matching_indices("SLAMMING and ｃｄ４６", queries), [0])
        self.assertEqual(matching_indices("SLAM and CD-46", queries), [])

    def test_combined_field_includes_text_only_and_cross_field_matches(self):
        queries = [["alpha", "beta"]]
        self.assertEqual(field_text("", "beta", "abstract+text"), "beta")
        self.assertEqual(field_text("alpha", "beta", "abstract+text"), "alpha\n\nbeta")
        self.assertEqual(matching_indices(field_text("alpha", "beta", "abstract"), queries), [])
        self.assertEqual(matching_indices(field_text("alpha", "beta", "abstract+text"), queries), [0])


if __name__ == "__main__":
    unittest.main()
