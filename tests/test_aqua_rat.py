"""Unit tests for the AQuA-RAT record normaliser (no torch, no network)."""
import unittest

from src.benchmarks.aqua_rat import _from_record
from src.benchmarks.base import StandardRow

# Two hand-written records in the shape of the deepmind/aqua_rat "raw" config
# (no network: the loader's _from_record is pure).
RECORD_ICE = {
    "question": "A grocery sells a bag of ice for $1.25, and makes 20% profit. "
                "If it sells 500 bags of ice, how much total profit does it make?",
    "options": ["A)125", "B)150", "C)225", "D)250", "E)275"],
    "rationale": "Profit per bag = 1.25 * 0.20 = 0.25\nTotal profit = 500 * 0.25 = 125\nAnswer is A.",
    "correct": "A",
}
RECORD_SPEED = {
    "question": "  A train covers 60 km in 45 minutes. What is its speed in km/h? ",
    "options": ["A) 45", "B)60", "C )75", "D)80", "e)90"],
    "rationale": "60 km / 0.75 h = 80 km/h. Answer D.",
    "correct": "d",
}


class FromRecordTest(unittest.TestCase):
    def test_choices_drop_the_letter_prefix(self):
        row = _from_record(RECORD_ICE, 0, "test")
        self.assertIsNotNone(row)
        self.assertEqual(row.choices, {"A": "125", "B": "150", "C": "225", "D": "250", "E": "275"})
        self.assertEqual(row.ground_truth, "A")
        self.assertEqual(row.benchmark_name, "aqua_rat")
        self.assertEqual(row.task_subtype, "aqua_rat")
        self.assertEqual(row.split, "test")
        self.assertEqual(row.example_id, 0)
        self.assertEqual(row.context, "")
        self.assertEqual(row.metadata["n_options"], 5)
        self.assertEqual(row.metadata["rationale"], RECORD_ICE["rationale"])

    def test_prefix_whitespace_and_case_are_tolerated(self):
        row = _from_record(RECORD_SPEED, 7, "validation")
        self.assertEqual(row.choices, {"A": "45", "B": "60", "C": "75", "D": "80", "E": "90"})
        self.assertEqual(row.ground_truth, "D")
        self.assertEqual(row.question, "A train covers 60 km in 45 minutes. What is its speed in km/h?")
        self.assertEqual(row.example_id, 7)
        # The HF "validation" split is the pipeline's "dev" split.
        self.assertEqual(row.split, "dev")

    def test_to_dict_round_trips_through_standard_row(self):
        row = _from_record(RECORD_ICE, 3, "train")
        d = row.to_dict()
        self.assertEqual(d["split"], "train")
        self.assertEqual(StandardRow(**d), row)

    def test_unknown_correct_letter_is_rejected(self):
        self.assertIsNone(_from_record(dict(RECORD_ICE, correct="F"), 0, "test"))
        self.assertIsNone(_from_record(dict(RECORD_ICE, correct=""), 0, "test"))

    def test_missing_question_or_options_is_rejected(self):
        self.assertIsNone(_from_record(dict(RECORD_ICE, question=""), 0, "test"))
        self.assertIsNone(_from_record(dict(RECORD_ICE, options=[]), 0, "test"))


if __name__ == "__main__":
    unittest.main()
