import unittest
from graffitiai.optimist import Optimist


class TestOptimist(unittest.TestCase):

    def setUp(self):
        """Set up an Optimist instance with sample data for testing."""
        self.optimist = Optimist()
        self.optimist.load_sample_3_regular_polytope_data()

    def test_load_sample_data(self):
        """Test loading the sample data."""
        self.assertIsNotNone(self.optimist.knowledge_table, "Knowledge table should not be None after loading sample data.")
        self.assertGreater(len(self.optimist.knowledge_table), 0, "Knowledge table should have rows.")

    def test_get_possible_invariants(self):
        """Test identifying numerical columns."""
        invariants = self.optimist.get_possible_invariants()
        self.assertIn("independence_number", invariants, "'independence_number' should be recognized as a numerical invariant.")
        self.assertIn("n", invariants, "'n' should be recognized as a numerical invariant.")

    def test_get_possible_hypotheses(self):
        """Test identifying boolean columns."""
        hypotheses = self.optimist.get_possible_hypotheses()
        self.assertIn("cubic_polytope", hypotheses, "'cubic_polytope' should be recognized as a boolean hypothesis.")
if __name__ == "__main__":
    unittest.main()
