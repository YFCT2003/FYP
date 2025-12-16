"""
Unit tests for the inverse folding scoring module
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.scoring import InverseFolder


class TestInverseFolder(unittest.TestCase):
    """Test cases for the InverseFolder class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.scorer = InverseFolder()
    
    def test_initialization(self):
        """Test InverseFolder initialization"""
        self.assertIsInstance(self.scorer, InverseFolder)
    
    def test_generate_mock_scores(self):
        """Test generation of mock scores"""
        sequence = "CASSLGQGDNIQYF"
        scores = self.scorer._generate_mock_scores(sequence)
        
        # Check that scores dict has the expected keys
        expected_keys = {'log_likelihood', 'perplexity', 'confidence', 'sequence_length'}
        self.assertEqual(set(scores.keys()), expected_keys)
        
        # Check value types and constraints
        self.assertIsInstance(scores['log_likelihood'], float)
        self.assertIsInstance(scores['perplexity'], float)
        self.assertIsInstance(scores['confidence'], float)
        self.assertIsInstance(scores['sequence_length'], int)
        
        # Check value constraints
        self.assertLess(scores['log_likelihood'], 0)  # Log-likelihood should be negative
        self.assertGreater(scores['perplexity'], 0)   # Perplexity should be positive
        self.assertGreaterEqual(scores['confidence'], 0)  # Confidence should be non-negative
        self.assertLessEqual(scores['confidence'], 1)     # Confidence should be <= 1
        self.assertEqual(scores['sequence_length'], len(sequence))
    
    def test_batch_score_sequences(self):
        """Test batch scoring of sequences"""
        sequences = ["CASSLGQGDNIQYF", "CASSTGQGDNIQYF", "CASSLGYDNIQYF"]
        
        # Test batch scoring (mock implementation)
        scores_list = self.scorer.batch_score_sequences("fake_structure.pdb", sequences)
        
        # Check that we get results for all sequences
        self.assertEqual(len(scores_list), len(sequences))
        
        # Check each result
        for i, scores in enumerate(scores_list):
            expected_keys = {'log_likelihood', 'perplexity', 'confidence', 'sequence_length'}
            self.assertEqual(set(scores.keys()), expected_keys)
            self.assertEqual(scores['sequence_length'], len(sequences[i]))


if __name__ == '__main__':
    unittest.main()