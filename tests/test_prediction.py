"""
Unit tests for the prediction module
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.prediction import ImmunogenicityPredictor


class TestImmunogenicityPredictor(unittest.TestCase):
    """Test cases for the ImmunogenicityPredictor class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.predictor = ImmunogenicityPredictor()
    
    def test_initialization(self):
        """Test ImmunogenicityPredictor initialization"""
        self.assertIsInstance(self.predictor, ImmunogenicityPredictor)
        
        # Test different model types
        lr_predictor = ImmunogenicityPredictor('logistic')
        rf_predictor = ImmunogenicityPredictor('random_forest')
        
        self.assertIsNotNone(lr_predictor.model)
        self.assertIsNotNone(rf_predictor.model)
        
        # Test unsupported model type
        with self.assertRaises(ValueError):
            ImmunogenicityPredictor('unsupported_model')
    
    def test_prepare_features(self):
        """Test feature preparation"""
        # Mock scoring results
        scoring_results = {
            'log_likelihood': -25.3,
            'perplexity': 8.7,
            'confidence': 0.85,
            'sequence_length': 14
        }
        
        # Mock interface features
        interface_features = {
            'geometric': {
                'curvature': [0.1, 0.2, 0.3, 0.4, 0.5],
                'shape_index': [-0.5, -0.4, -0.3, -0.2, -0.1]
            },
            'chemical': {
                'hydrophobicity': [0.8, 0.7, 0.6, 0.5, 0.4],
                'charge': [1.0, 0.8, 0.6, 0.4, 0.2]
            }
        }
        
        # Prepare features
        features = self.predictor.prepare_features(scoring_results, interface_features)
        
        # Check feature array
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[0], 1)
        self.assertGreater(features.shape[1], 0)
        
        # Check feature names were stored
        self.assertIsNotNone(self.predictor.feature_names)
        self.assertIsInstance(self.predictor.feature_names, list)
        self.assertGreater(len(self.predictor.feature_names), 0)
    
    def test_predict(self):
        """Test prediction functionality"""
        # Mock features
        features = np.random.rand(1, 20)
        
        # Test prediction (will be mock since model is not trained)
        result = self.predictor.predict(features)
        
        # Check result structure
        self.assertIn('probability', result)
        self.assertIn('prediction', result)
        self.assertIn('threshold', result)
        
        # Check value types
        self.assertIsInstance(result['probability'], float)
        self.assertIsInstance(result['prediction'], bool)
        self.assertIsInstance(result['threshold'], float)
        
        # Check value constraints
        self.assertGreaterEqual(result['probability'], 0.0)
        self.assertLessEqual(result['probability'], 1.0)
        self.assertEqual(result['threshold'], 0.5)  # Default threshold
    
    def test_train_and_evaluate(self):
        """Test training and evaluation functionality"""
        # Mock training data
        training_data = [
            {
                'scoring': {
                    'log_likelihood': -20.0,
                    'perplexity': 7.5,
                    'confidence': 0.9,
                    'sequence_length': 14
                },
                'interface': {
                    'geometric': {
                        'curvature': [0.1, 0.2, 0.3],
                        'shape_index': [-0.5, -0.4, -0.3]
                    },
                    'chemical': {
                        'hydrophobicity': [0.8, 0.7, 0.6],
                        'charge': [1.0, 0.8, 0.6]
                    }
                }
            },
            {
                'scoring': {
                    'log_likelihood': -30.0,
                    'perplexity': 12.0,
                    'confidence': 0.6,
                    'sequence_length': 12
                },
                'interface': {
                    'geometric': {
                        'curvature': [0.2, 0.3, 0.4],
                        'shape_index': [-0.4, -0.3, -0.2]
                    },
                    'chemical': {
                        'hydrophobicity': [0.6, 0.5, 0.4],
                        'charge': [0.8, 0.6, 0.4]
                    }
                }
            }
        ]
        
        # Mock labels (1 for immunogenic, 0 for non-immunogenic)
        labels = [1, 0]
        
        # Test training
        metrics = self.predictor.train(training_data, labels)
        
        # Check metrics structure
        expected_metrics = {'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'}
        self.assertEqual(set(metrics.keys()), expected_metrics)
        
        # Check that model is now trained
        self.assertTrue(self.predictor.is_trained)
        
        # Test evaluation with same data
        eval_metrics = self.predictor.evaluate(training_data, labels)
        self.assertEqual(set(eval_metrics.keys()), expected_metrics)


if __name__ == '__main__':
    unittest.main()