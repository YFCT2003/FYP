"""
Unit tests for the interface encoding module
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from interface.encoding import InterfaceEncoder


class TestInterfaceEncoder(unittest.TestCase):
    """Test cases for the InterfaceEncoder class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.encoder = InterfaceEncoder()
    
    def test_initialization(self):
        """Test InterfaceEncoder initialization"""
        self.assertIsInstance(self.encoder, InterfaceEncoder)
    
    def test_generate_mock_features(self):
        """Test generation of mock features"""
        features = self.encoder._generate_mock_features()
        
        # Check that features dict has the expected keys
        self.assertIn('geometric', features)
        self.assertIn('chemical', features)
        self.assertIn('feature_vector', features)
        
        # Check geometric features
        geom = features['geometric']
        self.assertIn('curvature', geom)
        self.assertIn('shape_index', geom)
        self.assertIn('mean_curvature', geom)
        
        # Check chemical features
        chem = features['chemical']
        self.assertIn('hydrophobicity', chem)
        self.assertIn('charge', chem)
        self.assertIn('electrostatic_potential', chem)
        
        # Check feature vector
        feature_vec = features['feature_vector']
        self.assertIsInstance(feature_vec, list)
        self.assertEqual(len(feature_vec), 500)
    
    def test_save_features(self):
        """Test saving features to file"""
        import tempfile
        import json
        
        # Generate mock features
        features = self.encoder._generate_mock_features()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save features
            self.encoder.save_features(features, tmp_path)
            
            # Check that file was created and contains valid JSON
            self.assertTrue(os.path.exists(tmp_path))
            
            with open(tmp_path, 'r') as f:
                loaded_features = json.load(f)
            
            # Check that loaded features match original
            self.assertEqual(loaded_features.keys(), features.keys())
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()