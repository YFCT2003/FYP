"""
Unit tests for the structure modeling module
"""

import unittest
import tempfile
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from structure.modeling import StructureModeler


class TestStructureModeler(unittest.TestCase):
    """Test cases for the StructureModeler class"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.modeler = StructureModeler()
    
    def test_initialization(self):
        """Test StructureModeler initialization"""
        self.assertIsInstance(self.modeler, StructureModeler)
    
    def test_build_complex_structure(self):
        """Test building complex structure (mock test)"""
        # This is a mock test since we don't have actual PANDORA installed
        pmhc_seq = "SIINFEKL.A.B"
        cdr3b_seq = "CASSLGQGDNIQYF"
        
        # Test that the method can be called without error
        # In a real test, we would check the output structure
        try:
            result_path = self.modeler.build_complex_structure(pmhc_seq, cdr3b_seq)
            self.assertIsInstance(result_path, str)
        except RuntimeError as e:
            # Expected when PANDORA is not available
            self.assertIn("not available", str(e))
    
    def test_validate_structure(self):
        """Test structure validation (mock test)"""
        # Test with non-existent file
        fake_path = "fake_structure.pdb"
        result = self.modeler.validate_structure(fake_path)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()