#!/usr/bin/env python3
"""
Unit tests for the proximity analysis script.
"""

import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import proximity_analysis
from utils.geo_utils import Point, categorize_elevation

class TestProximityAnalysis(unittest.TestCase):
    """Tests for the proximity_analysis module."""
    
    @patch('proximity_analysis.load_erratic_by_id')
    def test_calculate_proximity_valid_erratic(self, mock_load_erratic):
        """Test calculate_proximity with a valid erratic ID."""
        # Arrange
        erratic_id = 1
        mock_erratic = {
            'id': erratic_id,
            'name': 'Test Erratic',
            'longitude': -70.6619,
            'latitude': 41.958,
            'elevation': 50
        }
        mock_load_erratic.return_value = mock_erratic
        
        # Act
        result = proximity_analysis.calculate_proximity(erratic_id)
        
        # Assert
        self.assertEqual(result['erratic_id'], erratic_id)
        self.assertEqual(result['erratic_name'], 'Test Erratic')
        self.assertEqual(result['location']['longitude'], -70.6619)
        self.assertEqual(result['location']['latitude'], 41.958)
        self.assertIn('proximity_analysis', result)
        self.assertEqual(result['proximity_analysis']['elevation_category'], 'lowland')
    
    @patch('proximity_analysis.load_erratic_by_id')
    def test_calculate_proximity_invalid_erratic(self, mock_load_erratic):
        """Test calculate_proximity with an invalid erratic ID."""
        # Arrange
        erratic_id = 999
        mock_load_erratic.return_value = None
        
        # Act
        result = proximity_analysis.calculate_proximity(erratic_id)
        
        # Assert
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Erratic with ID 999 not found')
    
    @patch('proximity_analysis.load_erratic_by_id')
    def test_calculate_proximity_with_feature_layers(self, mock_load_erratic):
        """Test calculate_proximity with specific feature layers."""
        # Arrange
        erratic_id = 1
        feature_layers = ['water_bodies', 'settlements']
        mock_erratic = {
            'id': erratic_id,
            'name': 'Test Erratic',
            'longitude': -70.6619,
            'latitude': 41.958,
            'elevation': 50
        }
        mock_load_erratic.return_value = mock_erratic
        
        # Act
        result = proximity_analysis.calculate_proximity(erratic_id, feature_layers)
        
        # Assert
        self.assertEqual(result['erratic_id'], erratic_id)
        proximity = result['proximity_analysis']
        self.assertIn('nearest_water_body_dist', proximity)
        self.assertIn('nearest_settlement_dist', proximity)
    
    def test_categorize_elevation(self):
        """Test elevation categorization."""
        # Test cases
        test_cases = [
            (0, 'lowland'),
            (50, 'lowland'),
            (99, 'lowland'),
            (100, 'mid-elevation'),
            (300, 'mid-elevation'),
            (499, 'mid-elevation'),
            (500, 'highland'),
            (800, 'highland'),
            (999, 'highland'),
            (1000, 'mountain'),
            (2000, 'mountain')
        ]
        
        # Test each case
        for elevation, expected_category in test_cases:
            with self.subTest(elevation=elevation):
                category = categorize_elevation(elevation)
                self.assertEqual(category, expected_category)

class TestProximityAnalysisCommandLine(unittest.TestCase):
    """Tests for the command line interface."""
    
    @patch('proximity_analysis.calculate_proximity')
    @patch('proximity_analysis.update_erratic_analysis_data')
    @patch('proximity_analysis.json_to_file')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_with_valid_erratic(self, mock_parse_args, mock_json_to_file, 
                                    mock_update_db, mock_calculate_proximity):
        """Test main function with a valid erratic ID."""
        # Arrange
        mock_args = MagicMock()
        mock_args.erratic_id = 1
        mock_args.features = ['water_bodies', 'settlements']
        mock_args.update_db = True
        mock_args.output = 'test_output.json'
        mock_parse_args.return_value = mock_args
        
        mock_result = {
            'erratic_id': 1,
            'erratic_name': 'Test Erratic',
            'proximity_analysis': {
                'elevation_category': 'lowland',
                'nearest_water_body_dist': 1200
            }
        }
        mock_calculate_proximity.return_value = mock_result
        mock_update_db.return_value = True
        
        # Act
        exit_code = proximity_analysis.main()
        
        # Assert
        self.assertEqual(exit_code, 0)
        mock_calculate_proximity.assert_called_once_with(1, ['water_bodies', 'settlements'])
        mock_update_db.assert_called_once_with(1, mock_result['proximity_analysis'])
        mock_json_to_file.assert_called_once_with(mock_result, 'test_output.json')
    
    @patch('proximity_analysis.calculate_proximity')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_with_error(self, mock_parse_args, mock_calculate_proximity):
        """Test main function with an erratic that returns an error."""
        # Arrange
        mock_args = MagicMock()
        mock_args.erratic_id = 999
        mock_args.features = None
        mock_args.update_db = False
        mock_args.output = None
        mock_parse_args.return_value = mock_args
        
        mock_result = {'error': 'Erratic with ID 999 not found'}
        mock_calculate_proximity.return_value = mock_result
        
        # Act
        exit_code = proximity_analysis.main()
        
        # Assert
        self.assertEqual(exit_code, 1)
        mock_calculate_proximity.assert_called_once_with(999, None)

if __name__ == '__main__':
    unittest.main() 