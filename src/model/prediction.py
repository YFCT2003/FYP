"""
Prediction Module for ImmunBert
This module handles the final prediction of immunogenicity based on scoring results
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImmunogenicityPredictor:
    """
    Class to predict immunogenicity based on sequence scoring and interface features
    """
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize the ImmunogenicityPredictor
        
        Args:
            model_type: Type of classifier to use ('logistic', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
        # Initialize model
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Initialized {model_type} predictor")
    
    def prepare_features(self, scoring_results: Dict, interface_features: Optional[Dict] = None) -> np.ndarray:
        """
        Prepare features for prediction from scoring results and interface features
        
        Args:
            scoring_results: Dictionary with scoring metrics from ProteinMPNN
            interface_features: Optional dictionary with interface features from MaSIF
            
        Returns:
            np.ndarray: Feature vector for prediction
        """
        features = []
        feature_names = []
        
        # Add scoring features
        if 'log_likelihood' in scoring_results and scoring_results['log_likelihood'] is not None:
            features.append(scoring_results['log_likelihood'])
            feature_names.append('log_likelihood')
            
        if 'perplexity' in scoring_results and scoring_results['perplexity'] is not None:
            features.append(scoring_results['perplexity'])
            feature_names.append('perplexity')
            
        if 'confidence' in scoring_results and scoring_results['confidence'] is not None:
            features.append(scoring_results['confidence'])
            feature_names.append('confidence')
            
        # Add sequence length
        if 'sequence_length' in scoring_results:
            features.append(scoring_results['sequence_length'])
            feature_names.append('sequence_length')
        
        # Add interface features if provided
        if interface_features:
            # Add geometric features statistics
            if 'geometric' in interface_features:
                geom_features = interface_features['geometric']
                for key, values in geom_features.items():
                    if isinstance(values, list) and len(values) > 0:
                        features.extend([
                            np.mean(values),  # Mean
                            np.std(values),   # Standard deviation
                            np.max(values),   # Maximum
                            np.min(values)    # Minimum
                        ])
                        feature_names.extend([
                            f'{key}_mean', f'{key}_std', f'{key}_max', f'{key}_min'
                        ])
            
            # Add chemical features statistics
            if 'chemical' in interface_features:
                chem_features = interface_features['chemical']
                for key, values in chem_features.items():
                    if isinstance(values, list) and len(values) > 0:
                        features.extend([
                            np.mean(values),  # Mean
                            np.std(values),   # Standard deviation
                            np.max(values),   # Maximum
                            np.min(values)    # Minimum
                        ])
                        feature_names.extend([
                            f'{key}_mean', f'{key}_std', f'{key}_max', f'{key}_min'
                        ])
        
        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)
    
    def predict(self, features: np.ndarray, threshold: float = 0.5) -> Dict[str, Union[float, bool]]:
        """
        Predict immunogenicity based on features
        
        Args:
            features: Feature vector for prediction
            threshold: Probability threshold for binary classification
            
        Returns:
            Dict: Prediction results including probability and binary classification
        """
        if not self.is_trained:
            logger.warning("Model is not trained yet. Returning mock predictions.")
            # Generate mock prediction for demonstration
            probability = np.random.uniform(0, 1)
            prediction = probability >= threshold
            return {
                'probability': probability,
                'prediction': prediction,
                'threshold': threshold
            }
        
        # Get prediction probability
        try:
            probability = self.model.predict_proba(features)[0][1]  # Probability of positive class
            prediction = probability >= threshold
            
            return {
                'probability': float(probability),
                'prediction': bool(prediction),
                'threshold': threshold
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def train(self, training_data: List[Dict], labels: List[int]) -> Dict[str, float]:
        """
        Train the predictor with training data
        
        Args:
            training_data: List of dictionaries with scoring results and features
            labels: List of binary labels (0 for non-immunogenic, 1 for immunogenic)
            
        Returns:
            Dict: Training metrics
        """
        logger.info(f"Training predictor with {len(training_data)} samples")
        
        # Prepare feature matrix
        X_list = []
        for data in training_data:
            scoring_results = data.get('scoring', {})
            interface_features = data.get('interface', {})
            features = self.prepare_features(scoring_results, interface_features)
            X_list.append(features.flatten())
        
        X = np.array(X_list)
        y = np.array(labels)
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0
        }
        
        logger.info(f"Training completed. Accuracy: {metrics['accuracy']:.3f}")
        return metrics
    
    def evaluate(self, test_data: List[Dict], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the predictor on test data
        
        Args:
            test_data: List of dictionaries with scoring results and features
            labels: List of binary labels
            
        Returns:
            Dict: Evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
            
        logger.info(f"Evaluating predictor with {len(test_data)} samples")
        
        # Prepare feature matrix
        X_list = []
        for data in test_data:
            scoring_results = data.get('scoring', {})
            interface_features = data.get('interface', {})
            features = self.prepare_features(scoring_results, interface_features)
            X_list.append(features.flatten())
        
        X = np.array(X_list)
        y = np.array(labels)
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0
        }
        
        logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.3f}")
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
            
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        import joblib
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from: {filepath}")


def main():
    """Main function for testing the prediction module"""
    # Example usage
    predictor = ImmunogenicityPredictor(model_type='logistic')
    
    # Example scoring results (mock data)
    scoring_results = {
        'log_likelihood': -25.3,
        'perplexity': 8.7,
        'confidence': 0.85,
        'sequence_length': 14
    }
    
    # Example interface features (mock data)
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
    
    try:
        # Prepare features
        features = predictor.prepare_features(scoring_results, interface_features)
        print(f"Prepared feature vector with {features.shape[1]} features")
        
        # Make prediction (will be mock since model is not trained)
        result = predictor.predict(features)
        print(f"Prediction probability: {result['probability']:.3f}")
        print(f"Binary prediction: {result['prediction']}")
        print(f"Threshold: {result['threshold']}")
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")


if __name__ == "__main__":
    main()