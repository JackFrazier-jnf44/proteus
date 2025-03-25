"""Core ensemble prediction functionality."""

import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """Enumeration of available ensemble methods."""
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"
    VOTING = "voting"
    BAGGING = "bagging"
    BOOSTING = "boosting"

@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictions."""
    method: EnsembleMethod
    weights: Optional[Dict[str, float]] = None
    meta_model: Optional[str] = None  # For stacking method
    confidence_threshold: float = 0.7
    n_estimators: int = 100  # For bagging and boosting
    learning_rate: float = 0.1  # For boosting
    n_folds: int = 5  # For stacking

class EnsemblePredictor:
    """
    Class for combining predictions from multiple models using various ensemble methods.
    """
    
    def __init__(self, config: EnsembleConfig):
        """
        Initialize the ensemble predictor.
        
        Args:
            config: Ensemble configuration specifying the method and parameters
        """
        self.config = config
        self.model_weights = config.weights or {}
        self.meta_models = {}
        logger.info(f"Initialized ensemble predictor with method: {config.method.value}")
    
    def combine_predictions(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        confidence_scores: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Combine predictions from multiple models using the specified ensemble method.
        
        Args:
            predictions: Dictionary mapping model names to their predictions
            confidence_scores: Optional dictionary mapping model names to confidence scores
            
        Returns:
            Dictionary containing combined predictions
        """
        try:
            if self.config.method == EnsembleMethod.AVERAGE:
                return self._average_ensemble(predictions)
            elif self.config.method == EnsembleMethod.WEIGHTED_AVERAGE:
                return self._weighted_average_ensemble(predictions, confidence_scores)
            elif self.config.method == EnsembleMethod.STACKING:
                return self._stacking_ensemble(predictions)
            elif self.config.method == EnsembleMethod.VOTING:
                return self._voting_ensemble(predictions, confidence_scores)
            elif self.config.method == EnsembleMethod.BAGGING:
                return self._bagging_ensemble(predictions)
            elif self.config.method == EnsembleMethod.BOOSTING:
                return self._boosting_ensemble(predictions)
            else:
                raise ValueError(f"Unsupported ensemble method: {self.config.method}")
        except Exception as e:
            logger.error(f"Failed to combine predictions: {str(e)}")
            raise
    
    def _average_ensemble(self, predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine predictions using simple averaging."""
        combined = {}
        for key in predictions[list(predictions.keys())[0]].keys():
            combined[key] = np.mean([pred[key] for pred in predictions.values()], axis=0)
        return combined
    
    def _weighted_average_ensemble(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        confidence_scores: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Combine predictions using weighted averaging based on confidence scores or provided weights."""
        combined = {}
        weights = self._get_weights(predictions, confidence_scores)
        
        for key in predictions[list(predictions.keys())[0]].keys():
            combined[key] = np.average(
                [pred[key] for pred in predictions.values()],
                weights=[weights[model] for model in predictions.keys()],
                axis=0
            )
        return combined
    
    def _stacking_ensemble(self, predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine predictions using stacking with a meta-model."""
        combined = {}
        
        # Prepare features for meta-model
        for key in predictions[list(predictions.keys())[0]].keys():
            # Stack predictions from all models
            X = np.stack([pred[key] for pred in predictions.values()], axis=1)
            
            # Initialize meta-model if not exists
            if key not in self.meta_models:
                self.meta_models[key] = RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=42
                )
            
            # Train meta-model using k-fold cross-validation
            kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
            meta_predictions = np.zeros_like(X[:, 0])
            
            for train_idx, val_idx in kf.split(X):
                # Train on training fold
                self.meta_models[key].fit(X[train_idx], np.mean(X[train_idx], axis=1))
                # Predict on validation fold
                meta_predictions[val_idx] = self.meta_models[key].predict(X[val_idx])
            
            combined[key] = meta_predictions
        
        return combined
    
    def _bagging_ensemble(self, predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine predictions using bagging with random forests."""
        combined = {}
        
        for key in predictions[list(predictions.keys())[0]].keys():
            # Stack predictions from all models
            X = np.stack([pred[key] for pred in predictions.values()], axis=1)
            
            # Initialize bagging model if not exists
            if key not in self.meta_models:
                self.meta_models[key] = RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    random_state=42
                )
            
            # Train bagging model
            self.meta_models[key].fit(X, np.mean(X, axis=1))
            combined[key] = self.meta_models[key].predict(X)
        
        return combined
    
    def _boosting_ensemble(self, predictions: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Combine predictions using gradient boosting."""
        combined = {}
        
        for key in predictions[list(predictions.keys())[0]].keys():
            # Stack predictions from all models
            X = np.stack([pred[key] for pred in predictions.values()], axis=1)
            
            # Initialize boosting model if not exists
            if key not in self.meta_models:
                self.meta_models[key] = GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    random_state=42
                )
            
            # Train boosting model
            self.meta_models[key].fit(X, np.mean(X, axis=1))
            combined[key] = self.meta_models[key].predict(X)
        
        return combined
    
    def _voting_ensemble(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        confidence_scores: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Combine predictions using majority voting with confidence threshold."""
        combined = {}
        weights = self._get_weights(predictions, confidence_scores)
        
        for key in predictions[list(predictions.keys())[0]].keys():
            # Get predictions for this key
            preds = np.stack([pred[key] for pred in predictions.values()], axis=0)
            
            # Apply confidence threshold
            if confidence_scores:
                mask = np.array([scores > self.config.confidence_threshold 
                                for scores in confidence_scores.values()])
                preds = np.where(mask, preds, np.nan)
            
            # Use weighted majority voting
            combined[key] = np.nanmean(preds, axis=0, weights=list(weights.values()))
        
        return combined
    
    def _get_weights(
        self,
        predictions: Dict[str, Dict[str, np.ndarray]],
        confidence_scores: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """Get weights for ensemble combination."""
        if self.model_weights:
            return self.model_weights
        
        if confidence_scores:
            # Use normalized confidence scores as weights
            weights = {
                model: np.mean(scores) 
                for model, scores in confidence_scores.items()
            }
            total = sum(weights.values())
            return {model: w/total for model, w in weights.items()}
        
        # Default to uniform weights
        return {model: 1.0/len(predictions) for model in predictions.keys()} 