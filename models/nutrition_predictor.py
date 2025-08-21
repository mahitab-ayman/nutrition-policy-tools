"""
Machine Learning Models for Nutrition Gap Prediction
Implements various ML algorithms for predicting nutrition risks and gaps
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutritionPredictor:
    """
    Main class for nutrition gap prediction using machine learning
    """
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_model = None
        self.model_performance = {}
        
        # Initialize base models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize various machine learning models
        """
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # Create ensemble model
        self.models['ensemble'] = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('gb', self.models['gradient_boosting']),
                ('lr', self.models['logistic_regression'])
            ],
            voting='soft'
        )
    
    def prepare_features(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training by handling missing values and scaling
        
        Args:
            features: Feature DataFrame
            target: Target Series
            
        Returns:
            Tuple of (prepared_features, target)
        """
        try:
            # Check if we have enough data
            if len(features) < 10:
                logger.warning(f"Insufficient data for training: only {len(features)} samples")
                return features, target
            
            # Check if target has multiple classes
            unique_targets = target.unique()
            if len(unique_targets) < 2:
                logger.warning(f"Target has only {len(unique_targets)} unique class(es): {unique_targets}")
                return features, target
            
            # Handle missing values
            features_clean = features.fillna(features.mean())
            
            # Feature selection using statistical tests
            if self.feature_selector is None:
                k = min(15, len(features.columns))  # Select top k features
                self.feature_selector = SelectKBest(score_func=f_classif, k=k)
                features_selected = self.feature_selector.fit_transform(features_clean, target)
                self.selected_features = features_clean.columns[self.feature_selector.get_support()].tolist()
                selected_features = self.selected_features
                logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
            else:
                # Use the same features that were selected during training
                if hasattr(self, 'selected_features'):
                    # Ensure we only use the features that were selected during training
                    available_features = [col for col in self.selected_features if col in features_clean.columns]
                    if len(available_features) < len(self.selected_features):
                        logger.warning(f"Some training features missing: {set(self.selected_features) - set(features_clean.columns)}")
                    
                    features_clean = features_clean[available_features]
                    features_selected = self.feature_selector.transform(features_clean)
                    selected_features = available_features
                else:
                    # Fallback: refit the selector
                    features_selected = self.feature_selector.transform(features_clean)
                    selected_features = features_clean.columns[self.feature_selector.get_support()].tolist()
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_selected)
            
            # Create DataFrame with selected and scaled features
            prepared_features = pd.DataFrame(
                features_scaled,
                columns=selected_features,
                index=features.index
            )
            
            return prepared_features, target
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return features, target
    
    def train_models(self, features: pd.DataFrame, target: pd.Series, 
                     test_size: float = 0.2) -> Dict:
        """
        Train all models and evaluate performance
        
        Args:
            features: Feature DataFrame
            target: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results
        """
        try:
            # Prepare features
            prepared_features, target = self.prepare_features(features, target)
            
            # Check if preparation was successful
            if prepared_features.empty or target.empty:
                logger.error("Feature preparation failed - insufficient or invalid data")
                return {}
            
            # Check if we have enough samples for splitting
            if len(prepared_features) < 20:
                logger.error(f"Not enough samples for training: {len(prepared_features)} (need at least 20)")
                return {}
            
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    prepared_features, target, test_size=test_size, random_state=42, stratify=target
                )
            except ValueError as e:
                logger.error(f"Data splitting failed: {str(e)}")
                # Try without stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    prepared_features, target, test_size=test_size, random_state=42
                )
            
            results = {}
            
            # Train and evaluate each model
            for name, model in self.models.items():
                logger.info(f"Training {name} model...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Handle predict_proba safely
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X_test)
                        # Check if proba is 2D (has multiple classes)
                        if proba.ndim == 2 and proba.shape[1] > 1:
                            y_pred_proba = proba[:, 1]
                        elif proba.ndim == 1:
                            # Single class case - use the probability directly
                            y_pred_proba = proba
                        else:
                            y_pred_proba = None
                    except Exception as e:
                        logger.warning(f"Could not get probabilities for {name}: {str(e)}")
                        y_pred_proba = None
                
                # Calculate metrics
                accuracy = (y_pred == y_test).mean()
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(
                        prepared_features.columns, model.feature_importances_
                    ))
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
            self.best_model = results[best_model_name]['model']
            self.model_performance = results
            
            logger.info(f"Best model: {best_model_name} with CV score: {results[best_model_name]['cv_mean']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return {}
    
    def predict_risk(self, features: pd.DataFrame, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict nutrition risk using trained model
        
        Args:
            features: Feature DataFrame
            model_name: Name of specific model to use (if None, uses best model)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            if model_name is None:
                if self.best_model is None:
                    raise ValueError("No trained model available. Please train models first.")
                model = self.best_model
            else:
                if model_name not in self.models:
                    raise ValueError(f"Model {model_name} not found")
                model = self.models[model_name]
            
            # Prepare features for prediction (without refitting)
            prepared_features = self._prepare_features_for_prediction(features)
            
            if prepared_features.empty:
                logger.error("Feature preparation failed for prediction")
                return np.array([]), np.array([])
            
            # Make predictions
            predictions = model.predict(prepared_features)
            
            # Handle predict_proba safely
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(prepared_features)
                    # Check if proba is 2D (has multiple classes)
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        probabilities = proba[:, 1]
                    elif proba.ndim == 1:
                        # Single class case - use the probability directly
                        probabilities = proba
                    else:
                        probabilities = None
                except Exception as e:
                    logger.warning(f"Could not get probabilities for prediction: {str(e)}")
                    probabilities = None
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.array([]), np.array([])
    
    def _prepare_features_for_prediction(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction without refitting selectors/scalers
        """
        try:
            if not hasattr(self, 'selected_features') or self.selected_features is None:
                logger.error("No features selected during training")
                return pd.DataFrame()
            
            # Handle missing values
            features_clean = features.fillna(features.mean())
            
            # Select only the features that were used during training
            available_features = [col for col in self.selected_features if col in features_clean.columns]
            if len(available_features) < len(self.selected_features):
                missing_features = set(self.selected_features) - set(features_clean.columns)
                logger.warning(f"Missing features during prediction: {missing_features}")
                # Fill missing features with zeros
                for col in missing_features:
                    features_clean[col] = 0
            
            # Ensure all training features are present in the correct order
            features_clean = features_clean[self.selected_features]
            
            # The features are already selected, so we don't need to transform them again
            # Just scale them using the fitted scaler
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_clean)
            else:
                features_scaled = features_clean.values
            
            # Create DataFrame with selected and scaled features
            prepared_features = pd.DataFrame(
                features_scaled,
                columns=self.selected_features,
                index=features.index
            )
            
            return prepared_features
            
        except Exception as e:
            logger.error(f"Error preparing features for prediction: {str(e)}")
            return pd.DataFrame()
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.feature_importance:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_importance[model_name].keys()),
            'importance': list(self.feature_importance[model_name].values())
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def simulate_intervention(self, features: pd.DataFrame, 
                            intervention_type: str,
                            intervention_strength: float = 0.2) -> Dict:
        """
        Simulate the impact of nutrition interventions
        
        Args:
            features: Feature DataFrame
            intervention_type: Type of intervention ('supplementation', 'dietary_change', 'infrastructure')
            intervention_strength: Strength of intervention (0.0 to 1.0)
            
        Returns:
            Dictionary with intervention results
        """
        try:
            # Get baseline predictions
            baseline_pred, baseline_prob = self.predict_risk(features)
            
            # Apply intervention effects
            modified_features = features.copy()
            
            if intervention_type == 'supplementation':
                # Improve nutrient adequacy scores
                nutrient_cols = [col for col in features.columns if 'adequacy' in col.lower()]
                for col in nutrient_cols:
                    modified_features[col] = features[col] * (1 + intervention_strength)
                    
            elif intervention_type == 'dietary_change':
                # Improve food security indicators
                food_cols = [col for col in features.columns if 'food' in col.lower()]
                for col in food_cols:
                    modified_features[col] = features[col] * (1 + intervention_strength)
                    
            elif intervention_type == 'infrastructure':
                # Improve market and climate indicators
                infra_cols = [col for col in features.columns if any(x in col.lower() for x in ['market', 'climate', 'rainfall'])]
                for col in infra_cols:
                    modified_features[col] = features[col] * (1 + intervention_strength)
            
            # Get post-intervention predictions
            intervention_pred, intervention_prob = self.predict_risk(modified_features)
            
            # Calculate impact
            risk_reduction = np.array([0])
            risk_reduction_pct = np.array([0])
            
            if baseline_prob is not None and intervention_prob is not None:
                # Ensure both are numpy arrays
                if not isinstance(baseline_prob, np.ndarray):
                    baseline_prob = np.array([baseline_prob])
                if not isinstance(intervention_prob, np.ndarray):
                    intervention_prob = np.array([intervention_prob])
                
                # Avoid division by zero
                risk_reduction = baseline_prob - intervention_prob
                risk_reduction_pct = np.where(baseline_prob > 0, 
                                           (risk_reduction / baseline_prob * 100), 
                                           0)
            
            results = {
                'intervention_type': intervention_type,
                'intervention_strength': intervention_strength,
                'baseline_risk': baseline_prob.mean() if baseline_prob is not None else 0,
                'post_intervention_risk': intervention_prob.mean() if intervention_prob is not None else 0,
                'risk_reduction': risk_reduction.mean() if risk_reduction is not None else 0,
                'risk_reduction_percentage': risk_reduction_pct.mean() if risk_reduction_pct is not None else 0,
                'high_risk_baseline': (baseline_pred == 1).sum(),
                'high_risk_post_intervention': (intervention_pred == 1).sum(),
                'people_benefited': (baseline_pred == 1).sum() - (intervention_pred == 1).sum()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error simulating intervention: {str(e)}")
            return {}
    
    def generate_policy_recommendations(self, features: pd.DataFrame, 
                                      country: str, region: str) -> List[Dict]:
        """
        Generate policy recommendations based on predicted risks
        
        Args:
            features: Feature DataFrame
            country: Country name
            region: Region name
            
        Returns:
            List of policy recommendations
        """
        try:
            # Get risk predictions
            predictions, probabilities = self.predict_risk(features)
            
            # Get feature importance
            importance_df = self.get_feature_importance()
            
            recommendations = []
            
            # High nutrition risk
            if probabilities is not None and len(probabilities) > 0 and probabilities.mean() > 0.7:
                recommendations.append({
                    'priority': 'High',
                    'category': 'Immediate Action',
                    'recommendation': f'Implement emergency nutrition supplementation program in {region}, {country}',
                    'rationale': f'Predicted nutrition risk is {probabilities.mean():.1%}',
                    'estimated_impact': 'High - immediate reduction in malnutrition risk',
                    'cost_estimate': 'Medium to High',
                    'timeline': '1-3 months'
                })
            
            # Moderate risk
            elif probabilities is not None and len(probabilities) > 0 and probabilities.mean() > 0.4:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Preventive Action',
                    'recommendation': f'Develop targeted nutrition education and food security programs in {region}',
                    'rationale': f'Predicted nutrition risk is {probabilities.mean():.1%}',
                    'estimated_impact': 'Medium - gradual improvement in nutrition outcomes',
                    'cost_estimate': 'Low to Medium',
                    'timeline': '3-6 months'
                })
            
            # Low risk
            elif probabilities is not None and len(probabilities) > 0:
                recommendations.append({
                    'priority': 'Low',
                    'category': 'Monitoring',
                    'recommendation': f'Continue monitoring nutrition indicators in {region}',
                    'rationale': f'Predicted nutrition risk is {probabilities.mean():.1%}',
                    'estimated_impact': 'Low - maintain current good nutrition status',
                    'cost_estimate': 'Low',
                    'timeline': 'Ongoing'
                })
            
            # Add specific recommendations based on feature importance
            top_features = importance_df.head(3)
            for _, row in top_features.iterrows():
                feature = row['feature']
                importance = row['importance']
                
                if 'adequacy' in feature.lower():
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Nutrient-Specific',
                        'recommendation': f'Focus on improving {feature.replace("_", " ")} in {region}',
                        'rationale': f'This factor has high importance ({importance:.3f}) in risk prediction',
                        'estimated_impact': 'Medium - targeted improvement in specific nutrient',
                        'cost_estimate': 'Low to Medium',
                        'timeline': '3-12 months'
                    })
            
            # Fallback case when no probabilities available
            if probabilities is None or len(probabilities) == 0:
                recommendations.append({
                    'priority': 'Medium',
                    'category': 'Data Analysis',
                    'recommendation': f'Improve data collection and monitoring in {region}, {country}',
                    'rationale': 'Insufficient data for risk prediction - need better monitoring',
                    'estimated_impact': 'Medium - improved data quality for future analysis',
                    'cost_estimate': 'Low to Medium',
                    'timeline': '6-12 months'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating policy recommendations: {str(e)}")
            return []
    
    def save_model(self, filepath: str, model_name: str = 'best_model'):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
            model_name: Name of the model to save
        """
        try:
            if model_name == 'best_model':
                model = self.best_model
            else:
                model = self.models.get(model_name)
            
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            
            # Save model and related objects
            model_data = {
                'model': model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Args:
            filepath: Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.feature_importance = model_data['feature_importance']
            self.model_performance = model_data['model_performance']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def plot_feature_importance(self, model_name: str = 'random_forest', top_n: int = 10):
        """
        Plot feature importance for a specific model
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to show
        """
        importance_df = self.get_feature_importance(model_name)
        
        if importance_df.empty:
            logger.warning(f"No feature importance data for {model_name}")
            return
        
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name.replace("_", " ").title()}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self):
        """
        Plot ROC curves for all trained models
        """
        plt.figure(figsize=(10, 8))
        
        for name, results in self.model_performance.items():
            if results['probabilities'] is not None and len(results['probabilities']) > 0:
                try:
                    fpr, tpr, _ = roc_curve(
                        [1 if x == 1 else 0 for x in results['predictions']], 
                        results['probabilities']
                    )
                    auc = results['auc']
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
                except Exception as e:
                    logger.warning(f"Could not plot ROC curve for {name}: {str(e)}")
                    continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # This would be used with actual data from the data processor
    print("Nutrition Predictor Module")
    print("This module provides ML models for nutrition gap prediction")
    print("Use with NutritionDataProcessor to get training data")
