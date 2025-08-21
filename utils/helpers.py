"""
Utility Helper Functions for Nutrition Policy Tool
Provides common functionality used across the application
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Class for validating nutrition data quality and consistency
    """
    
    @staticmethod
    def validate_nutrition_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate nutrition data for quality issues
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'total_records': len(data),
            'missing_values': {},
            'outliers': {},
            'data_quality_score': 0.0,
            'issues': []
        }
        
        try:
            # Check for missing values
            missing_counts = data.isnull().sum()
            validation_results['missing_values'] = missing_counts[missing_counts > 0].to_dict()
            
            # Check for outliers in numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            outlier_counts = {}
            
            for col in numerical_cols:
                if col in data.columns:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    if outliers > 0:
                        outlier_counts[col] = outliers
            
            validation_results['outliers'] = outlier_counts
            
            # Calculate data quality score
            total_issues = len(validation_results['missing_values']) + len(validation_results['outliers'])
            max_issues = len(data.columns) * 0.1  # Allow 10% issues
            
            if total_issues > 0:
                validation_results['data_quality_score'] = max(0, 1 - (total_issues / max_issues))
                validation_results['is_valid'] = validation_results['data_quality_score'] > 0.7
            
            # Generate issue descriptions
            if validation_results['missing_values']:
                validation_results['issues'].append(
                    f"Missing values detected in {len(validation_results['missing_values'])} columns"
                )
            
            if validation_results['outliers']:
                validation_results['issues'].append(
                    f"Outliers detected in {len(validation_results['outliers'])} columns"
                )
            
            if not validation_results['issues']:
                validation_results['issues'].append("No data quality issues detected")
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def clean_nutrition_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean nutrition data by handling missing values and outliers
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        try:
            # Handle missing values
            numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if cleaned_data[col].isnull().sum() > 0:
                    # Use median for numerical columns
                    cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
            
            # Handle outliers using IQR method
            for col in numerical_cols:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers at bounds
                cleaned_data[col] = cleaned_data[col].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info("Data cleaning completed successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
        
        return cleaned_data

class DataExporter:
    """
    Class for exporting data and results in various formats
    """
    
    @staticmethod
    def export_to_csv(data: pd.DataFrame, filename: str) -> str:
        """
        Export data to CSV format
        
        Args:
            data: DataFrame to export
            filename: Output filename
            
        Returns:
            File path of exported data
        """
        try:
            filepath = f"{filename}.csv"
            data.to_csv(filepath, index=False)
            logger.info(f"Data exported to CSV: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return ""
    
    @staticmethod
    def export_to_excel(data: pd.DataFrame, filename: str, sheet_name: str = "Data") -> str:
        """
        Export data to Excel format
        
        Args:
            data: DataFrame to export
            filename: Output filename
            sheet_name: Excel sheet name
            
        Returns:
            File path of exported data
        """
        try:
            filepath = f"{filename}.xlsx"
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Data exported to Excel: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            return ""
    
    @staticmethod
    def export_to_json(data: pd.DataFrame, filename: str) -> str:
        """
        Export data to JSON format
        
        Args:
            data: DataFrame to export
            filename: Output filename
            
        Returns:
            File path of exported data
        """
        try:
            filepath = f"{filename}.json"
            data.to_json(filepath, orient='records', indent=2)
            logger.info(f"Data exported to JSON: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            return ""
    
    @staticmethod
    def export_recommendations_to_pdf(recommendations: List[Dict], filename: str) -> str:
        """
        Export policy recommendations to PDF format
        
        Args:
            recommendations: List of recommendation dictionaries
            filename: Output filename
            
        Returns:
            File path of exported PDF
        """
        try:
            # This would integrate with a PDF library like reportlab or fpdf
            # For now, we'll create a simple text representation
            filepath = f"{filename}.txt"
            
            with open(filepath, 'w') as f:
                f.write("NUTRITION POLICY RECOMMENDATIONS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"Recommendation {i}\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Priority: {rec.get('priority', 'N/A')}\n")
                    f.write(f"Category: {rec.get('category', 'N/A')}\n")
                    f.write(f"Recommendation: {rec.get('recommendation', 'N/A')}\n")
                    f.write(f"Rationale: {rec.get('rationale', 'N/A')}\n")
                    f.write(f"Impact: {rec.get('estimated_impact', 'N/A')}\n")
                    f.write(f"Cost: {rec.get('cost_estimate', 'N/A')}\n")
                    f.write(f"Timeline: {rec.get('timeline', 'N/A')}\n\n")
            
            logger.info(f"Recommendations exported to text: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting recommendations: {str(e)}")
            return ""

class DataAnalyzer:
    """
    Class for additional data analysis functions
    """
    
    @staticmethod
    def calculate_trends(data: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """
        Calculate trends in time series data
        
        Args:
            data: DataFrame with time series data
            date_col: Name of date column
            value_col: Name of value column to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Ensure date column is datetime
            data[date_col] = pd.to_datetime(data[date_col])
            
            # Sort by date
            data_sorted = data.sort_values(date_col)
            
            # Calculate simple linear trend
            x = np.arange(len(data_sorted))
            y = data_sorted[value_col].values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}
            
            # Linear regression
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            
            # Calculate R-squared
            y_pred = slope * x_clean + intercept
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction
            if slope > 0.01:
                trend = 'increasing'
            elif slope < -0.01:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'slope': slope,
                'r_squared': r_squared,
                'change_per_period': slope,
                'total_change': slope * len(data_sorted)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trends: {str(e)}")
            return {'trend': 'error', 'slope': 0, 'r_squared': 0}
    
    @staticmethod
    def detect_seasonality(data: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """
        Detect seasonality in time series data
        
        Args:
            data: DataFrame with time series data
            date_col: Name of date column
            value_col: Name of value column to analyze
            
        Returns:
            Dictionary with seasonality analysis results
        """
        try:
            # Ensure date column is datetime
            data[date_col] = pd.to_datetime(data[date_col])
            
            # Extract month and calculate monthly averages
            data['month'] = data[date_col].dt.month
            monthly_avg = data.groupby('month')[value_col].mean()
            
            # Calculate seasonality strength
            overall_mean = monthly_avg.mean()
            seasonal_variance = monthly_avg.var()
            total_variance = data[value_col].var()
            
            seasonality_strength = seasonal_variance / total_variance if total_variance > 0 else 0
            
            # Find peak and trough months
            peak_month = monthly_avg.idxmax()
            trough_month = monthly_avg.idxmin()
            
            return {
                'has_seasonality': seasonality_strength > 0.1,
                'seasonality_strength': seasonality_strength,
                'peak_month': peak_month,
                'trough_month': trough_month,
                'peak_value': monthly_avg[peak_month],
                'trough_value': monthly_avg[trough_month],
                'monthly_pattern': monthly_avg.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {str(e)}")
            return {'has_seasonality': False, 'seasonality_strength': 0}

class RiskCalculator:
    """
    Class for calculating various risk metrics
    """
    
    @staticmethod
    def calculate_population_at_risk(data: pd.DataFrame, risk_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Calculate population at risk based on risk scores
        
        Args:
            data: DataFrame with population and risk data
            risk_threshold: Threshold for high risk classification
            
        Returns:
            Dictionary with risk population statistics
        """
        try:
            if 'population' not in data.columns or 'overall_risk_score' not in data.columns:
                return {'error': 'Required columns not found'}
            
            # Calculate population at different risk levels
            high_risk_pop = data[data['overall_risk_score'] > risk_threshold]['population'].sum()
            medium_risk_pop = data[
                (data['overall_risk_score'] > 0.4) & 
                (data['overall_risk_score'] <= risk_threshold)
            ]['population'].sum()
            low_risk_pop = data[data['overall_risk_score'] <= 0.4]['population'].sum()
            
            total_pop = data['population'].sum()
            
            return {
                'total_population': total_pop,
                'high_risk_population': high_risk_pop,
                'medium_risk_population': medium_risk_pop,
                'low_risk_population': low_risk_pop,
                'high_risk_percentage': (high_risk_pop / total_pop * 100) if total_pop > 0 else 0,
                'medium_risk_percentage': (medium_risk_pop / total_pop * 100) if total_pop > 0 else 0,
                'low_risk_percentage': (low_risk_pop / total_pop * 100) if total_pop > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating population at risk: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def calculate_vulnerability_index(data: pd.DataFrame) -> pd.Series:
        """
        Calculate vulnerability index based on multiple factors
        
        Args:
            data: DataFrame with vulnerability factors
            
        Returns:
            Series with vulnerability index values
        """
        try:
            # Define vulnerability factors and weights
            factors = {
                'poverty_rate': 0.3,
                'rural_percentage': 0.2,
                'children_under_5_percentage': 0.3,
                'food_access_index': 0.2
            }
            
            vulnerability_index = pd.Series(0.0, index=data.index)
            
            for factor, weight in factors.items():
                if factor in data.columns:
                    # Normalize factor to 0-1 scale
                    if factor == 'food_access_index':
                        # Lower food access = higher vulnerability
                        normalized_factor = (100 - data[factor]) / 100
                    else:
                        normalized_factor = data[factor]
                    
                    vulnerability_index += normalized_factor * weight
            
            return vulnerability_index
            
        except Exception as e:
            logger.error(f"Error calculating vulnerability index: {str(e)}")
            return pd.Series(0.0, index=data.index)

# Example usage
if __name__ == "__main__":
    print("Utility Helper Functions Module")
    print("This module provides common functionality for the nutrition policy tool")
    print("Use these functions across the application for data validation, export, and analysis")
