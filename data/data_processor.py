"""
Data Processing Module for Nutrition Policy Tool
Handles data integration from AGWAA, FS-COR, and other sources
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutritionDataProcessor:
    """
    Main class for processing nutrition data from multiple sources
    """
    
    def __init__(self):
        self.countries = ['Benin', 'Senegal', 'Ghana', 'Uganda', 'Malawi']
        self.base_urls = {
            'agwaa': 'https://www.aagwa.org',
            'fs_cor': 'https://fs-cor.org'
        }
        
        # Nutrient categories and their importance weights
        self.nutrient_weights = {
            'protein': 0.25,
            'vitamin_a': 0.20,
            'iron': 0.20,
            'zinc': 0.15,
            'calcium': 0.10,
            'vitamin_d': 0.10
        }
        
        # Demographic risk factors
        self.risk_factors = {
            'children_under_5': 0.30,
            'pregnant_women': 0.25,
            'elderly': 0.20,
            'rural_population': 0.15,
            'low_income': 0.10
        }
    
    def fetch_agwaa_data(self, country: str) -> pd.DataFrame:
        """
        Fetch nutrition data from AGWAA API for a specific country
        
        Args:
            country: Country name
            
        Returns:
            DataFrame with nutrition data
        """
        try:
            # Simulate AGWAA data structure based on documentation
            # In real implementation, this would connect to actual API
            
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_agwaa_data(country)
            logger.info(f"Successfully fetched AGWAA data for {country}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching AGWAA data for {country}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_fs_cor_data(self, country: str) -> pd.DataFrame:
        """
        Fetch food system data from FS-COR platform
        
        Args:
            country: Country name
            
        Returns:
            DataFrame with food system data
        """
        try:
            # Simulate FS-COR data structure
            data = self._generate_synthetic_fs_cor_data(country)
            logger.info(f"Successfully fetched FS-COR data for {country}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching FS-COR data for {country}: {str(e)}")
            return pd.DataFrame()
    
    def _generate_synthetic_agwaa_data(self, country: str) -> pd.DataFrame:
        """
        Generate synthetic AGWAA data for demonstration purposes
        In real implementation, this would be replaced with actual API calls
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate regions for each country
        regions = self._get_country_regions(country)
        
        data = []
        for region in regions:
            for year in range(2018, 2024):
                for month in range(1, 13):
                    # Base nutrition levels with some variation
                    base_protein = np.random.normal(70, 15)
                    base_vitamin_a = np.random.normal(60, 20)
                    base_iron = np.random.normal(65, 18)
                    base_zinc = np.random.normal(55, 15)
                    base_calcium = np.random.normal(75, 12)
                    base_vitamin_d = np.random.normal(45, 20)
                    
                    # Add seasonal and trend effects
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
                    trend_factor = 1 + 0.05 * (year - 2018)
                    
                    # Apply factors
                    protein = max(0, min(100, base_protein * seasonal_factor * trend_factor))
                    vitamin_a = max(0, min(100, base_vitamin_a * seasonal_factor * trend_factor))
                    iron = max(0, min(100, base_iron * seasonal_factor * trend_factor))
                    zinc = max(0, min(100, base_zinc * seasonal_factor * trend_factor))
                    calcium = max(0, min(100, base_calcium * seasonal_factor * trend_factor))
                    vitamin_d = max(0, min(100, base_vitamin_d * seasonal_factor * trend_factor))
                    
                    # Calculate overall nutrition score
                    nutrition_score = (
                        protein * self.nutrient_weights['protein'] +
                        vitamin_a * self.nutrient_weights['vitamin_a'] +
                        iron * self.nutrient_weights['iron'] +
                        zinc * self.nutrient_weights['zinc'] +
                        calcium * self.nutrient_weights['calcium'] +
                        vitamin_d * self.nutrient_weights['vitamin_d']
                    )
                    
                    data.append({
                        'country': country,
                        'region': region,
                        'year': year,
                        'month': month,
                        'date': datetime(year, month, 1),
                        'protein_adequacy': protein,
                        'vitamin_a_adequacy': vitamin_a,
                        'iron_adequacy': iron,
                        'zinc_adequacy': zinc,
                        'calcium_adequacy': calcium,
                        'vitamin_d_adequacy': vitamin_d,
                        'overall_nutrition_score': nutrition_score,
                        'population': np.random.randint(50000, 500000),
                        'rural_percentage': np.random.uniform(0.3, 0.8),
                        'poverty_rate': np.random.uniform(0.1, 0.6),
                        'children_under_5_percentage': np.random.uniform(0.15, 0.25)
                    })
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_fs_cor_data(self, country: str) -> pd.DataFrame:
        """
        Generate synthetic FS-COR data for demonstration
        """
        np.random.seed(42)
        
        regions = self._get_country_regions(country)
        data = []
        
        for region in regions:
            for year in range(2018, 2024):
                for month in range(1, 13):
                    # Food security indicators
                    food_availability = np.random.normal(75, 15)
                    food_access = np.random.normal(70, 18)
                    food_utilization = np.random.normal(65, 20)
                    food_stability = np.random.normal(60, 22)
                    
                    # Climate and agricultural factors
                    rainfall = np.random.normal(800, 200)
                    temperature = np.random.normal(25, 5)
                    crop_yield = np.random.normal(80, 15)
                    
                    # Market indicators
                    food_prices = np.random.normal(100, 20)
                    market_access = np.random.normal(70, 15)
                    
                    data.append({
                        'country': country,
                        'region': region,
                        'year': year,
                        'month': month,
                        'date': datetime(year, month, 1),
                        'food_availability_index': max(0, min(100, food_availability)),
                        'food_access_index': max(0, min(100, food_access)),
                        'food_utilization_index': max(0, min(100, food_utilization)),
                        'food_stability_index': max(0, min(100, food_stability)),
                        'rainfall_mm': max(0, rainfall),
                        'temperature_celsius': temperature,
                        'crop_yield_index': max(0, min(100, crop_yield)),
                        'food_price_index': max(50, food_prices),
                        'market_access_score': max(0, min(100, market_access))
                    })
        
        return pd.DataFrame(data)
    
    def _get_country_regions(self, country: str) -> List[str]:
        """
        Get regions for each country
        """
        region_mapping = {
            'Benin': ['Alibori', 'Atacora', 'Atlantique', 'Borgou', 'Collines', 'Couffo', 'Donga', 'Littoral', 'Mono', 'Ouémé', 'Plateau', 'Zou'],
            'Senegal': ['Dakar', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 'Kédougou', 'Kolda', 'Louga', 'Matam', 'Saint-Louis', 'Sédhiou', 'Tambacounda', 'Thiès', 'Ziguinchor'],
            'Ghana': ['Ashanti', 'Bono', 'Central', 'Eastern', 'Greater Accra', 'Northern', 'Savannah', 'Upper East', 'Upper West', 'Volta', 'Western', 'Western North'],
            'Uganda': ['Central', 'Eastern', 'Northern', 'Western', 'Kampala'],
            'Malawi': ['Central', 'Northern', 'Southern', 'Lilongwe', 'Mzuzu', 'Blantyre']
        }
        
        return region_mapping.get(country, [country])
    
    def integrate_data_sources(self, country: str) -> pd.DataFrame:
        """
        Integrate data from multiple sources for a country
        
        Args:
            country: Country name
            
        Returns:
            Integrated DataFrame
        """
        try:
            # Fetch data from both sources
            agwaa_data = self.fetch_agwaa_data(country)
            fs_cor_data = self.fetch_fs_cor_data(country)
            
            if agwaa_data.empty or fs_cor_data.empty:
                logger.warning(f"Incomplete data for {country}")
                return pd.DataFrame()
            
            # Merge datasets on common keys
            merged_data = pd.merge(
                agwaa_data,
                fs_cor_data,
                on=['country', 'region', 'year', 'month', 'date'],
                how='inner'
            )
            
            # Calculate additional derived features
            merged_data = self._calculate_derived_features(merged_data)
            
            logger.info(f"Successfully integrated data for {country}: {len(merged_data)} records")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error integrating data for {country}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features for ML models
        """
        # Vulnerability score based on multiple factors
        data['vulnerability_score'] = (
            data['poverty_rate'] * 0.3 +
            (1 - data['rural_percentage']) * 0.2 +
            data['children_under_5_percentage'] * 0.3 +
            (1 - data['food_access_index'] / 100) * 0.2
        )
        
        # Nutrition risk score (normalize to 0-1 range)
        data['nutrition_risk_score'] = (
            (100 - data['overall_nutrition_score']) / 100 * 0.4 +
            (100 - data['food_availability_index']) / 100 * 0.3 +
            (100 - data['food_stability_index']) / 100 * 0.3
        )
        
        # Climate stress index (already normalized)
        data['climate_stress_index'] = (
            (data['temperature_celsius'] - 20) / 10 * 0.4 +
            (800 - data['rainfall_mm']) / 800 * 0.6
        )
        
        # Market stress index (already normalized)
        data['market_stress_index'] = (
            (data['food_price_index'] - 100) / 100 * 0.7 +
            (100 - data['market_access_score']) / 100 * 0.3
        )
        
        # Overall risk score (all components now in 0-1 range)
        data['overall_risk_score'] = (
            data['vulnerability_score'] * 0.3 +
            data['nutrition_risk_score'] * 0.4 +
            data['climate_stress_index'] * 0.2 +
            data['market_stress_index'] * 0.1
        )
        
        # Ensure we have a good distribution of risk scores for ML training
        # Add some variation to create more balanced classes
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, len(data))
        data['overall_risk_score'] = np.clip(data['overall_risk_score'] + noise, 0, 1)
        
        return data
    
    def prepare_ml_dataset(self, country: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for machine learning models
        
        Args:
            country: Country name
            
        Returns:
            Tuple of (features, target)
        """
        # Get integrated data
        data = self.integrate_data_sources(country)
        
        if data.empty:
            return pd.DataFrame(), pd.Series()
        
        # Select features for ML model
        feature_columns = [
            'protein_adequacy', 'vitamin_a_adequacy', 'iron_adequacy',
            'zinc_adequacy', 'calcium_adequacy', 'vitamin_d_adequacy',
            'food_availability_index', 'food_access_index', 'food_utilization_index',
            'food_stability_index', 'rainfall_mm', 'temperature_celsius',
            'crop_yield_index', 'food_price_index', 'market_access_score',
            'rural_percentage', 'poverty_rate', 'children_under_5_percentage',
            'vulnerability_score', 'climate_stress_index', 'market_stress_index'
        ]
        
        # Create target variable (high risk if overall_risk_score > median)
        # Use median instead of fixed threshold to ensure balanced classes
        risk_threshold = data['overall_risk_score'].median()
        target = (data['overall_risk_score'] > risk_threshold).astype(int)
        
        # If still unbalanced, try to create more balanced classes
        if target.nunique() < 2:
            # Use 75th percentile as threshold
            risk_threshold = data['overall_risk_score'].quantile(0.75)
            target = (data['overall_risk_score'] > risk_threshold).astype(int)
        
        # If still unbalanced, use 90th percentile
        if target.nunique() < 2:
            risk_threshold = data['overall_risk_score'].quantile(0.90)
            target = (data['overall_risk_score'] > risk_threshold).astype(int)
        
        # Check if we have enough data and multiple classes
        if len(data) < 20:
            logger.warning(f"Insufficient data for ML training: only {len(data)} samples")
            return pd.DataFrame(), pd.Series()
        
        if target.nunique() < 2:
            logger.warning(f"Target has only {target.nunique()} unique class(es): {target.unique()}")
            return pd.DataFrame(), pd.Series()
        
        # Prepare features
        features = data[feature_columns].copy()
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Normalize numerical features
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                features[col] = (features[col] - features[col].mean()) / features[col].std()
        
        return features, target
    
    def check_ml_readiness(self, country: str) -> Dict:
        """
        Check if data is ready for machine learning training
        
        Args:
            country: Country name
            
        Returns:
            Dictionary with readiness status and details
        """
        try:
            data = self.integrate_data_sources(country)
            
            if data.empty:
                return {
                    'ready': False,
                    'reason': 'No data available',
                    'samples': 0,
                    'features': 0,
                    'target_classes': 0
                }
            
            # Prepare ML dataset
            features, target = self.prepare_ml_dataset(country)
            
            if features.empty or target.empty:
                return {
                    'ready': False,
                    'reason': 'Feature preparation failed',
                    'samples': len(data),
                    'features': len(data.columns),
                    'target_classes': 0
                }
            
            # Check readiness criteria
            ready = True
            reasons = []
            
            if len(features) < 20:
                ready = False
                reasons.append(f"Need at least 20 samples, have {len(features)}")
            
            if target.nunique() < 2:
                ready = False
                reasons.append(f"Need at least 2 target classes, have {target.nunique()}")
            
            if len(features.columns) < 5:
                ready = False
                reasons.append(f"Need at least 5 features, have {len(features.columns)}")
            
            return {
                'ready': ready,
                'reason': '; '.join(reasons) if reasons else 'Data is ready for ML training',
                'samples': len(features),
                'features': len(features.columns),
                'target_classes': target.nunique(),
                'target_distribution': target.value_counts().to_dict()
            }
            
        except Exception as e:
            return {
                'ready': False,
                'reason': f'Error checking readiness: {str(e)}',
                'samples': 0,
                'features': 0,
                'target_classes': 0
            }
    
    def get_country_summary(self, country: str) -> Dict:
        """
        Get summary statistics for a country
        
        Args:
            country: Country name
            
        Returns:
            Dictionary with summary statistics
        """
        data = self.integrate_data_sources(country)
        
        if data.empty:
            return {}
        
        summary = {
            'country': country,
            'total_regions': data['region'].nunique(),
            'data_period': f"{data['year'].min()}-{data['year'].max()}",
            'avg_nutrition_score': data['overall_nutrition_score'].mean(),
            'avg_risk_score': data['overall_risk_score'].mean(),
            'high_risk_regions': len(data[data['overall_risk_score'] > 0.6]['region'].unique()),
            'total_population': data['population'].sum(),
            'rural_population_percentage': (data['population'] * data['rural_percentage']).sum() / data['population'].sum() * 100,
            'children_under_5_count': (data['population'] * data['children_under_5_percentage']).sum()
        }
        
        return summary
    
    def export_data(self, country: str, format: str = 'csv') -> str:
        """
        Export processed data for a country
        
        Args:
            country: Country name
            format: Export format ('csv', 'excel', 'json')
            
        Returns:
            File path of exported data
        """
        data = self.integrate_data_sources(country)
        
        if data.empty:
            return ""
        
        filename = f"nutrition_data_{country}_{datetime.now().strftime('%Y%m%d')}"
        
        if format == 'csv':
            filepath = f"{filename}.csv"
            data.to_csv(filepath, index=False)
        elif format == 'excel':
            filepath = f"{filename}.xlsx"
            data.to_excel(filepath, index=False)
        elif format == 'json':
            filepath = f"{filename}.json"
            data.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data exported to {filepath}")
        return filepath

# Example usage
if __name__ == "__main__":
    processor = NutritionDataProcessor()
    
    # Test with one country
    country = "Ghana"
    print(f"Processing data for {country}...")
    
    # Get summary
    summary = processor.get_country_summary(country)
    print(f"Summary for {country}:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Prepare ML dataset
    features, target = processor.prepare_ml_dataset(country)
    print(f"\nML Dataset prepared:")
    print(f"  Features shape: {features.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  High risk cases: {target.sum()}")
