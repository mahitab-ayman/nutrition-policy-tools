#!/usr/bin/env python3
"""
Test Script for Nutrition Policy Tool
Verifies that all components are working correctly
"""

import sys
import os
import traceback

# Add project modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import streamlit as st
        print("✅ streamlit imported successfully")
        
        import plotly.express as px
        print("✅ plotly imported successfully")
        
        import sklearn
        print("✅ scikit-learn imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_processor():
    """Test the data processor module"""
    print("\n🔍 Testing data processor...")
    
    try:
        from data.data_processor import NutritionDataProcessor
        
        processor = NutritionDataProcessor()
        print("✅ NutritionDataProcessor created successfully")
        
        # Test country summary
        summary = processor.get_country_summary("Ghana")
        if summary:
            print("✅ Country summary generated successfully")
            print(f"   - Total regions: {summary.get('total_regions', 'N/A')}")
            print(f"   - Data period: {summary.get('data_period', 'N/A')}")
        else:
            print("❌ Country summary failed")
            return False
        
        # Test data integration
        data = processor.integrate_data_sources("Ghana")
        if not data.empty:
            print("✅ Data integration successful")
            print(f"   - Records: {len(data)}")
            print(f"   - Columns: {len(data.columns)}")
        else:
            print("❌ Data integration failed")
            return False
        
        # Test ML dataset preparation
        features, target = processor.prepare_ml_dataset("Ghana")
        if not features.empty and not target.empty:
            print("✅ ML dataset preparation successful")
            print(f"   - Features: {features.shape}")
            print(f"   - Target: {target.shape}")
        else:
            print("❌ ML dataset preparation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        traceback.print_exc()
        return False

def test_ml_predictor():
    """Test the machine learning predictor module"""
    print("\n🔍 Testing ML predictor...")
    
    try:
        from models.nutrition_predictor import NutritionPredictor
        
        predictor = NutritionPredictor()
        print("✅ NutritionPredictor created successfully")
        
        # Test model initialization
        if predictor.models:
            print("✅ Models initialized successfully")
            print(f"   - Number of models: {len(predictor.models)}")
        else:
            print("❌ Model initialization failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ML predictor test failed: {e}")
        traceback.print_exc()
        return False

def test_utils():
    """Test utility functions"""
    print("\n🔍 Testing utility functions...")
    
    try:
        from utils.helpers import DataValidator, DataExporter, DataAnalyzer
        
        print("✅ Utility classes imported successfully")
        
        # Test data validator
        validator = DataValidator()
        print("✅ DataValidator created successfully")
        
        # Test data exporter
        exporter = DataExporter()
        print("✅ DataExporter created successfully")
        
        # Test data analyzer
        analyzer = DataAnalyzer()
        print("✅ DataAnalyzer created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test end-to-end functionality"""
    print("\n🔍 Testing end-to-end functionality...")
    
    try:
        from data.data_processor import NutritionDataProcessor
        from models.nutrition_predictor import NutritionPredictor
        
        # Create instances
        processor = NutritionDataProcessor()
        predictor = NutritionPredictor()
        
        # Get data for Ghana
        data = processor.integrate_data_sources("Ghana")
        if data.empty:
            print("❌ No data available for end-to-end test")
            return False
        
        # Prepare ML dataset
        features, target = processor.prepare_ml_dataset("Ghana")
        if features.empty or target.empty:
            print("❌ ML dataset preparation failed")
            return False
        
        # Train models (this will take some time)
        print("   - Training models (this may take a few minutes)...")
        results = predictor.train_models(features, target)
        
        if results:
            print("✅ End-to-end test successful!")
            print(f"   - Models trained: {len(results)}")
            
            # Test prediction
            predictions, probabilities = predictor.predict_risk(features)
            if len(predictions) > 0:
                print("✅ Predictions generated successfully")
                print(f"   - High risk cases: {predictions.sum()}")
            else:
                print("❌ Prediction failed")
                return False
        else:
            print("❌ Model training failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Nutrition Policy Tool Tests")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Processor", test_data_processor),
        ("ML Predictor", test_ml_predictor),
        ("Utility Functions", test_utils),
        ("End-to-End", test_end_to_end)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The tool is ready to use.")
        print("\n🚀 To run the dashboard:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   You may need to install missing dependencies or fix configuration issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
