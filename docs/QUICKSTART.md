# Quick Start Guide
## Nutrition Policy Tool - Hackathon Challenge Track 3

### 🚀 Getting Started

This guide will help you get the Nutrition Policy Tool up and running quickly. The tool provides AI-driven nutrition gap analysis and intervention simulation for policymakers and community leaders.

### 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8 or higher** installed on your system
- **Git** for cloning the repository
- **Internet connection** for downloading dependencies
- **At least 4GB RAM** for running the machine learning models

### 🛠️ Installation

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd nutrition-policy-tool
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter any issues with specific packages, try installing them individually:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

#### Step 3: Verify Installation

```bash
python -c "import streamlit, pandas, numpy, sklearn; print('All packages installed successfully!')"
```

### 🎯 Running the Application

#### Start the Dashboard

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

#### Alternative: Run with Custom Port

```bash
streamlit run app.py --server.port 8080
```

### 📱 Using the Dashboard

#### 1. **Country Selection**
- Use the sidebar to select your target country
- Choose from: Benin, Senegal, Ghana, Uganda, Malawi
- Select specific regions for detailed analysis

#### 2. **Data Analysis Tab**
- View nutrition trends over time
- Compare regional nutrition indicators
- Analyze nutrient adequacy breakdowns
- Export data in various formats

#### 3. **Risk Prediction Tab**
- Train machine learning models on your data
- View predicted nutrition risks
- Analyze feature importance
- Monitor model performance metrics

#### 4. **Intervention Simulation Tab**
- Simulate different intervention types:
  - **Supplementation**: Vitamin and mineral programs
  - **Dietary Change**: Food security improvements
  - **Infrastructure**: Market and climate resilience
- Adjust intervention strength (10-50%)
- View cost-benefit analysis and ROI

#### 5. **Policy Recommendations Tab**
- Generate AI-driven policy recommendations
- Add custom recommendations
- Export recommendations for stakeholders

### 🔧 Configuration

#### Environment Variables

Create a `.env` file in the project root:

```env
# Data source configurations
AGWAA_API_KEY=your_api_key_here
FS_COR_API_KEY=your_api_key_here

# Model parameters
RISK_THRESHOLD=0.6
CROSS_VALIDATION_FOLDS=5

# Export settings
DEFAULT_EXPORT_FORMAT=csv
MAX_EXPORT_RECORDS=10000
```

#### Customizing Risk Thresholds

Edit `models/nutrition_predictor.py` to adjust risk classification:

```python
# Change the risk threshold for high-risk classification
target = (data['overall_risk_score'] > 0.7).astype(int)  # Default: 0.6
```

### 📊 Data Sources

#### Current Implementation

The tool currently uses **synthetic data** for demonstration purposes. To connect to real data sources:

1. **AGWAA API**: Update `data/data_processor.py` with real API endpoints
2. **FS-COR Platform**: Implement web scraping or API integration
3. **Custom Data**: Modify the data loading functions to accept your datasets

#### Adding New Data Sources

1. Create a new method in `NutritionDataProcessor` class
2. Implement data fetching and cleaning logic
3. Update the `integrate_data_sources` method
4. Add validation and error handling

### 🤖 Machine Learning Models

#### Model Training

Models are trained automatically when you click "Train Model" in the Risk Prediction tab:

1. **Data Preparation**: Feature selection and scaling
2. **Model Training**: Ensemble of 4 algorithms
3. **Validation**: 5-fold cross-validation
4. **Selection**: Best model chosen automatically

#### Model Performance

Monitor these metrics:
- **Accuracy**: Overall prediction correctness
- **AUC**: Area under ROC curve
- **CV Score**: Cross-validation performance
- **Feature Importance**: Key factors driving predictions

#### Saving and Loading Models

```python
# Save trained model
predictor.save_model('models/nutrition_model.pkl')

# Load saved model
predictor.load_model('models/nutrition_model.pkl')
```

### 📈 Customization

#### Adding New Countries

1. Update `countries` list in `NutritionDataProcessor`
2. Add region mapping in `_get_country_regions`
3. Implement country-specific data processing if needed

#### Custom Interventions

Add new intervention types in `simulate_intervention` method:

```python
elif intervention_type == 'education':
    # Improve education-related indicators
    education_cols = [col for col in features.columns if 'education' in col.lower()]
    for col in education_cols:
        modified_features[col] = features[col] * (1 + intervention_strength)
```

#### Custom Metrics

Extend the dashboard with new visualizations:

```python
# Add new chart in _show_data_analysis method
st.subheader("Custom Metric")
fig = px.line(data, x='date', y='your_metric')
st.plotly_chart(fig, use_container_width=True)
```

### 🚨 Troubleshooting

#### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the correct directory
pwd  # Should show nutrition-policy-tool path
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. Memory Issues**
```bash
# Reduce data size for testing
# Edit data_processor.py to generate fewer records
for year in range(2020, 2024):  # Reduce from 2018-2024
```

**3. Model Training Fails**
- Check data quality and completeness
- Reduce feature set size
- Increase available memory
- Use simpler models initially

**4. Dashboard Won't Load**
```bash
# Check if port is in use
netstat -an | grep 8501
# Kill process if needed
kill -9 <process_id>
```

#### Performance Optimization

**For Large Datasets:**
```python
# Enable chunked processing
CHUNK_SIZE = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=CHUNK_SIZE):
    # Process chunk
    pass
```

**For Faster Model Training:**
```python
# Reduce cross-validation folds
cv_scores = cross_val_score(model, X_train, y_train, cv=3)  # Default: 5
```

### 📚 Next Steps

#### For Users
1. **Explore the Dashboard**: Familiarize yourself with all tabs
2. **Train Models**: Start with one country to understand the process
3. **Run Simulations**: Test different intervention scenarios
4. **Generate Reports**: Export findings for stakeholders

#### For Developers
1. **Review the Code**: Understand the modular architecture
2. **Add Real Data**: Implement actual API connections
3. **Enhance Models**: Experiment with different algorithms
4. **Extend Features**: Add new analysis capabilities

#### For Researchers
1. **Validate Results**: Compare with existing research
2. **Improve Models**: Contribute to methodology
3. **Add Indicators**: Include new nutrition metrics
4. **Publish Findings**: Share insights with the community

### 📞 Support

#### Getting Help
- **Documentation**: Check the `docs/` folder
- **Code Comments**: Inline documentation in source files
- **Logs**: Check console output for error messages
- **Issues**: Report bugs and feature requests

#### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### 🎉 Success!

You're now ready to use the Nutrition Policy Tool! Start by:

1. **Running the dashboard**: `streamlit run app.py`
2. **Selecting a country**: Choose from the sidebar
3. **Training a model**: Go to Risk Prediction tab
4. **Exploring data**: Use Data Analysis tab
5. **Simulating interventions**: Try Intervention Simulation tab

The tool will help you make data-driven decisions about nutrition policies and interventions. Happy analyzing! 🥗📊

---

*This quick start guide is part of the Nutrition Policy Tool developed for the Hackathon Challenge Track 3: Inclusive Nutrition Policies.*
