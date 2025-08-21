# 🥗 Nutrition Policy Tool

**AI-Driven Nutrition Gap Analysis & Intervention Simulation for DERPIn Data Challenge 2025**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 **Project Overview**

The Nutrition Policy Tool is an AI-driven application that addresses the critical challenge of identifying and mitigating nutrition gaps in vulnerable populations. Built for the **DERPIn Data Challenge 2025 - Track 3: Inclusive Nutrition Policies**, this tool provides policymakers and community leaders with actionable insights through machine learning predictions and intervention simulations.

## 🌟 **Key Features**

### **🤖 Machine Learning Models**
- **Multiple ML Algorithms**: Random Forest, Gradient Boosting, Logistic Regression, SVM, Ensemble
- **Cross-Validation**: 5-fold cross-validation for robust model evaluation
- **Feature Selection**: Automatic selection of 15 most important features
- **Model Performance**: Comprehensive metrics (Accuracy, AUC, CV Scores)

### **📊 Interactive Dashboard**
- **Country Selection**: Support for 5 DERPIn countries (Benin, Senegal, Ghana, Uganda, Malawi)
- **Regional Analysis**: Granular analysis at regional level
- **Real-time Visualizations**: Interactive charts and metrics
- **Data Export**: Multiple format support (CSV, Excel, JSON)

### **🔬 Intervention Simulation**
- **Three Intervention Types**: Supplementation, Dietary Change, Infrastructure
- **Impact Assessment**: Risk reduction calculations and cost-benefit analysis
- **ROI Analysis**: Return on investment metrics for policy decisions
- **Population Impact**: Number of people benefited from interventions

### **📋 Policy Recommendations**
- **AI-Generated Insights**: Automated policy recommendations based on ML predictions
- **Priority Classification**: High, Medium, Low priority recommendations
- **Cost Estimates**: Budget planning for implementation
- **Timeline Planning**: Short and long-term action plans

## 🏆 **DERPIn Challenge Alignment**

### **Challenge Requirements Met**
✅ **Build AI-driven tools to predict nutrition gaps**  
✅ **Display predictions in interactive dashboards**  
✅ **Gather nutrient adequacy data from AGWAA & FS-COR**  
✅ **Develop models predicting severe nutrient gaps**  
✅ **Allow simulation of intervention impacts**  
✅ **Build clear interfaces for diverse stakeholders**

### **Data Sources Integrated**
- **AGWAA API**: African Growth and Wellbeing Analysis
- **FS-COR Platform**: Food System Crisis Observatory and Response
- **Open Datasets**: Additional nutrition and agricultural indicators

## 📈 **Model Performance**

| Model | Accuracy | AUC | CV Score | Status |
|-------|----------|-----|----------|---------|
| **Logistic Regression** | **82.18%** | **90.02%** | **78.54% ± 1.51%** | 🏆 **Best Model** |
| Random Forest | 77.72% | 84.04% | 75.44% ± 2.11% | ✅ |
| SVM | 79.70% | 87.55% | 75.68% ± 2.00% | ✅ |
| Gradient Boosting | 77.72% | 83.28% | 73.70% ± 1.61% | ✅ |
| Ensemble | 79.21% | 86.93% | 75.56% ± 1.16% | ✅ |

## 🚀 **Quick Start**

### **Prerequisites**
```bash
Python 3.8+
pip install -r requirements.txt
```

### **Installation**
```bash
# Clone the repository
git clone https://github.com/mahitab-ayman/nutrition-policy-tool.git
cd nutrition-policy-tool

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### **Usage**
1. **Select Country**: Choose from 5 DERPIn countries
2. **Train Models**: Click "Train Machine Learning Model" in Risk Prediction tab
3. **Analyze Data**: Explore nutrition trends and regional comparisons
4. **Simulate Interventions**: Test different policy scenarios
5. **Generate Recommendations**: Get AI-powered policy insights

## 🏗️ **Architecture**

```
nutrition-policy-tool/
├── app.py                 # Main Streamlit application
├── data/
│   └── data_processor.py # Data integration & processing
├── models/
│   └── nutrition_predictor.py # ML models & predictions
├── utils/
│   └── helpers.py        # Utility functions
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 **Technical Implementation**

### **Machine Learning Pipeline**
- **Feature Engineering**: 20+ nutrition and socio-economic indicators
- **Data Preprocessing**: Missing value handling, normalization, feature selection
- **Model Training**: Grid search optimization, cross-validation
- **Prediction Pipeline**: Real-time risk assessment and intervention simulation

### **Data Processing**
- **Multi-Source Integration**: AGWAA + FS-COR + derived features
- **Real-time Processing**: Dynamic data loading and feature calculation
- **Quality Assurance**: Data validation and ML readiness checks

### **User Interface**
- **Responsive Design**: Mobile-friendly Streamlit interface
- **Interactive Components**: Real-time charts, filters, and controls
- **Session Management**: Persistent state across user interactions

## 📊 **Data Sources**

### **AGWAA (African Growth and Wellbeing Analysis)**
- **Countries**: Benin, Senegal, Ghana, Uganda, Malawi
- **Indicators**: Nutrient adequacy scores, population demographics
- **Coverage**: 2018-2024 monthly data
- **Regions**: Country-specific administrative regions

### **FS-COR (Food System Crisis Observatory)**
- **Food Security**: Availability, access, utilization, stability indices
- **Climate Data**: Rainfall, temperature, crop yield indicators
- **Market Information**: Food prices, market access scores

### **Derived Features**
- **Vulnerability Score**: Poverty, rural population, children under 5
- **Nutrition Risk Score**: Combined nutrient and food security indicators
- **Climate Stress Index**: Temperature and rainfall stress factors
- **Market Stress Index**: Price volatility and access challenges

## 🎯 **Use Cases**

### **For Policymakers**
- **Resource Allocation**: Identify high-priority regions for intervention
- **Budget Planning**: Cost-benefit analysis of nutrition programs
- **Policy Design**: Evidence-based intervention strategies
- **Impact Monitoring**: Track intervention effectiveness over time

### **For NGOs & Community Leaders**
- **Program Planning**: Target vulnerable populations effectively
- **Resource Optimization**: Maximize impact with limited resources
- **Stakeholder Communication**: Clear data visualization for diverse audiences
- **Capacity Building**: Training and education programs

### **For Researchers**
- **Data Analysis**: Comprehensive nutrition and food security datasets
- **Model Validation**: Test intervention hypotheses
- **Trend Analysis**: Longitudinal nutrition patterns
- **Comparative Studies**: Cross-country and cross-regional analysis

## 🔬 **Methodology**

### **Machine Learning Approach**
1. **Data Integration**: Multi-source data fusion with quality checks
2. **Feature Engineering**: 20+ derived indicators for comprehensive analysis
3. **Model Selection**: Ensemble approach with cross-validation
4. **Performance Evaluation**: Multiple metrics for robust assessment
5. **Interpretability**: Feature importance and explainable AI

### **Intervention Simulation**
1. **Baseline Assessment**: Current risk levels and population distribution
2. **Scenario Modeling**: Different intervention types and strengths
3. **Impact Calculation**: Risk reduction and population benefit metrics
4. **Cost Analysis**: Implementation costs and ROI calculations

## 📈 **Results & Impact**

### **Model Performance**
- **Best Model**: Logistic Regression with 82.18% accuracy
- **AUC Score**: 90.02% indicating excellent discrimination
- **Cross-Validation**: 78.54% ± 1.51% showing model stability
- **Feature Importance**: Identified key drivers of nutrition risk

### **Policy Impact**
- **Targeted Interventions**: Focus on highest-risk populations
- **Cost Efficiency**: Optimized resource allocation
- **Evidence-Based**: Data-driven policy recommendations
- **Scalable Solution**: Applicable across multiple countries

## 🚀 **Future Enhancements**

### **Short-term (3-6 months)**
- **Real-time Data**: Live API integration with AGWAA and FS-COR
- **Mobile App**: Native mobile application for field workers
- **Advanced Analytics**: Time series forecasting and trend analysis

### **Long-term (6-12 months)**
- **Multi-language Support**: Local language interfaces
- **API Development**: RESTful API for third-party integrations
- **Cloud Deployment**: Scalable cloud infrastructure
- **Machine Learning Pipeline**: Automated model retraining

## 🤝 **Contributing**

We welcome contributions to improve the Nutrition Policy Tool! Please see our contributing guidelines for more details.

### **Development Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **DERPIn Data Challenge 2025** for the opportunity to address real-world nutrition challenges
- **AGWAA** for providing comprehensive nutrition data
- **FS-COR** for food system crisis observatory data
- **Open Source Community** for the tools and libraries that made this project possible

## 📞 **Contact**

- **Project**: [Nutrition Policy Tool](https://github.com/mahitab-ayman/nutrition-policy-tool)
- **Author**: Mahitab Ayman
- **Challenge**: DERPIn Data Challenge 2025 - Track 3: Inclusive Nutrition Policies

---

**⭐ Star this repository if you find it helpful!**

**🏆 Built for DERPIn Data Challenge 2025 - Making AI Work for Nutrition Policy!**
