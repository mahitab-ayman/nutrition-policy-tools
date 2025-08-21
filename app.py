"""
Main Streamlit Application for Nutrition Policy Tool
Provides interactive dashboard for nutrition gap analysis and intervention simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import sys
import os

# Add project modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_processor import NutritionDataProcessor
from models.nutrition_predictor import NutritionPredictor

# Page configuration
st.set_page_config(
    page_title="Nutrition Policy Tool",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        color: #d62728;
        font-weight: bold;
    }
    .medium-risk {
        color: #ff7f0e;
        font-weight: bold;
    }
    .low-risk {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class NutritionPolicyDashboard:
    """
    Main dashboard class for nutrition policy analysis
    """
    
    def __init__(self):
        self.data_processor = NutritionDataProcessor()
        self.predictor = NutritionPredictor()
        self.countries = ['Benin', 'Senegal', 'Ghana', 'Uganda', 'Malawi']
        
        # Initialize session state
        if 'current_country' not in st.session_state:
            st.session_state.current_country = 'Ghana'
        if 'current_region' not in st.session_state:
            st.session_state.current_region = 'All'
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        if 'model_country' not in st.session_state:
            st.session_state.model_country = None
    
    def run(self):
        """
        Main method to run the dashboard
        """
        # Header
        st.markdown('<h1 class="main-header">🥗 Nutrition Policy Tool</h1>', unsafe_allow_html=True)
        st.markdown("### AI-Driven Nutrition Gap Analysis & Intervention Simulation")
        st.markdown("---")
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        if st.session_state.current_country:
            self._show_country_overview()
            
            # Tabs for different functionalities
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Data Analysis", 
                "🤖 Risk Prediction", 
                "🔬 Intervention Simulation",
                "📋 Policy Recommendations"
            ])
            
            with tab1:
                self._show_data_analysis()
            
            with tab2:
                self._show_risk_prediction()
            
            with tab3:
                self._show_intervention_simulation()
            
            with tab4:
                self._show_policy_recommendations()
    
    def _create_sidebar(self):
        """
        Create the sidebar with navigation options
        """
        st.sidebar.title("Navigation")
        
        # Country selection
        selected_country = st.sidebar.selectbox(
            "Select Country",
            self.countries,
            index=self.countries.index(st.session_state.current_country)
        )
        
        if selected_country != st.session_state.current_country:
            st.session_state.current_country = selected_country
            st.session_state.current_region = 'All'
            # Update model training status for new country
            if selected_country in st.session_state.trained_models:
                st.session_state.model_trained = True
                st.session_state.model_country = selected_country
            else:
                st.session_state.model_trained = False
                st.session_state.model_country = None
            st.rerun()
        
        # Region selection
        if st.session_state.current_country:
            regions = ['All'] + self.data_processor._get_country_regions(st.session_state.current_country)
            selected_region = st.sidebar.selectbox(
                "Select Region",
                regions,
                index=regions.index(st.session_state.current_region)
            )
            
            if selected_region != st.session_state.current_region:
                st.session_state.current_region = selected_region
                st.rerun()
        
        # Data refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Data quality check button
        if st.sidebar.button("🔍 Check Data Quality"):
            with st.sidebar:
                readiness = self.data_processor.check_ml_readiness(st.session_state.current_country)
                if readiness['ready']:
                    st.success("✅ Data Ready")
                else:
                    st.error("❌ Data Not Ready")
                st.info(f"📊 {readiness['samples']} samples")
                st.info(f"🎯 {readiness['target_classes']} classes")
        
        # Show trained models status
        st.sidebar.markdown("### 🤖 Trained Models")
        if st.session_state.trained_models:
            for country in st.session_state.trained_models.keys():
                if country == st.session_state.current_country:
                    st.sidebar.success(f"✅ {country} (Current)")
                else:
                    st.sidebar.info(f"✅ {country}")
        else:
            st.sidebar.info("No models trained yet")
        
        st.sidebar.markdown("---")
        
        # About section
        st.sidebar.markdown("### About")
        st.sidebar.markdown("""
        This tool helps policymakers and community leaders:
        - Identify nutrition gaps
        - Predict future risks
        - Simulate interventions
        - Generate policy recommendations
        """)
        
        # Data sources
        st.sidebar.markdown("### Data Sources")
        st.sidebar.markdown("""
        - **AGWAA**: African Growth and Wellbeing Analysis
        - **FS-COR**: Food System Crisis Observatory
        - **Open Datasets**: Nutrition and agricultural data
        """)
    
    def _show_country_overview(self):
        """
        Show overview metrics for the selected country
        """
        st.header(f"🇺🇳 {st.session_state.current_country} Overview")
        
        # Get country summary
        summary = self.data_processor.get_country_summary(st.session_state.current_country)
        
        if not summary:
            st.error("Unable to load country data. Please try refreshing.")
            return
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Population",
                f"{summary['total_population']:,.0f}",
                help="Total population across all regions"
            )
        
        with col2:
            st.metric(
                "Rural Population",
                f"{summary['rural_population_percentage']:.1f}%",
                help="Percentage of population living in rural areas"
            )
        
        with col3:
            st.metric(
                "Children Under 5",
                f"{summary['children_under_5_count']:,.0f}",
                help="Number of children under 5 years old"
            )
        
        with col4:
            st.metric(
                "High Risk Regions",
                summary['high_risk_regions'],
                help="Number of regions with high nutrition risk"
            )
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Average Nutrition Score",
                f"{summary['avg_nutrition_score']:.1f}/100",
                help="Overall nutrition adequacy score"
            )
        
        with col2:
            risk_score = summary['avg_risk_score']
            risk_color = "high-risk" if risk_score > 0.6 else "medium-risk" if risk_score > 0.4 else "low-risk"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Average Risk Score</h4>
                <p class="{risk_color}">{risk_score:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _show_data_analysis(self):
        """
        Show data analysis and visualization
        """
        st.header("📊 Data Analysis")
        
        # Load data
        data = self.data_processor.integrate_data_sources(st.session_state.current_country)
        
        if data.empty:
            st.error("No data available for analysis.")
            return
        
        # Filter by region if selected
        if st.session_state.current_region != 'All':
            data = data[data['region'] == st.session_state.current_region]
        
        # Time series analysis
        st.subheader("📈 Nutrition Trends Over Time")
        
        # Aggregate by year and month
        time_data = data.groupby(['year', 'month']).agg({
            'overall_nutrition_score': 'mean',
            'overall_risk_score': 'mean',
            'food_availability_index': 'mean',
            'food_access_index': 'mean'
        }).reset_index()
        
        time_data['date'] = pd.to_datetime(time_data[['year', 'month']].assign(day=1))
        
        # Create time series plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Nutrition Score', 'Risk Score', 'Food Availability', 'Food Access'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=time_data['date'], y=time_data['overall_nutrition_score'], 
                      name='Nutrition Score', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_data['date'], y=time_data['overall_risk_score'], 
                      name='Risk Score', line=dict(color='red')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=time_data['date'], y=time_data['food_availability_index'], 
                      name='Food Availability', line=dict(color='blue')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_data['date'], y=time_data['food_access_index'], 
                      name='Food Access', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional comparison
        st.subheader("🗺️ Regional Comparison")
        
        if st.session_state.current_region == 'All':
            regional_data = data.groupby('region').agg({
                'overall_nutrition_score': 'mean',
                'overall_risk_score': 'mean',
                'population': 'sum',
                'poverty_rate': 'mean'
            }).reset_index()
            
            # Create regional comparison chart
            fig = px.scatter(
                regional_data,
                x='overall_nutrition_score',
                y='overall_risk_score',
                size='population',
                color='poverty_rate',
                hover_data=['region'],
                title='Regional Nutrition vs Risk Analysis',
                labels={
                    'overall_nutrition_score': 'Nutrition Score',
                    'overall_risk_score': 'Risk Score',
                    'population': 'Population Size',
                    'poverty_rate': 'Poverty Rate'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Nutrient breakdown
        st.subheader("🥗 Nutrient Adequacy Breakdown")
        
        nutrient_cols = [col for col in data.columns if 'adequacy' in col.lower()]
        if nutrient_cols:
            nutrient_data = data[nutrient_cols].mean()
            
            fig = px.bar(
                x=nutrient_data.values,
                y=[col.replace('_', ' ').title() for col in nutrient_data.index],
                orientation='h',
                title='Average Nutrient Adequacy Scores',
                labels={'x': 'Score', 'y': 'Nutrient'}
            )
            
            fig.add_vline(x=70, line_dash="dash", line_color="red", 
                         annotation_text="Target Level")
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_risk_prediction(self):
        """
        Show risk prediction functionality
        """
        st.header("🤖 Risk Prediction")
        
        # Check if model is trained for current country
        current_model_available = (
            st.session_state.model_trained and 
            st.session_state.model_country == st.session_state.current_country and
            st.session_state.current_country in st.session_state.trained_models
        )
        
        if not current_model_available:
            st.info("⚠️ Model needs to be trained first. Click 'Train Model' below.")
            
            if st.button("🚀 Train Machine Learning Model"):
                with st.spinner("Checking data readiness..."):
                    # Check if data is ready for ML training
                    readiness = self.data_processor.check_ml_readiness(st.session_state.current_country)
                    
                    if not readiness['ready']:
                        st.error(f"❌ Data not ready for training: {readiness['reason']}")
                        st.info(f"📊 Current status: {readiness['samples']} samples, {readiness['features']} features, {readiness['target_classes']} target classes")
                        return
                    
                    st.success(f"✅ Data is ready! {readiness['samples']} samples, {readiness['features']} features, {readiness['target_classes']} target classes")
                    st.info(f"🎯 Target distribution: {readiness['target_distribution']}")
                
                with st.spinner("Training models... This may take a few minutes."):
                    # Prepare data for training
                    features, target = self.data_processor.prepare_ml_dataset(st.session_state.current_country)
                    
                    # Train models
                    results = self.predictor.train_models(features, target)
                    
                    if results:
                        st.session_state.model_trained = True
                        st.session_state.model_country = st.session_state.current_country
                        st.session_state.trained_models[st.session_state.current_country] = self.predictor
                        st.success("✅ Model trained successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Model training failed. Please check your data and try again.")
            
            return
        
        # Model is trained, show prediction interface
        st.success("✅ Model is ready for predictions!")
        
        # Get the trained model for current country
        trained_predictor = st.session_state.trained_models[st.session_state.current_country]
        
        # Load current data for prediction
        data = self.data_processor.integrate_data_sources(st.session_state.current_country)
        
        if data.empty:
            st.error("No data available for prediction.")
            return
        
        # Filter by region if selected
        if st.session_state.current_region != 'All':
            data = data[data['region'] == st.session_state.current_region]
        
        # Prepare features for prediction
        features, _ = self.data_processor.prepare_ml_dataset(st.session_state.current_country)
        
        if not features.empty:
            # Make predictions using the trained model
            predictions, probabilities = trained_predictor.predict_risk(features)
            
            # Display prediction results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Predicted High Risk Cases",
                    f"{predictions.sum()}/{len(predictions)}",
                    f"({predictions.sum()/len(predictions)*100:.1f}%)"
                )
            
            with col2:
                avg_risk = probabilities.mean() if probabilities is not None else 0
                risk_color = "high-risk" if avg_risk > 0.6 else "medium-risk" if avg_risk > 0.4 else "low-risk"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Average Predicted Risk</h4>
                    <p class="{risk_color}">{avg_risk:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk distribution
            st.subheader("📊 Risk Distribution")
            
            if probabilities is not None:
                fig = px.histogram(
                    x=probabilities,
                    nbins=20,
                    title='Distribution of Predicted Risk Scores',
                    labels={'x': 'Risk Score', 'y': 'Frequency'}
                )
                
                fig.add_vline(x=0.6, line_dash="dash", line_color="red", 
                             annotation_text="High Risk Threshold")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("🔍 Feature Importance")
            
            importance_df = trained_predictor.get_feature_importance()
            if not importance_df.empty:
                fig = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 Most Important Features',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        if hasattr(trained_predictor, 'model_performance') and trained_predictor.model_performance:
            st.subheader("📈 Model Performance")
            
            performance_data = []
            for name, results in trained_predictor.model_performance.items():
                performance_data.append({
                    'Model': name.replace('_', ' ').title(),
                    'Accuracy': f"{results['accuracy']:.3f}",
                    'AUC': f"{results['auc']:.3f}" if results['auc'] else 'N/A',
                    'CV Score': f"{results['cv_mean']:.3f} ± {results['cv_std']:.3f}"
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
    
    def _show_intervention_simulation(self):
        """
        Show intervention simulation functionality
        """
        st.header("🔬 Intervention Simulation")
        
        # Check if model is trained for current country
        current_model_available = (
            st.session_state.model_trained and 
            st.session_state.model_country == st.session_state.current_country and
            st.session_state.current_country in st.session_state.trained_models
        )
        
        if not current_model_available:
            st.warning("⚠️ Please train the model first in the Risk Prediction tab.")
            return
        
        st.info("💡 Simulate the impact of different nutrition interventions on risk levels.")
        
        # Intervention parameters
        col1, col2 = st.columns(2)
        
        with col1:
            intervention_type = st.selectbox(
                "Intervention Type",
                ['supplementation', 'dietary_change', 'infrastructure'],
                help="Type of nutrition intervention to simulate"
            )
        
        with col2:
            intervention_strength = st.slider(
                "Intervention Strength",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Strength of the intervention (10% to 50% improvement)"
            )
        
        # Run simulation
        if st.button("🚀 Run Simulation"):
            with st.spinner("Running simulation..."):
                # Load data for simulation
                data = self.data_processor.integrate_data_sources(st.session_state.current_country)
                
                if not data.empty:
                    # Filter by region if selected
                    if st.session_state.current_region != 'All':
                        data = data[data['region'] == st.session_state.current_region]
                    
                    # Prepare features
                    features, _ = self.data_processor.prepare_ml_dataset(st.session_state.current_country)
                    
                    if not features.empty:
                        # Get the trained model for current country
                        trained_predictor = st.session_state.trained_models[st.session_state.current_country]
                        
                        # Run simulation
                        results = trained_predictor.simulate_intervention(
                            features, intervention_type, intervention_strength
                        )
                        
                        if results:
                            # Display results
                            st.success("✅ Simulation completed!")
                            
                            # Results metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Baseline Risk",
                                    f"{results['baseline_risk']:.3f}",
                                    help="Risk level before intervention"
                                )
                            
                            with col2:
                                st.metric(
                                    "Post-Intervention Risk",
                                    f"{results['post_intervention_risk']:.3f}",
                                    help="Risk level after intervention"
                                )
                            
                            with col3:
                                st.metric(
                                    "Risk Reduction",
                                    f"{results['risk_reduction']:.3f}",
                                    f"{results['risk_reduction_percentage']:.1f}%",
                                    help="Absolute and percentage risk reduction"
                                )
                            
                            with col4:
                                st.metric(
                                    "People Benefited",
                                    results['people_benefited'],
                                    help="Number of people moved from high to low risk"
                                )
                            
                            # Visualization
                            st.subheader("📊 Intervention Impact")
                            
                            # Before vs After comparison
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                name='Baseline',
                                x=['High Risk', 'Low Risk'],
                                y=[results['high_risk_baseline'], 
                                   len(features) - results['high_risk_baseline']],
                                marker_color=['red', 'green']
                            ))
                            
                            fig.add_trace(go.Bar(
                                name='Post-Intervention',
                                x=['High Risk', 'Low Risk'],
                                y=[results['high_risk_post_intervention'], 
                                   len(features) - results['high_risk_post_intervention']],
                                marker_color=['darkred', 'darkgreen']
                            ))
                            
                            fig.update_layout(
                                title=f'Risk Distribution: Before vs After {intervention_type.title()} Intervention',
                                barmode='group',
                                xaxis_title='Risk Category',
                                yaxis_title='Number of Cases'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Cost-benefit analysis
                            st.subheader("💰 Cost-Benefit Analysis")
                            
                            cost_estimates = {
                                'supplementation': {'low': 50, 'medium': 150, 'high': 300},
                                'dietary_change': {'low': 20, 'medium': 80, 'high': 200},
                                'infrastructure': {'low': 100, 'medium': 500, 'high': 1000}
                            }
                            
                            costs = cost_estimates[intervention_type]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Low Cost Estimate",
                                    f"${costs['low']:,.0f}",
                                    help="Per person per year"
                                )
                            
                            with col2:
                                st.metric(
                                    "Medium Cost Estimate",
                                    f"${costs['medium']:,.0f}",
                                    help="Per person per year"
                                )
                            
                            with col3:
                                st.metric(
                                    "High Cost Estimate",
                                    f"${costs['high']:,.0f}",
                                    help="Per person per year"
                                )
                            
                            # ROI calculation
                            if results['people_benefited'] > 0:
                                avg_cost = costs['medium']  # Use medium cost for ROI
                                total_cost = results['people_benefited'] * avg_cost
                                roi = (results['risk_reduction'] * 100) / (total_cost / 1000)  # Risk reduction per $1000
                                
                                st.info(f"""
                                💡 **ROI Analysis**: 
                                - Total intervention cost: ${total_cost:,.0f}
                                - Risk reduction per $1000 spent: {roi:.2f}%
                                - Cost per person benefited: ${avg_cost:,.0f}
                                """)
                        else:
                            st.error("❌ Simulation failed. Please check your data and model.")
                    else:
                        st.error("❌ Insufficient data for simulation.")
                else:
                    st.error("❌ No data available for simulation.")
    
    def _show_policy_recommendations(self):
        """
        Show policy recommendations
        """
        st.header("📋 Policy Recommendations")
        
        # Check if model is trained for current country
        current_model_available = (
            st.session_state.model_trained and 
            st.session_state.model_country == st.session_state.current_country and
            st.session_state.current_country in st.session_state.trained_models
        )
        
        if not current_model_available:
            st.warning("⚠️ Please train the model first in the Risk Prediction tab.")
            return
        
        st.info("💡 AI-generated policy recommendations based on current risk analysis.")
        
        # Generate recommendations
        if st.button("🔍 Generate Recommendations"):
            with st.spinner("Analyzing data and generating recommendations..."):
                # Load data
                data = self.data_processor.integrate_data_sources(st.session_state.current_country)
                
                if not data.empty:
                    # Filter by region if selected
                    if st.session_state.current_region != 'All':
                        data = data[data['region'] == st.session_state.current_region]
                    
                    # Prepare features
                    features, _ = self.data_processor.prepare_ml_dataset(st.session_state.current_country)
                    
                    if not features.empty:
                        # Get the trained model for current country
                        trained_predictor = st.session_state.trained_models[st.session_state.current_country]
                        
                        # Generate recommendations
                        recommendations = trained_predictor.generate_policy_recommendations(
                            features, st.session_state.current_country, st.session_state.current_region
                        )
                        
                        if recommendations:
                            st.success(f"✅ Generated {len(recommendations)} recommendations!")
                            
                            # Display recommendations
                            for i, rec in enumerate(recommendations, 1):
                                priority_color = {
                                    'High': '🔴',
                                    'Medium': '🟡',
                                    'Low': '🟢'
                                }.get(rec['priority'], '⚪')
                                
                                with st.expander(f"{priority_color} {rec['priority']} Priority: {rec['category']}"):
                                    st.markdown(f"""
                                    **Recommendation:** {rec['recommendation']}
                                    
                                    **Rationale:** {rec['rationale']}
                                    
                                    **Estimated Impact:** {rec['estimated_impact']}
                                    
                                    **Cost Estimate:** {rec['cost_estimate']}
                                    
                                    **Timeline:** {rec['timeline']}
                                    """)
                        else:
                            st.warning("⚠️ No specific recommendations generated. Data may indicate low risk levels.")
                    else:
                        st.error("❌ Insufficient data for generating recommendations.")
                else:
                    st.error("❌ No data available for generating recommendations.")
        
        # Manual recommendation input
        st.subheader("✍️ Add Custom Recommendations")
        
        with st.form("custom_recommendation"):
            custom_priority = st.selectbox("Priority", ["High", "Medium", "Low"])
            custom_category = st.text_input("Category", placeholder="e.g., Infrastructure, Education, Emergency")
            custom_recommendation = st.text_area("Recommendation", placeholder="Enter your recommendation here...")
            custom_rationale = st.text_area("Rationale", placeholder="Why is this recommendation important?")
            custom_timeline = st.text_input("Timeline", placeholder="e.g., 3-6 months")
            
            if st.form_submit_button("💾 Save Recommendation"):
                if custom_recommendation and custom_rationale:
                    st.success("✅ Custom recommendation saved!")
                else:
                    st.error("❌ Please fill in all required fields.")
        
        # Export recommendations
        st.subheader("📤 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export to PDF"):
                st.info("PDF export functionality would be implemented here.")
        
        with col2:
            if st.button("📊 Export to Excel"):
                st.info("Excel export functionality would be implemented here.")

def main():
    """
    Main function to run the dashboard
    """
    try:
        dashboard = NutritionPolicyDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the console for more details.")

if __name__ == "__main__":
    main()
