import pandas as pd
import numpy as np
import streamlit as st
from urllib.parse import urlparse
import re
from xgboost import XGBClassifier
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from phishing_analyzer import URLAnalyzer
from federated_learning import (
    FederatedPhishingServer, 
    FederatedPhishingClient, 
    DifferentialPrivacyConfig,
    simulate_federated_training,
    split_data_for_federated_learning
)

# Page configuration
st.set_page_config(
    page_title="🛡️ Advanced Phishing Detection with Federated Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .privacy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .federated-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Advanced Phishing Detection with Federated Learning</h1>
    <p>Privacy-Preserving Machine Learning for Collaborative Threat Detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Federated Learning Settings
    st.subheader("🤝 Federated Learning")
    num_clients = st.slider("Number of Clients", 2, 10, 5)
    num_rounds = st.slider("Training Rounds", 5, 20, 10)
    
    # Differential Privacy Settings
    st.subheader("🔒 Differential Privacy")
    privacy_epsilon = st.slider("Privacy Budget (ε)", 0.1, 5.0, 1.0, 0.1)
    privacy_delta = st.selectbox("Privacy Failure Probability (δ)", 
                                [1e-5, 1e-4, 1e-3], format_func=lambda x: f"{x:.0e}")
    noise_multiplier = st.slider("Noise Multiplier", 0.5, 2.0, 1.1, 0.1)
    
    # Model Settings
    st.subheader("🧠 Model Configuration")
    use_federated = st.checkbox("Enable Federated Learning", value=True)
    use_differential_privacy = st.checkbox("Enable Differential Privacy", value=True)
    
    # Privacy Configuration
    privacy_config = DifferentialPrivacyConfig(
        epsilon=privacy_epsilon,
        delta=privacy_delta,
        noise_multiplier=noise_multiplier
    )

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 URL Analysis", 
    "🤝 Federated Learning", 
    "🔒 Privacy Dashboard", 
    "📊 Model Performance"
])

# Tab 1: URL Analysis
with tab1:
    st.header("🔍 Real-time URL Analysis")
    
    url_input = st.text_input("🔗 Enter a URL to analyze:", "")
    
    if st.button("🔍 Analyze URL", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a URL to analyze.")
        else:
            try:
                analyzer = URLAnalyzer()
                
                with st.spinner("Analyzing URL..."):
                    analysis_results = analyzer.analyze_url(url_input)
                
                # Display results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    risk_score = analysis_results['risk_score']
                    risk_level = analysis_results['risk_level']
                    
                    # Create gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Phishing Risk Score"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 75
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"### Risk Assessment: {risk_level}")
                    st.markdown(f"**Score:** {risk_score}/100")
                    
                    st.markdown("### 🚨 Recommendation")
                    st.markdown(f"**{analysis_results['recommendation']}**")
                    
                    if analysis_results['reasons']:
                        st.markdown("### 📋 Risk Factors")
                        for reason in analysis_results['reasons']:
                            st.markdown(f"• {reason}")
                    
                    if analysis_results['safe_url']:
                        st.markdown("### ✅ Safe Alternative")
                        st.markdown(f"**Official website:** [{analysis_results['safe_url']}]({analysis_results['safe_url']})")
                
                # Add Technical Details section
                st.markdown("---")
                st.markdown("### 🔍 Technical Details")
                
                features = analysis_results['features']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 🌐 Domain Information")
                    st.markdown(f"**Domain:** {features.get('domain', 'N/A')}")
                    st.markdown(f"**TLD:** {features.get('tld', 'N/A')}")
                    st.markdown(f"**Subdomain:** {features.get('subdomain', 'N/A') or 'None'}")
                    st.markdown(f"**Full hostname:** {features.get('hostname', 'N/A')}")
                    if 'domain_age' in features and features['domain_age'] != -1:
                        st.markdown(f"**Domain age:** {features['domain_age']} days")
                
                with col2:
                    st.markdown("#### 🔗 URL Characteristics")
                    st.markdown(f"**URL Length:** {features.get('url_length', 'N/A')}")
                    st.markdown(f"**HTTPS:** {'Yes' if features.get('has_https', 0) == 1 else 'No'}")
                    st.markdown(f"**Path:** {features.get('path', 'N/A') or '/'}")
                    st.markdown(f"**Query string:** {features.get('query_string', 'N/A') or 'None'}")
                
                with col3:
                    st.markdown("#### 🛡️ Security Indicators")
                    st.markdown(f"**Website IP Address:** {features.get('ip_address', 'N/A')}")
                    st.markdown(f"**Contains @ symbol:** {'Yes' if features.get('has_at', 0) == 1 else 'No'}")
                    st.markdown(f"**Contains dash:** {'Yes' if features.get('has_dash', 0) == 1 else 'No'}")
                    st.markdown(f"**Uses URL shortener:** {'Yes' if features.get('uses_shortening_service', 0) == 1 else 'No'}")
                    st.markdown(f"**Typosquatting:** {'Yes' if features.get('is_typosquatting', 0) == 1 else 'No'}")
                
                # If any brands were detected
                if features.get('detected_brands'):
                    st.markdown("#### 🏢 Brand Detection")
                    brand_list = ", ".join(features['detected_brands'])
                    st.markdown(f"Detected references to: **{brand_list}**")
                
                # Additional information about dataset matches
                if features.get('known_phishing'):
                    st.markdown("""
                    <div style="padding: 10px; background-color: #ffeeee; border-left: 5px solid #ff0000; margin: 10px 0;">
                    <h4 style="color: #ff0000;">⚠️ URL Found in Phishing Database</h4>
                    <p>This URL has been previously identified as a phishing attempt in our database.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing URL: {e}")

# Tab 2: Federated Learning
with tab2:
    st.header("🤝 Federated Learning Dashboard")
    
    if use_federated:
        st.markdown("""
        <div class="federated-card">
            <h3>🚀 Collaborative Training Without Data Sharing</h3>
            <p>Multiple organizations can train a phishing detection model together while keeping their data private.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Start Federated Training", type="primary"):
                try:
                    with st.spinner("Loading dataset and starting federated training..."):
                        df = pd.read_csv('phishing_dataset.csv')
                        
                        # Clean and prepare data
                        df.columns = df.columns.str.strip().str.lower()
                        df['label'] = df['label'].astype(str).str.strip().str.lower()
                        df['label'] = df['label'].fillna('legitimate')
                        df['label'] = df['label'].map({'bad': 1, 'good': 0, 'legitimate': 0})
                        df = df.dropna(subset=['label'])
                        
                        # Start federated training
                        server, history, privacy_report = simulate_federated_training(
                            data=df,
                            num_clients=num_clients,
                            num_rounds=num_rounds,
                            privacy_epsilon=privacy_epsilon
                        )
                        
                        # Store results in session state
                        st.session_state.federated_results = {
                            'server': server,
                            'history': history,
                            'privacy_report': privacy_report
                        }
                        
                        st.success("✅ Federated training completed successfully!")
                        
                except Exception as e:
                    st.error(f"Error during federated training: {e}")
        
        with col2:
            if st.button("📊 View Training Results"):
                if 'federated_results' in st.session_state:
                    st.success("Training results loaded!")
                else:
                    st.warning("No training results available. Start federated training first.")
        
        # Display results
        if 'federated_results' in st.session_state:
            results = st.session_state.federated_results
            
            if results['history']:
                st.subheader("📈 Training Progress")
                
                rounds = [r['round'] for r in results['history']]
                accuracies = [r.get('global_accuracy', 0) for r in results['history']]
                privacy_consumed = [r['avg_privacy_consumed'] for r in results['history']]
                
                # Accuracy chart
                fig_acc = px.line(
                    x=rounds, 
                    y=accuracies,
                    title="Global Model Accuracy Over Training Rounds",
                    labels={'x': 'Training Round', 'y': 'Accuracy'}
                )
                st.plotly_chart(fig_acc, use_container_width=True)
                
                # Privacy consumption chart
                fig_privacy = px.line(
                    x=rounds,
                    y=privacy_consumed,
                    title="Privacy Consumption Over Training Rounds",
                    labels={'x': 'Training Round', 'y': 'Privacy Consumed (ε)'}
                )
                st.plotly_chart(fig_privacy, use_container_width=True)
    
    else:
        st.info("Enable federated learning in the sidebar to access this feature.")

# Tab 3: Privacy Dashboard
with tab3:
    st.header("🔒 Privacy & Security Dashboard")
    
    st.markdown("""
    <div class="privacy-card">
        <h3>🛡️ Differential Privacy Protection</h3>
        <p>Your data remains private during training with mathematical privacy guarantees.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Privacy Metrics")
        
        if use_differential_privacy:
            privacy_budget = privacy_config.epsilon
            privacy_used = 0
            
            if 'federated_results' in st.session_state:
                privacy_used = st.session_state.federated_results['privacy_report']['total_privacy_consumed']
            
            remaining_budget = max(0, privacy_budget - privacy_used)
            
            # Privacy budget gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=remaining_budget,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Privacy Budget Remaining"},
                gauge={
                    'axis': {'range': [None, privacy_budget]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, privacy_budget * 0.3], 'color': "red"},
                        {'range': [privacy_budget * 0.3, privacy_budget * 0.7], 'color': "yellow"},
                        {'range': [privacy_budget * 0.7, privacy_budget], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': privacy_budget * 0.3
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Total Privacy Budget", f"{privacy_budget:.2f} ε")
            st.metric("Privacy Used", f"{privacy_used:.2f} ε")
            st.metric("Budget Remaining", f"{remaining_budget:.2f} ε")
        else:
            st.warning("Differential privacy is not enabled.")
    
    with col2:
        st.markdown("### 🔐 Security Features")
        
        security_features = [
            "✅ Federated Learning (No data sharing)",
            "✅ Differential Privacy (Mathematical guarantees)",
            "✅ Secure Aggregation (Encrypted updates)",
            "✅ Model Verification (HMAC integrity)",
            "✅ Privacy Budget Management",
            "✅ Noise Injection (Privacy protection)"
        ]
        
        for feature in security_features:
            st.markdown(feature)

# Tab 4: Model Performance
with tab4:
    st.header("📊 Model Performance & Analytics")
    
    if 'federated_results' in st.session_state:
        results = st.session_state.federated_results
        
        if results['history']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Model Performance")
                final_round = results['history'][-1]
                final_accuracy = final_round.get('global_accuracy', 0)
                
                st.metric("Final Model Accuracy", f"{final_accuracy:.4f}")
                st.metric("Total Training Rounds", len(results['history']))
                st.metric("Average Clients per Round", 
                         f"{np.mean([r['clients_participated'] for r in results['history']]):.1f}")
            
            with col2:
                st.markdown("### 📈 Training Efficiency")
                total_examples = sum([r['total_examples'] for r in results['history']])
                avg_privacy_per_round = np.mean([r['avg_privacy_consumed'] for r in results['history']])
                
                st.metric("Total Examples Processed", f"{total_examples:,}")
                st.metric("Avg Privacy per Round", f"{avg_privacy_per_round:.3f} ε")
                st.metric("Privacy Efficiency", 
                         f"{total_examples / (avg_privacy_per_round * len(results['history'])):.0f} examples/ε")
            
            # Performance summary table
            st.markdown("### 📋 Performance Summary")
            
            summary_data = []
            for r in results['history']:
                summary_data.append({
                    'Round': r['round'],
                    'Accuracy': f"{r.get('global_accuracy', 0):.4f}",
                    'Clients': r['clients_participated'],
                    'Examples': r['total_examples'],
                    'Privacy Used': f"{r['avg_privacy_consumed']:.3f} ε"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Start federated training to view model performance.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🛡️ Advanced Phishing Detection with Federated Learning & Differential Privacy</p>
    <p>Built with Streamlit, PyTorch, and Flower</p>
</div>
""", unsafe_allow_html=True)
