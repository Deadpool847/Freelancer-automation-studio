import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import time
from datetime import datetime
import io
import zipfile

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.api.main import app as fastapi_app
from engine.ingest.scraper import WebScraper
from engine.etl.cleaner import DataCleaner
from engine.features.builder import FeatureBuilder
from engine.ml.detector import TaskDetector
from engine.ml.trainer import ModelTrainer
from engine.ml.registry import ModelRegistry
from engine.explain.explainer import ModelExplainer
from engine.viz.charts import ChartGenerator
from engine.utils.manifest import ManifestManager
from engine.utils.io_helpers import IOHelper

# Page config
st.set_page_config(
    page_title="Freelancer Automation Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'run_id' not in st.session_state:
    st.session_state.run_id = None
if 'manifest' not in st.session_state:
    st.session_state.manifest = None
if 'results' not in st.session_state:
    st.session_state.results = {}

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Freelancer Automation Studio</h1>', unsafe_allow_html=True)
    st.markdown("**Full-Stack ML Pipeline: Ingest → ETL → AutoML → Explain → Export**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=FAS", use_column_width=True)
        st.markdown("### ⚙️ Configuration")
        
        use_gpu = st.checkbox("Enable GPU Acceleration", value=False, help="Use CUDA for deep learning tasks")
        parallel_jobs = st.slider("Parallel Jobs", 1, 8, 4, help="Number of parallel jobs for training")
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        # Display run info
        if st.session_state.run_id:
            st.success(f"✅ Active Run: {st.session_state.run_id[:8]}...")
        else:
            st.info("No active run")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📥 Ingest", "🧹 ETL & Profile", "🤖 AutoML", "📊 Explain", "📈 Visualize", "💾 Export"
    ])
    
    # Tab 1: Ingest
    with tab1:
        st.header("📥 Data Ingestion")
        
        ingest_mode = st.radio("Select Ingest Mode:", ["Upload File", "Web Scraping"], horizontal=True)
        
        if ingest_mode == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Dataset (CSV, Excel, Parquet, JSON)",
                type=['csv', 'xlsx', 'xls', 'parquet', 'json'],
                help="Upload your dataset for analysis"
            )
            
            if uploaded_file:
                if st.button("🚀 Process Uploaded File"):
                    with st.spinner("Processing file..."):
                        # Save to bronze
                        io_helper = IOHelper()
                        file_path = io_helper.save_upload(uploaded_file, "bronze")
                        
                        # Create manifest
                        manifest_mgr = ManifestManager()
                        run_id = manifest_mgr.create_run({
                            "mode": "upload",
                            "source": uploaded_file.name,
                            "timestamp": datetime.now().isoformat()
                        })
                        st.session_state.run_id = run_id
                        st.session_state.results['bronze_path'] = str(file_path)
                        
                        st.success(f"✅ File saved to bronze layer: {file_path.name}")
                        st.success(f"Run ID: {run_id}")
        
        else:  # Web Scraping
            col1, col2 = st.columns(2)
            
            with col1:
                scrape_url = st.text_input("Target URL", placeholder="https://example.com")
                max_pages = st.number_input("Max Pages", min_value=1, max_value=1000, value=10)
            
            with col2:
                respect_robots = st.checkbox("Respect robots.txt", value=True)
                rate_limit = st.number_input("Rate Limit (req/sec)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            
            if st.button("🕷️ Start Scraping"):
                if scrape_url:
                    with st.spinner("Scraping website..."):
                        scraper = WebScraper(
                            respect_robots=respect_robots,
                            rate_limit=rate_limit,
                            max_pages=max_pages
                        )
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            results = scraper.scrape(scrape_url, progress_callback=lambda p: progress_bar.progress(p))
                            
                            # Save results
                            io_helper = IOHelper()
                            file_path = io_helper.save_scraped_data(results, "bronze")
                            
                            # Create manifest
                            manifest_mgr = ManifestManager()
                            run_id = manifest_mgr.create_run({
                                "mode": "scrape",
                                "source": scrape_url,
                                "pages_scraped": len(results),
                                "timestamp": datetime.now().isoformat()
                            })
                            st.session_state.run_id = run_id
                            st.session_state.results['bronze_path'] = str(file_path)
                            
                            st.success(f"✅ Scraped {len(results)} pages successfully!")
                            st.success(f"Run ID: {run_id}")
                            
                        except Exception as e:
                            st.error(f"❌ Scraping failed: {str(e)}")
                else:
                    st.warning("Please enter a URL")
    
    # Tab 2: ETL & Profile
    with tab2:
        st.header("🧹 ETL & Data Profiling")
        
        if 'bronze_path' in st.session_state.results:
            bronze_path = st.session_state.results['bronze_path']
            st.info(f"📁 Source: {Path(bronze_path).name}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("ETL Configuration")
                remove_duplicates = st.checkbox("Remove Duplicates", value=True)
                handle_missing = st.selectbox("Handle Missing Values", ["drop", "fill_mean", "fill_median", "fill_mode"])
                standardize_dates = st.checkbox("Standardize Dates", value=True)
            
            with col2:
                st.subheader("Quality Thresholds")
                min_quality_score = st.slider("Min Quality Score", 0.0, 1.0, 0.7, 0.1)
            
            if st.button("🧹 Run ETL Pipeline"):
                with st.spinner("Cleaning and profiling data..."):
                    try:
                        cleaner = DataCleaner(
                            remove_duplicates=remove_duplicates,
                            handle_missing=handle_missing,
                            standardize_dates=standardize_dates
                        )
                        
                        # Run ETL
                        cleaned_df, profile, quality_score = cleaner.clean_and_profile(bronze_path)
                        
                        # Save to silver
                        io_helper = IOHelper()
                        silver_path = io_helper.save_to_silver(cleaned_df, st.session_state.run_id)
                        
                        st.session_state.results['silver_path'] = str(silver_path)
                        st.session_state.results['profile'] = profile
                        st.session_state.results['quality_score'] = quality_score
                        
                        # Display results
                        st.success(f"✅ ETL Complete! Quality Score: {quality_score:.2%}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(cleaned_df):,}")
                        with col2:
                            st.metric("Columns", len(cleaned_df.columns))
                        with col3:
                            st.metric("Memory", f"{cleaned_df.estimated_size('mb'):.2f} MB")
                        
                        # Show profile
                        st.subheader("📊 Data Profile")
                        st.json(profile)
                        
                        # Preview data
                        st.subheader("👀 Data Preview")
                        st.dataframe(cleaned_df.head(100).to_pandas(), use_container_width=True)
                    
                    except ValueError as e:
                        st.error(f"❌ ETL Error: {str(e)}")
                        st.info("💡 **Tip**: Try changing 'Handle Missing Values' to 'fill_mean', 'fill_median', or 'fill_mode' instead of 'drop'")
                    except Exception as e:
                        st.error(f"❌ Unexpected Error: {str(e)}")
                        st.exception(e)
        else:
            st.warning("⚠️ Please ingest data first (Tab 1)")
    
    # Tab 3: AutoML
    with tab3:
        st.header("🤖 Automated Machine Learning")
        
        if 'silver_path' in st.session_state.results:
            silver_path = st.session_state.results['silver_path']
            st.info(f"📁 Source: {Path(silver_path).name}")
            
            # Auto-detect or manual selection
            detection_mode = st.radio("Task Detection:", ["Auto-Detect", "Manual Select"], horizontal=True)
            
            if detection_mode == "Auto-Detect":
                if st.button("🔍 Detect Task Type"):
                    with st.spinner("Analyzing data and detecting task type..."):
                        detector = TaskDetector()
                        task_info = detector.detect_task(silver_path)
                        
                        st.session_state.results['task_info'] = task_info
                        
                        st.success(f"✅ Detected: **{task_info['task_type']}** | **{task_info['ml_type']}**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.json({
                                "Task Type": task_info['task_type'],
                                "ML Type": task_info['ml_type'],
                                "Target Column": task_info.get('target_column'),
                                "Feature Count": len(task_info.get('features', []))
                            })
                        
                        with col2:
                            if 'recommendations' in task_info:
                                st.markdown("**Recommendations:**")
                                for rec in task_info['recommendations']:
                                    st.markdown(f"- {rec}")
            
            else:  # Manual Select
                col1, col2 = st.columns(2)
                with col1:
                    task_type = st.selectbox("Task Type", [
                        "classification", "regression", "time_series", "nlp", "computer_vision", "clustering"
                    ])
                with col2:
                    target_col = st.text_input("Target Column (if applicable)")
                
                if st.button("✅ Confirm Manual Selection"):
                    st.session_state.results['task_info'] = {
                        'task_type': task_type,
                        'ml_type': task_type,
                        'target_column': target_col if target_col else None
                    }
                    st.success(f"✅ Task set to: {task_type}")
            
            # Train model
            if 'task_info' in st.session_state.results:
                st.markdown("---")
                st.subheader("🎯 Model Training")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
                with col2:
                    cv_folds = st.slider("CV Folds", 3, 10, 5)
                with col3:
                    optuna_trials = st.slider("Optuna Trials", 10, 100, 20, 10)
                
                if st.button("🚀 Train Model"):
                    with st.spinner("Training models... This may take a few minutes."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Feature engineering
                        status_text.text("Building features...")
                        progress_bar.progress(10)
                        
                        feature_builder = FeatureBuilder()
                        X_train, X_test, y_train, y_test = feature_builder.build_features(
                            silver_path,
                            st.session_state.results['task_info'],
                            test_size=test_size
                        )
                        
                        progress_bar.progress(30)
                        status_text.text("Training models...")
                        
                        # Train
                        trainer = ModelTrainer(
                            task_type=st.session_state.results['task_info']['ml_type'],
                            use_gpu=use_gpu,
                            n_jobs=parallel_jobs
                        )
                        
                        model, metrics = trainer.train(
                            X_train, X_test, y_train, y_test,
                            optuna_trials=optuna_trials,
                            cv_folds=cv_folds,
                            progress_callback=lambda p: progress_bar.progress(30 + int(p * 0.6))
                        )
                        
                        progress_bar.progress(90)
                        status_text.text("Saving model...")
                        
                        # Save model
                        registry = ModelRegistry()
                        model_path = registry.save_model(
                            model,
                            st.session_state.run_id,
                            st.session_state.results['task_info']['ml_type'],
                            metrics
                        )
                        
                        st.session_state.results['model'] = model
                        st.session_state.results['metrics'] = metrics
                        st.session_state.results['model_path'] = str(model_path)
                        st.session_state.results['X_test'] = X_test
                        st.session_state.results['y_test'] = y_test
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Training complete!")
                        
                        # Display metrics
                        st.success("🎉 Model Trained Successfully!")
                        
                        st.subheader("📊 Model Performance")
                        metric_cols = st.columns(len(metrics))
                        for i, (key, value) in enumerate(metrics.items()):
                            with metric_cols[i]:
                                st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
        else:
            st.warning("⚠️ Please complete ETL first (Tab 2)")
    
    # Tab 4: Explain
    with tab4:
        st.header("📊 Model Explainability")
        
        if 'model' in st.session_state.results:
            st.info("🔍 Generating SHAP explanations and feature importance...")
            
            if st.button("🧠 Generate Explanations"):
                with st.spinner("Computing SHAP values..."):
                    explainer = ModelExplainer(
                        model=st.session_state.results['model'],
                        task_type=st.session_state.results['task_info']['ml_type']
                    )
                    
                    explanations = explainer.explain(
                        st.session_state.results['X_test'],
                        st.session_state.results['y_test']
                    )
                    
                    st.session_state.results['explanations'] = explanations
                    
                    st.success("✅ Explanations generated!")
                    
                    # Feature Importance
                    st.subheader("🎯 Top Feature Importance")
                    if 'feature_importance' in explanations:
                        importance_df = pd.DataFrame(explanations['feature_importance'])
                        
                        fig = px.bar(
                            importance_df.head(20),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 20 Features",
                            color='importance',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # SHAP Summary
                    st.subheader("🌊 SHAP Summary")
                    if 'shap_summary' in explanations:
                        st.image(explanations['shap_summary'], use_column_width=True)
                    
                    # Error Analysis
                    if 'error_analysis' in explanations:
                        st.subheader("❌ Error Analysis")
                        st.json(explanations['error_analysis'])
        else:
            st.warning("⚠️ Please train a model first (Tab 3)")
    
    # Tab 5: Visualize
    with tab5:
        st.header("📈 Data Visualization")
        
        if 'silver_path' in st.session_state.results:
            chart_gen = ChartGenerator()
            
            # Load data
            import polars as pl
            df = pl.read_parquet(st.session_state.results['silver_path'])
            
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Distribution Plot", "Correlation Heatmap", "Time Series", "Scatter Matrix", "Box Plot"]
            )
            
            if viz_type == "Distribution Plot":
                col = st.selectbox("Select Column", df.columns)
                if st.button("Generate Plot"):
                    fig = chart_gen.distribution_plot(df, col)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Correlation Heatmap":
                if st.button("Generate Heatmap"):
                    fig = chart_gen.correlation_heatmap(df)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Time Series":
                date_col = st.selectbox("Date Column", df.columns)
                value_col = st.selectbox("Value Column", df.columns)
                if st.button("Generate Time Series"):
                    fig = chart_gen.time_series_plot(df, date_col, value_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Scatter Matrix":
                cols = st.multiselect("Select Columns (max 5)", df.columns, max_selections=5)
                if cols and st.button("Generate Matrix"):
                    fig = chart_gen.scatter_matrix(df, cols)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Box Plot":
                col = st.selectbox("Select Column", df.columns)
                if st.button("Generate Box Plot"):
                    fig = chart_gen.box_plot(df, col)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Please complete ETL first (Tab 2)")
    
    # Tab 6: Export
    with tab6:
        st.header("💾 Export & Download")
        
        if st.session_state.run_id:
            st.success(f"✅ Run ID: {st.session_state.run_id}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📦 Available Artifacts")
                
                artifacts = []
                if 'silver_path' in st.session_state.results:
                    artifacts.append("Cleaned Data")
                if 'model_path' in st.session_state.results:
                    artifacts.append("Trained Model")
                if 'explanations' in st.session_state.results:
                    artifacts.append("Explanations")
                if 'profile' in st.session_state.results:
                    artifacts.append("Data Profile")
                
                for artifact in artifacts:
                    st.checkbox(artifact, value=True, key=f"export_{artifact}")
            
            with col2:
                st.subheader("⚙️ Export Settings")
                export_format = st.selectbox("Format", ["Parquet", "CSV", "Excel", "JSON"])
                include_manifest = st.checkbox("Include Manifest", value=True)
            
            if st.button("📥 Download All"):
                with st.spinner("Preparing download..."):
                    io_helper = IOHelper()
                    
                    # Create zip
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Add artifacts
                        if 'silver_path' in st.session_state.results:
                            zip_file.write(
                                st.session_state.results['silver_path'],
                                f"data/cleaned_data.{export_format.lower()}"
                            )
                        
                        if 'model_path' in st.session_state.results:
                            zip_file.write(
                                st.session_state.results['model_path'],
                                "models/trained_model.joblib"
                            )
                        
                        if include_manifest:
                            manifest_mgr = ManifestManager()
                            manifest = manifest_mgr.get_manifest(st.session_state.run_id)
                            zip_file.writestr(
                                "manifest.json",
                                json.dumps(manifest, indent=2)
                            )
                        
                        # Add metrics
                        if 'metrics' in st.session_state.results:
                            zip_file.writestr(
                                "metrics.json",
                                json.dumps(st.session_state.results['metrics'], indent=2)
                            )
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download ZIP",
                        data=zip_buffer,
                        file_name=f"fas_export_{st.session_state.run_id[:8]}.zip",
                        mime="application/zip"
                    )
                    
                    st.success("✅ Export ready for download!")
        else:
            st.warning("⚠️ No active run. Please complete a pipeline first.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>" 
        "🚀 Freelancer Automation Studio | Built with Streamlit & FastAPI"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()