"""Interactive web dashboard for the Precise MRD Pipeline."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from streamlit_plotly_events import plotly_events

from .config import PipelineConfig, load_config, dump_config, ConfigValidator, PredefinedTemplates
from .api import job_manager


class MRDDashboard:
    """Interactive web dashboard for MRD pipeline monitoring and control."""

    def __init__(self):
        """Initialize the dashboard."""
        self.title = "🔬 Precise MRD Pipeline Dashboard"
        self.description = "Interactive interface for Minimal Residual Disease detection and analysis"

        # Set page configuration
        st.set_page_config(
            page_title="Precise MRD Dashboard",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize session state
        if 'jobs' not in st.session_state:
            st.session_state.jobs = {}
        if 'current_job' not in st.session_state:
            st.session_state.current_job = None
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True

    def render_header(self):
        """Render the dashboard header."""
        st.title(self.title)
        st.markdown(f"*{self.description}*")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Active Jobs", len([j for j in st.session_state.jobs.values() if j.get('status') in ['pending', 'running']]))

        with col2:
            completed_jobs = len([j for j in st.session_state.jobs.values() if j.get('status') == 'completed'])
            st.metric("Completed Jobs", completed_jobs)

        with col3:
            failed_jobs = len([j for j in st.session_state.jobs.values() if j.get('status') == 'failed'])
            st.metric("Failed Jobs", failed_jobs, delta=f"{failed_jobs} errors" if failed_jobs > 0 else "No errors")

    def render_sidebar(self):
        """Render the sidebar with navigation and controls."""
        st.sidebar.title("🎛️ Controls")

        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Auto-refresh",
            value=st.session_state.auto_refresh,
            help="Automatically refresh job status and metrics"
        )

        # Refresh interval
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=30,
            value=5,
            help="How often to refresh the dashboard"
        )

        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Quick Actions")

        if st.sidebar.button("🆕 New Pipeline Job", use_container_width=True):
            st.session_state.current_view = "new_job"

        if st.sidebar.button("📊 Job Status", use_container_width=True):
            st.session_state.current_view = "jobs"

        if st.sidebar.button("📈 Performance Metrics", use_container_width=True):
            st.session_state.current_view = "metrics"

        if st.sidebar.button("⚙️ Configuration", use_container_width=True):
            st.session_state.current_view = "config"

        if st.sidebar.button("📚 Documentation", use_container_width=True):
            st.session_state.current_view = "docs"

        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

    def render_main_content(self):
        """Render the main content area."""
        # Determine current view
        current_view = getattr(st.session_state, 'current_view', 'overview')

        if current_view == "new_job":
            self.render_job_submission()
        elif current_view == "jobs":
            self.render_job_monitoring()
        elif current_view == "metrics":
            self.render_performance_metrics()
        elif current_view == "config":
            self.render_configuration_manager()
        elif current_view == "docs":
            self.render_documentation()
        else:
            self.render_overview()

    def render_overview(self):
        """Render the overview/dashboard home page."""
        st.header("📊 Pipeline Overview")

        # Recent jobs summary
        st.subheader("Recent Jobs")

        if not st.session_state.jobs:
            st.info("No jobs have been submitted yet. Use 'New Pipeline Job' to get started.")
            return

        # Job status summary
        status_counts = {}
        for job in st.session_state.jobs.values():
            status = job.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        # Create status chart
        if status_counts:
            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Job Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recent jobs table
        st.subheader("Recent Jobs")

        job_data = []
        for job_id, job in st.session_state.jobs.items():
            job_data.append({
                'Job ID': job_id[:8] + "...",
                'Status': job.get('status', 'unknown').upper(),
                'Run ID': job.get('run_id', 'N/A'),
                'Submitted': job.get('start_time', 'N/A'),
                'Duration': self._format_duration(job.get('start_time'), job.get('end_time'))
            })

        if job_data:
            df = pd.DataFrame(job_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    def render_job_submission(self):
        """Render the job submission interface."""
        st.header("🚀 Submit New Pipeline Job")

        # Job configuration form
        with st.form("job_submission_form"):
            col1, col2 = st.columns(2)

            with col1:
                run_id = st.text_input(
                    "Run ID",
                    value=f"run_{int(time.time())}",
                    help="Unique identifier for this pipeline run"
                )

                seed = st.number_input(
                    "Random Seed",
                    min_value=0,
                    value=7,
                    help="Seed for reproducible results"
                )

            with col2:
                use_parallel = st.checkbox(
                    "Enable Parallel Processing",
                    value=True,
                    help="Use multiple CPU cores for faster processing"
                )

                use_ml = st.checkbox(
                    "Enable ML-based Calling",
                    value=True,
                    help="Use machine learning for variant detection"
                )

                use_dl = st.checkbox(
                    "Enable Deep Learning",
                    value=False,
                    help="Use deep neural networks for enhanced detection"
                )

            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                ml_model_type = st.selectbox(
                    "ML Model Type",
                    options=["ensemble", "xgboost", "lightgbm", "gbm"],
                    value="ensemble",
                    help="Machine learning model architecture"
                )

                dl_model_type = st.selectbox(
                    "Deep Learning Model Type",
                    options=["cnn_lstm", "hybrid", "transformer"],
                    value="cnn_lstm",
                    help="Deep learning model architecture"
                )

            # Submit button
            submitted = st.form_submit_button(
                "🚀 Submit Job",
                use_container_width=True,
                type="primary"
            )

            if submitted:
                # Create job request
                config_request = {
                    "run_id": run_id,
                    "seed": seed,
                    "use_parallel": use_parallel,
                    "use_ml_calling": use_ml,
                    "use_deep_learning": use_dl,
                    "ml_model_type": ml_model_type,
                    "dl_model_type": dl_model_type
                }

                # Submit job (this would integrate with the API)
                job_id = self._submit_job(config_request)

                st.success(f"✅ Job submitted successfully! Job ID: `{job_id}`")
                st.session_state.current_view = "jobs"

                # Auto-refresh to show the new job
                time.sleep(1)
                st.rerun()

    def render_job_monitoring(self):
        """Render the job monitoring interface."""
        st.header("📊 Job Monitoring")

        if not st.session_state.jobs:
            st.info("No jobs found. Submit a new job to get started.")
            return

        # Job selection
        job_options = {job_id: f"{job.get('run_id', 'Unknown')} ({job.get('status', 'unknown').upper()})"
                      for job_id, job in st.session_state.jobs.items()}

        selected_job_id = st.selectbox(
            "Select Job",
            options=list(job_options.keys()),
            format_func=lambda x: job_options[x],
            key="job_selector"
        )

        if selected_job_id:
            job = st.session_state.jobs[selected_job_id]

            # Job details
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Status", job.get('status', 'unknown').upper())

            with col2:
                progress = job.get('progress', 0)
                st.metric("Progress", f"{progress:.1f}%")

            with col3:
                duration = self._format_duration(job.get('start_time'), job.get('end_time'))
                st.metric("Duration", duration)

            # Progress bar
            st.progress(progress / 100)

            # Job details
            with st.expander("📋 Job Details", expanded=True):
                st.json(job)

            # Results (if completed)
            if job.get('status') == 'completed' and job.get('results'):
                st.subheader("📈 Results")

                results = job['results']

                # Metrics display
                if 'metrics' in results:
                    metrics = results['metrics']

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                    with metric_col1:
                        st.metric("Samples Processed", metrics.get('n_samples', 0))

                    with metric_col2:
                        st.metric("Variants Detected", metrics.get('n_variants', 0))

                    with metric_col3:
                        st.metric("Detection Rate", f"{metrics.get('variant_rate', 0):.3f}")

                    with metric_col4:
                        st.metric("Mean AF", f"{metrics.get('mean_allele_fraction', 0):.4f}")

                # Artifacts
                if 'artifacts' in results:
                    st.subheader("📁 Artifacts")

                    artifacts = results['artifacts']

                    for artifact_name, artifact_path in artifacts.items():
                        if Path(artifact_path).exists():
                            if artifact_path.endswith('.html'):
                                st.markdown(f"**{artifact_name.title()}**: [View Report](file://{artifact_path})")
                            elif artifact_path.endswith('.json'):
                                with open(artifact_path, 'r') as f:
                                    data = json.load(f)
                                    st.json(data)
                            elif artifact_path.endswith('.csv'):
                                df = pd.read_csv(artifact_path)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info(f"📄 {artifact_name.title()}: {artifact_path}")

    def render_performance_metrics(self):
        """Render performance metrics and analytics."""
        st.header("📈 Performance Metrics")

        # Overall metrics
        st.subheader("System Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_jobs = len(st.session_state.jobs)
            st.metric("Total Jobs", total_jobs)

        with col2:
            running_jobs = len([j for j in st.session_state.jobs.values() if j.get('status') == 'running'])
            st.metric("Running Jobs", running_jobs)

        with col3:
            avg_processing_time = self._calculate_avg_processing_time()
            st.metric("Avg Processing Time", f"{avg_processing_time:.1f}s" if avg_processing_time else "N/A")

        with col4:
            success_rate = self._calculate_success_rate()
            st.metric("Success Rate", f"{success_rate:.1%}" if success_rate else "N/A")

        # Performance charts
        st.subheader("📊 Performance Trends")

        # Job status over time
        if st.session_state.jobs:
            status_timeline = self._create_status_timeline()
            fig = px.line(
                status_timeline,
                x='timestamp',
                y='count',
                color='status',
                title="Job Status Over Time",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Processing time distribution
        processing_times = self._get_processing_times()
        if processing_times:
            fig = px.histogram(
                processing_times,
                title="Processing Time Distribution",
                labels={'value': 'Processing Time (seconds)'},
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_configuration_manager(self):
        """Render the configuration management interface."""
        st.header("⚙️ Configuration Management")

        tab1, tab2, tab3 = st.tabs(["📋 Current Config", "🔧 Config Builder", "📚 Templates"])

        with tab1:
            self.render_current_config()

        with tab2:
            self.render_config_builder()

        with tab3:
            self.render_config_templates()

    def render_current_config(self):
        """Render current configuration display."""
        st.subheader("Current Pipeline Configuration")

        # Load default config for display
        try:
            config = load_config("configs/smoke.yaml")

            # Configuration summary
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Run Configuration:**")
                st.info(f"Run ID: {config.run_id}")
                st.info(f"Seed: {config.seed}")
                st.info(f"Version: {config.config_version}")

            with col2:
                st.markdown("**Processing Options:**")
                if config.simulation:
                    st.info(f"Simulation: {len(config.simulation.allele_fractions)} AF levels")
                if config.umi:
                    st.info(f"UMI Processing: Min size {config.umi.min_family_size}")
                if config.stats:
                    st.info(f"Statistics: {config.stats.test_type} test")

            # Full configuration
            with st.expander("🔍 Full Configuration (YAML)", expanded=False):
                config_yaml = dump_config(config, None)
                st.code(config_yaml, language='yaml')

        except Exception as e:
            st.error(f"Failed to load configuration: {e}")

    def render_config_builder(self):
        """Render interactive configuration builder."""
        st.subheader("Build Custom Configuration")

        # Basic settings
        st.markdown("### Basic Settings")
        run_id = st.text_input("Run ID", value=f"custom_run_{int(time.time())}")
        seed = st.number_input("Random Seed", min_value=0, value=42)

        # Processing options
        st.markdown("### Processing Options")
        col1, col2 = st.columns(2)

        with col1:
            use_parallel = st.checkbox("Parallel Processing", value=True)
            use_ml = st.checkbox("ML-based Calling", value=True)

        with col2:
            use_dl = st.checkbox("Deep Learning", value=False)
            enable_caching = st.checkbox("Enable Caching", value=True)

        # Model selection (if ML/DL enabled)
        if use_ml or use_dl:
            st.markdown("### Model Selection")

            if use_ml:
                ml_model = st.selectbox(
                    "ML Model Type",
                    options=["ensemble", "xgboost", "lightgbm", "gbm"],
                    value="ensemble"
                )

            if use_dl:
                dl_model = st.selectbox(
                    "Deep Learning Model",
                    options=["cnn_lstm", "hybrid", "transformer"],
                    value="cnn_lstm"
                )

        # Generate configuration
        if st.button("🎯 Generate Configuration", type="primary"):
            # Build configuration based on selections
            config_dict = {
                'run_id': run_id,
                'seed': seed,
                'umi': {
                    'min_family_size': 3,
                    'max_family_size': 1000,
                    'quality_threshold': 20,
                    'consensus_threshold': 0.6
                },
                'stats': {
                    'test_type': 'poisson',
                    'alpha': 0.05,
                    'fdr_method': 'benjamini_hochberg'
                },
                'lod': {
                    'detection_threshold': 0.95,
                    'confidence_level': 0.95
                },
                'simulation': {
                    'allele_fractions': [0.01, 0.001, 0.0001],
                    'umi_depths': [1000, 5000],
                    'n_replicates': 10,
                    'n_bootstrap': 100
                }
            }

            # Create and validate configuration
            config = PipelineConfig(**config_dict)
            validation_result = ConfigValidator.validate_config(config)

            if validation_result['is_valid']:
                st.success("✅ Configuration is valid!")

                # Show configuration preview
                with st.expander("📋 Configuration Preview", expanded=True):
                    st.json(config_dict)

                # Download configuration
                config_yaml = dump_config(config, None)
                st.download_button(
                    label="💾 Download Configuration",
                    data=config_yaml,
                    file_name=f"{run_id}_config.yaml",
                    mime="text/yaml",
                    use_container_width=True
                )

            else:
                st.error("❌ Configuration has issues:")
                for issue in validation_result['issues']:
                    st.error(f"• {issue}")

    def render_config_templates(self):
        """Render configuration templates interface."""
        st.subheader("Configuration Templates")

        templates = [
            PredefinedTemplates.get_smoke_test_template(),
            PredefinedTemplates.get_production_template()
        ]

        for template in templates:
            with st.expander(f"🏷️ {template['template_name']} - {template['description']}"):
                st.markdown(f"**Tags:** {', '.join(template['tags'])}")
                st.markdown(f"**Version:** {template['version']}")

                if st.button(f"📋 Use {template['template_name']} Template", key=f"use_{template['template_name']}"):
                    # Create configuration from template
                    config = PipelineConfig.from_template(template, f"from_{template['template_name']}")

                    # Show configuration
                    st.json(config.to_dict())

                    # Download option
                    config_yaml = dump_config(config, None)
                    st.download_button(
                        label="💾 Download Template Configuration",
                        data=config_yaml,
                        file_name=f"{template['template_name']}_config.yaml",
                        mime="text/yaml",
                        key=f"download_{template['template_name']}"
                    )

    def render_documentation(self):
        """Render documentation and help."""
        st.header("📚 Documentation")

        st.markdown("""
        ## Welcome to the Precise MRD Pipeline Dashboard

        This interactive dashboard provides a user-friendly interface for:

        ### 🚀 **Pipeline Execution**
        - Submit new MRD analysis jobs with custom configurations
        - Monitor job progress in real-time
        - Download results and artifacts

        ### 📊 **Performance Monitoring**
        - View system performance metrics
        - Track job completion rates and processing times
        - Monitor resource utilization

        ### ⚙️ **Configuration Management**
        - Create custom pipeline configurations
        - Use predefined templates for common use cases
        - Validate configurations before execution

        ### 🔬 **Features**
        - **Parallel Processing**: Multi-core execution for faster analysis
        - **Machine Learning**: Enhanced variant detection with ensemble models
        - **Deep Learning**: CNN-LSTM architectures for sequence analysis
        - **Cloud-Native**: Kubernetes deployment with auto-scaling
        - **REST API**: Integration with laboratory information systems

        ### 🎯 **Getting Started**

        1. **Submit a Job**: Use "New Pipeline Job" to start an analysis
        2. **Monitor Progress**: Check "Job Status" for real-time updates
        3. **View Results**: Download reports and data files
        4. **Customize**: Use "Configuration" to create custom setups

        ### 📞 **Support**

        For technical support or feature requests, please contact the development team.
        """)

        # API endpoints documentation
        with st.expander("🔌 API Endpoints"):
            st.markdown("""
            **Core Endpoints:**
            - `POST /submit` - Submit pipeline job
            - `GET /status/{job_id}` - Check job status
            - `GET /results/{job_id}` - Get job results
            - `GET /health` - Health check

            **Configuration:**
            - `POST /validate-config` - Validate configuration
            - `GET /config-templates` - Get available templates
            - `POST /config-from-template` - Create config from template
            """)

    def _submit_job(self, config_request: Dict[str, Any]) -> str:
        """Submit a job (placeholder implementation)."""
        job_id = str(uuid.uuid4())

        # Create job entry
        st.session_state.jobs[job_id] = {
            'job_id': job_id,
            'status': 'pending',
            'progress': 0.0,
            'start_time': time.time(),
            'config_request': config_request
        }

        return job_id

    def _format_duration(self, start_time, end_time) -> str:
        """Format duration for display."""
        if not start_time:
            return "N/A"

        start = start_time if isinstance(start_time, float) else time.time()

        if end_time:
            end = end_time if isinstance(end_time, float) else time.time()
            duration = end - start
        else:
            duration = time.time() - start

        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"

    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time for completed jobs."""
        completed_jobs = [j for j in st.session_state.jobs.values() if j.get('status') == 'completed']

        if not completed_jobs:
            return 0.0

        total_time = 0
        for job in completed_jobs:
            start_time = job.get('start_time')
            end_time = job.get('end_time')

            if start_time and end_time:
                total_time += end_time - start_time

        return total_time / len(completed_jobs)

    def _calculate_success_rate(self) -> float:
        """Calculate success rate for completed jobs."""
        completed_jobs = [j for j in st.session_state.jobs.values() if j.get('status') in ['completed', 'failed']]

        if not completed_jobs:
            return 0.0

        successful_jobs = len([j for j in completed_jobs if j.get('status') == 'completed'])
        return successful_jobs / len(completed_jobs)

    def _create_status_timeline(self) -> pd.DataFrame:
        """Create status timeline for visualization."""
        timeline_data = []

        for job_id, job in st.session_state.jobs.items():
            if job.get('start_time'):
                timeline_data.append({
                    'timestamp': job['start_time'],
                    'status': job.get('status', 'unknown'),
                    'job_id': job_id[:8]
                })

        if not timeline_data:
            return pd.DataFrame()

        df = pd.DataFrame(timeline_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Group by time window and status
        df['time_window'] = df['timestamp'].dt.floor('1H')
        status_counts = df.groupby(['time_window', 'status']).size().reset_index(name='count')

        return status_counts

    def _get_processing_times(self) -> List[float]:
        """Get processing times for all completed jobs."""
        times = []

        for job in st.session_state.jobs.values():
            if job.get('status') == 'completed':
                start_time = job.get('start_time')
                end_time = job.get('end_time')

                if start_time and end_time:
                    times.append(end_time - start_time)

        return times


def run_dashboard():
    """Run the Streamlit dashboard."""
    dashboard = MRDDashboard()

    # Render layout
    dashboard.render_header()
    dashboard.render_sidebar()
    dashboard.render_main_content()


if __name__ == "__main__":
    run_dashboard()



