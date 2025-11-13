#!/usr/bin/env python3
"""Script to run the MRD Pipeline Dashboard with API integration."""

import streamlit as st
import pandas as pd
import requests
import time
from pathlib import Path
import yaml
from typing import Dict, Any, List, Optional

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
st.set_page_config(
    page_title="Precise MRD Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Communication Layer ---

def get_api_health() -> bool:
    """Check if the API is reachable."""
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

@st.cache_data(ttl=5)
def get_jobs() -> List[Dict[str, Any]]:
    """Fetch all recent jobs from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/jobs?limit=100")
        response.raise_for_status()
        return response.json().get("jobs", [])
    except requests.RequestException as e:
        st.error(f"Failed to fetch jobs: {e}")
        return []

@st.cache_data(ttl=5)
def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the status of a specific job."""
    try:
        response = requests.get(f"{API_BASE_URL}/status/{job_id}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch status for job {job_id}: {e}")
        return None
        
def submit_job(run_id: str, seed: int, use_parallel: bool, use_ml: bool, use_dl: bool) -> Optional[Dict[str, Any]]:
    """Submit a new job to the API."""
    form_data = {
        "run_id": run_id,
        "seed": seed,
        "use_parallel": use_parallel,
        "use_ml_calling": use_ml,
        "use_deep_learning": use_dl,
    }
    try:
        response = requests.post(f"{API_BASE_URL}/submit", data=form_data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to submit job: {e.content.decode()}")
        return None

# --- UI Components ---

def render_sidebar():
    """Render the sidebar for job submission."""
    with st.sidebar:
        st.header("🔬 Submit New Job")
        
        with st.form("job_submission_form"):
            run_id = st.text_input("Run ID", f"dashboard-run-{int(time.time())}")
            seed = st.number_input("Random Seed", value=42, min_value=0)
            
            st.write("**Configuration Options**")
            use_parallel = st.checkbox("Enable Parallel Processing", value=True)
            use_ml = st.checkbox("Use Machine Learning Caller")
            use_dl = st.checkbox("Use Deep Learning Caller")
            
            submitted = st.form_submit_button("🚀 Launch Run", use_container_width=True)
            
            if submitted:
                if not run_id:
                    st.warning("Run ID is required.")
                else:
                    with st.spinner("Submitting job..."):
                        result = submit_job(run_id, seed, use_parallel, use_ml, use_dl)
                        if result:
                            st.success(f"Job '{result['job_id']}' submitted successfully!")
                            st.session_state.selected_job_id = result['job_id']
                            st.rerun()

def render_job_list():
    """Render the main table of jobs."""
    st.header("📊 Job Monitor")
    
    jobs_data = get_jobs()
    
    if not jobs_data:
        st.info("No jobs found. Submit a new job from the sidebar to get started.")
        return

    df = pd.DataFrame(jobs_data)
    df = df[['job_id', 'status', 'progress', 'run_id', 'created_at']]
    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.sort_values('created_at', ascending=False)
    
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Automatically select the latest job if none is selected
    if 'selected_job_id' not in st.session_state and not df.empty:
        st.session_state.selected_job_id = df.iloc[0]['job_id']

    st.selectbox(
        "Select Job to View Details:",
        options=df['job_id'],
        key='selected_job_id',
        index=0 if not df.empty else None,
        label_visibility="collapsed"
    )

def render_job_details():
    """Render the detailed view for the selected job."""
    job_id = st.session_state.get('selected_job_id')
    if not job_id:
        return

    st.header(f"Details for Job: `{job_id}`")
    status_data = get_job_status(job_id)

    if not status_data:
        st.warning("Could not load job details. The job may have expired or failed.")
        return

    # --- Status & Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status_data['status'].upper())
    col2.metric("Progress", f"{status_data['progress']:.1%}")
    if status_data['start_time']:
        start = pd.to_datetime(status_data['start_time'])
        end = pd.to_datetime(status_data['end_time']) if status_data['end_time'] else pd.Timestamp.now(tz='UTC')
        duration = end - start
        col3.metric("Duration", f"{duration.seconds}s")

    # --- Progress Bar ---
    st.progress(status_data['progress'], text=f"Job is {status_data['status']}...")

    # --- Results ---
    if status_data['status'] == 'completed' and status_data['results']:
        st.subheader("✅ Results")
        results = status_data['results']
        
        # Display key metrics
        if 'metrics' in results:
            st.write("**Key Metrics:**")
            metrics = results['metrics']['overall']
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            m_col2.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            m_col3.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
            m_col4.metric("Specificity", f"{metrics.get('specificity', 0):.4f}")

        # Display run context
        if 'run_context' in results:
            with st.expander("Run Context & Configuration"):
                st.json(results['run_context'])

        # Display artifacts and download links
        if 'artifacts' in results:
            st.write("**Artifacts:**")
            artifacts_df = pd.DataFrame.from_dict(results['artifacts'], orient='index', columns=['Path'])
            artifacts_df['Download'] = artifacts_df.index.map(
                lambda x: f"[Link]({API_BASE_URL}/download/{job_id}/{x})"
            )
            st.dataframe(artifacts_df, use_container_width=True)

    elif status_data['status'] == 'failed':
        st.error(f"Job failed: {status_data['error']}")
    
    else:
        st.info("Job is still running. Results will be displayed here upon completion.")
        
# --- Main Application ---

def main():
    """Main function to run the Streamlit dashboard."""
    st.title("🧬 Precise MRD Pipeline Dashboard")

    if not get_api_health():
        st.error(
            "**API is not available.** Please ensure the FastAPI server is running."
            "\n\nYou can start it by running `uv run src.precise_mrd.api:create_api_app --factory --reload` in your terminal."
        )
        return

    render_sidebar()
    
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        render_job_list()

    with col2:
        render_job_details()

    # Auto-refresh the page periodically
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()














