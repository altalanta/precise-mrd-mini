#!/usr/bin/env python3
"""Script to run the MRD Pipeline Dashboard with API integration."""

import time
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
WS_BASE_URL = "ws://127.0.0.1:8000"
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
def get_jobs() -> list[dict[str, Any]]:
    """Fetch all recent jobs from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/jobs?limit=100")
        response.raise_for_status()
        return response.json().get("jobs", [])
    except requests.RequestException as e:
        st.error(f"Failed to fetch jobs: {e}")
        return []


@st.cache_data(ttl=5)
def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Fetch the status of a specific job."""
    try:
        response = requests.get(f"{API_BASE_URL}/status/{job_id}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to fetch status for job {job_id}: {e}")
        return None


def submit_job(
    run_id: str, seed: int, use_parallel: bool, use_ml: bool, use_dl: bool
) -> dict[str, Any] | None:
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
        st.error(f"Failed to submit job: {e.response.text if e.response else e}")
        return None


# --- UI Components ---


def websocket_updater(job_id: str):
    """Injects a JavaScript component to listen for WebSocket updates."""

    # We use a placeholder to inject the component and then update it.
    # This is a bit of a hack to force Streamlit to re-render the component
    # when the job_id changes.
    st.session_state["ws_placeholder"] = st.empty()

    with st.session_state["ws_placeholder"]:
        components.html(
            f"""
            <div id="status-container" style="display: none;"></div>
            <script>
                const jobId = "{job_id}";
                const wsUrl = `{WS_BASE_URL}/ws/status/${{jobId}}`;
                let socket = new WebSocket(wsUrl);

                socket.onopen = function(event) {{
                    console.log("[open] Connection established");
                    console.log("Sending to server");
                    socket.send("STATUS_REQUEST");
                }};

                socket.onmessage = function(event) {{
                    try {{
                    console.log(`[message] Data received from server: ${{event.data}}`);
                    const data = JSON.parse(event.data);
                    
                    // Trigger a re-run in Streamlit by setting a query param.
                    // This is a workaround to update the Python-side state.
                    const currentUrl = new URL(window.location.href);
                    if (currentUrl.searchParams.get("job_id") !== data.job_id ||
                        currentUrl.searchParams.get("status") !== data.status ||
                        currentUrl.searchParams.get("progress") !== data.progress) {{
                        
                        currentUrl.searchParams.set("job_id", data.job_id);
                        currentUrl.searchParams.set("status", data.status);
                            currentUrl.searchParams.set("progress", data.progress);
                        window.location.href = currentUrl.href;
                        }}

                        if (data.status === "completed" || data.status === "failed") {{
                            console.log("Job finished, closing WebSocket.");
                            socket.close();
                        }}
                    }} catch (e) {{
                        console.error("Failed to parse WebSocket message:", e);
                    }}
                }};

                socket.onclose = function(event) {{
                    if (event.wasClean) {{
                        console.log(`[close] Connection closed cleanly, code=${{event.code}} reason=${{event.reason}}`);
                    }} else {{
                        console.log('[close] Connection died');
                    }}
                }};

                socket.onerror = function(error) {{
                    console.log(`[error] ${{error.message}}`);
                }};
            </script>
        """,
            height=0,
        )


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
                            st.success(
                                f"Job '{result['job_id']}' submitted successfully!"
                            )
                            st.session_state.selected_job_id = result["job_id"]
                            st.rerun()


def render_job_list():
    """Render the main table of jobs."""
    st.header("📊 Job Monitor")

    jobs_data = get_jobs()

    if not jobs_data:
        st.info("No jobs found. Submit a new job from the sidebar to get started.")
        return

    df = pd.DataFrame(jobs_data)
    df["run_id"] = df.apply(
        lambda row: row.get("run_id") or f"Job_{row['job_id']}", axis=1
    )
    df = df[["job_id", "status", "progress", "run_id", "created_at"]]
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values("created_at", ascending=False).reset_index(drop=True)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Automatically select the latest job if none is selected
    if "selected_job_id" not in st.session_state and not df.empty:
        st.session_state.selected_job_id = df.iloc[0]["job_id"]

    st.selectbox(
        "Select Job to View Details:",
        options=df["job_id"],
        key="selected_job_id",
        index=0 if not df.empty else None,
        label_visibility="collapsed",
    )


def render_job_details():
    """Render the detailed view for the selected job."""
    job_id = st.session_state.get("selected_job_id")
    if not job_id:
        return

    st.header(f"Details for Job: `{job_id}`")

    # --- Real-time updater ---
    websocket_updater(job_id)

    status_data = get_job_status(job_id)

    if not status_data:
        st.warning("Could not load job details. The job may have expired or failed.")
        return

    # --- Status & Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status_data["status"].upper())
    progress_float = float(status_data.get("progress", 0.0))
    col2.metric("Progress", f"{progress_float:.1%}")
    if status_data.get("start_time"):
        start = pd.to_datetime(status_data["start_time"])
        end = (
            pd.to_datetime(status_data.get("end_time"))
            if status_data.get("end_time")
            else pd.Timestamp.now(tz="UTC")
        )
        duration = end - start
        col3.metric("Duration", f"{duration.seconds}s")

    # --- Progress Bar ---
    st.progress(progress_float, text=f"Job is {status_data['status']}...")

    # --- Results ---
    if (
        status_data["status"] == "completed"
        and "results" in status_data
        and status_data["results"]
    ):
        st.subheader("✅ Results")
        results = status_data["results"]
        st.json(results)

    elif (
        status_data["status"] == "failed"
        and "results" in status_data
        and status_data["results"]
    ):
        st.error("Job failed:")
        st.json(status_data["results"])

    elif status_data["status"] not in ["completed", "failed"]:
        st.info(
            "Job is running. Status will update in real-time. Results will be shown upon completion."
        )


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

    # The auto-refresh polling is now replaced by WebSockets
    # time.sleep(5)
    # st.rerun()


if __name__ == "__main__":
    main()
