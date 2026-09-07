"""Tenarai professional Streamlit frontend for the backend API.

Run the API first, then the UI:
    uvicorn src.api.app:app --reload --port 8000
    streamlit run src/app/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from src.app.api_client import ApiError, TenaraiApiClient
from src.nlq.llm_client import DEFAULT_MODEL

DEFAULT_API_URL = os.environ.get("TENARAI_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Tenarai | Logistics intelligence",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _initialize_state() -> None:
    defaults = {"messages": [], "last_upload": None, "suggested_question": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_result_details(message: dict) -> None:
    rows = message.get("rows", [])
    if rows:
        frame = pd.DataFrame(rows, columns=message.get("columns"))
        st.dataframe(frame, hide_index=True, width="stretch")

        numeric_columns = list(frame.select_dtypes(include="number").columns)
        if len(frame) > 1 and numeric_columns and len(frame.columns) >= 2:
            label_column = frame.columns[0]
            try:
                chart_frame = frame[[label_column, *numeric_columns]].set_index(label_column)
                st.bar_chart(chart_frame)
            except (TypeError, ValueError, KeyError):
                pass

    with st.expander("Query evidence", icon=":material/database:"):
        if message.get("sql_explanation"):
            st.caption(message["sql_explanation"])
        st.code(message.get("sql", ""), language="sql")
        st.caption(
            f"{message.get('row_count', 0):,} rows returned · request {message.get('request_id', 'unknown')}"
        )

    with st.expander("Retrieved business context", icon=":material/manage_search:"):
        context = message.get("retrieved_context", [])
        if not context:
            st.caption("No retrieval metadata was returned.")
        for item in context:
            st.markdown(f"**{item['title']}**  \n{item['content']}")
            st.caption(f"Document `{item['document_id']}` · relevance {item['score']}")

    if message.get("synthesis_warning"):
        st.warning(
            "The query succeeded, but the natural-language synthesis used a fallback. "
            f"Backend detail: {message['synthesis_warning']}"
        )


def _render_chat_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sql"):
            _render_result_details(message)


_initialize_state()

with st.sidebar:
    st.title("TENARAI")
    st.caption("Logistics intelligence workspace")
    st.space("small")
    api_url = st.text_input("Backend URL", value=DEFAULT_API_URL, help="FastAPI service base URL")
    admin_key = st.text_input(
        "Admin key",
        type="password",
        help="Required for uploads and backend logs when the deployed API is protected.",
    )

    client = TenaraiApiClient(api_url, admin_key=admin_key)
    try:
        health = client.health()
        st.success("Backend connected", icon=":material/check_circle:")
        st.caption("Active dataset is query-ready" if health.get("dataset_ready") else "Upload CSVs to create the active dataset")
        provider = health.get("llm_provider", "unknown")
        model_name = st.text_input(
            "Language model",
            value=health.get("model", DEFAULT_MODEL),
            disabled=not health.get("allow_model_override", True),
            help=f"Backend provider: {provider}",
        )
        backend_online = True
    except ApiError:
        st.error("Backend offline", icon=":material/cloud_off:")
        st.caption("Start `uvicorn src.api.app:app --reload --port 8000`")
        health = {"admin_protected": False, "allow_model_override": True}
        model_name = st.text_input("Language model", value=DEFAULT_MODEL)
        backend_online = False

    st.space("small")
    st.caption("Secure, local-first analytics · v2.0")

st.title("Logistics intelligence", icon=":material/query_stats:")
st.caption("Ask operational questions, inspect the evidence, and manage shipment data from one workspace.")

dataset = {"ready": False, "row_count": 0, "source_count": 0, "delay_rate_pct": None}
if backend_online:
    try:
        dataset = client.dataset()
    except ApiError as exc:
        st.warning(str(exc))

metric_columns = st.columns(4)
metric_columns[0].metric("Active shipments", f"{dataset.get('row_count', 0):,}")
metric_columns[1].metric("Source files", f"{dataset.get('source_count', 0):,}")
date_range = "—"
if dataset.get("min_ship_date") and dataset.get("max_ship_date"):
    date_range = f"{dataset['min_ship_date']} → {dataset['max_ship_date']}"
metric_columns[2].metric("Coverage", date_range)
delay_rate = dataset.get("delay_rate_pct")
metric_columns[3].metric("Delay rate", "—" if delay_rate is None else f"{delay_rate:.1f}%")

ask_tab, data_tab, rag_tab, logs_tab = st.tabs(
    [
        ":material/chat: Ask Tenarai",
        ":material/upload_file: Data workspace",
        ":material/account_tree: RAG & governance",
        ":material/monitor_heart: Backend monitor",
    ]
)

with ask_tab:
    st.subheader("Operational copilot", icon=":material/neurology:")
    st.caption("Every answer is grounded in the active dataset and includes its generated SQL and retrieved definitions.")

    if not st.session_state.messages:
        suggestions = {
            "Worst-performing routes": "Which routes have the highest delay rate?",
            "Carrier scorecard": "Compare carriers by shipment count, average delay days, and delay rate.",
            "Recent trend": "Show the monthly shipment volume and delay rate over the latest six months in the data.",
        }
        suggestion = st.pills("Suggested questions", list(suggestions), label_visibility="collapsed")
        if suggestion:
            st.session_state.suggested_question = suggestions[suggestion]

    for stored_message in st.session_state.messages:
        _render_chat_message(stored_message)

    typed_question = st.chat_input(
        "Ask about routes, carriers, delays, status, or data quality",
        disabled=not backend_online or not dataset.get("ready", False),
    )
    question = typed_question or st.session_state.pop("suggested_question", None)

    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        _render_chat_message(user_message)

        with st.chat_message("assistant"):
            with st.status("Grounding the question and querying DuckDB…", expanded=True) as status:
                st.write("Retrieving relevant business definitions")
                st.write("Generating and validating read-only SQL")
                st.write("Synthesizing the result into a natural answer")
                try:
                    response = client.chat(question, model_name)
                    status.update(label="Analysis complete", state="complete", expanded=False)
                except ApiError as exc:
                    status.update(label="Analysis failed", state="error", expanded=True)
                    st.error(str(exc))
                    response = None

            if response:
                assistant_message = {"role": "assistant", "content": response["answer"], **response}
                st.markdown(assistant_message["content"])
                _render_result_details(assistant_message)
                st.session_state.messages.append(assistant_message)

with data_tab:
    st.subheader("Data workspace", icon=":material/database_upload:")
    st.caption("Upload up to 20 CSV exports together. A successful batch atomically becomes the active DuckDB dataset.")
    if health.get("admin_protected", False) and not admin_key:
        st.info("Enter the admin key in the sidebar to enable dataset uploads.")

    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Shipment CSV files",
            type=["csv"],
            accept_multiple_files=True,
            max_upload_size=25,
            help="Each file can be up to 25 MB. Headers are reconciled across legacy source formats.",
        )

        if uploaded_files:
            upload_manifest = pd.DataFrame(
                {
                    "File": [item.name for item in uploaded_files],
                    "Size (KB)": [round(item.size / 1024, 1) for item in uploaded_files],
                }
            )
            st.dataframe(upload_manifest, hide_index=True, width="stretch")

        if st.button(
            "Process upload batch",
            type="primary",
            icon=":material/cloud_upload:",
            disabled=(
                not backend_online
                or not uploaded_files
                or (health.get("admin_protected", False) and not admin_key)
            ),
        ):
            with st.status("Validating and loading files…", expanded=True) as status:
                try:
                    result = client.upload([(item.name, item.getvalue()) for item in uploaded_files])
                    st.session_state.last_upload = result
                    st.session_state.messages = []
                    status.update(label="Dataset activated", state="complete", expanded=False)
                    st.toast("Upload batch processed successfully", icon=":material/check_circle:")
                except ApiError as exc:
                    status.update(label="Upload failed", state="error", expanded=True)
                    st.error(str(exc))

    upload_result = st.session_state.last_upload
    if upload_result:
        st.subheader("Latest ingestion", icon=":material/fact_check:")
        cols = st.columns(4)
        cols[0].metric("Files loaded", len(upload_result["files_loaded"]))
        cols[1].metric("Rows accepted", f"{upload_result['accepted_rows']:,}")
        cols[2].metric("Rows rejected", f"{upload_result['rejected_rows']:,}")
        cols[3].metric("Duplicates removed", f"{upload_result['duplicate_rows_removed']:,}")
        st.caption(f"Batch `{upload_result['batch_id']}` · {upload_result['duration_ms']:.0f} ms")
        with st.expander("Full quality report", icon=":material/description:"):
            st.json(upload_result)

with rag_tab:
    st.subheader("How retrieval-grounded Text-to-SQL works", icon=":material/account_tree:")
    if backend_online:
        try:
            rag = client.rag()
            st.caption(rag["strategy"].capitalize())
            for index, step in enumerate(rag["steps"], start=1):
                with st.container(border=True):
                    st.markdown(f"**{index}. {step}**")
            st.info(
                f"The current knowledge base contains {rag['knowledge_documents']} auditable domain documents; "
                f"the top {rag['retrieval_top_k']} are injected per question. {rag['note']}",
                icon=":material/info:",
            )
        except ApiError as exc:
            st.error(str(exc))
    else:
        st.info("Connect the backend to inspect the active RAG configuration.")

with logs_tab:
    st.subheader("Backend monitor", icon=":material/monitor_heart:")
    st.caption("Recent structured events from ingestion, retrieval, SQL validation, execution, synthesis, and HTTP requests.")
    controls = st.container(horizontal=True, vertical_alignment="bottom")
    with controls:
        log_level = st.selectbox("Level", ["All", "INFO", "WARNING", "ERROR", "CRITICAL"])
        log_limit = st.selectbox("Rows", [50, 100, 200, 500], index=2)
        st.button("Refresh", icon=":material/refresh:")

    if backend_online and health.get("admin_protected", False) and not admin_key:
        st.info("Enter the admin key in the sidebar to view operational logs.")
    elif backend_online:
        try:
            log_payload = client.logs(limit=log_limit, level=None if log_level == "All" else log_level)
            records = log_payload["records"]
            if records:
                log_frame = pd.DataFrame(records)
                ordered = [
                    column
                    for column in ["timestamp", "level", "event", "message", "duration_ms", "request_id", "details"]
                    if column in log_frame
                ]
                st.dataframe(log_frame[ordered], hide_index=True, width="stretch")
            else:
                st.info("No matching backend log records yet.")
        except ApiError as exc:
            st.error(str(exc))
    else:
        st.info("Connect the backend to monitor service logs.")
