#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tourism Guide RAG Q&A - Streamlit App

Usage:
    streamlit run gemini/main_qa.py
"""

import os
import sys
import time
from datetime import datetime

import streamlit as st

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

import google.genai as genai
from google.genai import types

from gemini.config import GeminiConfig
from gemini.query_logger import QueryLogger
from gemini.store_registry import StoreRegistry
from gemini.upload_manager import UploadManager
from gemini.upload_tracker import UploadTracker


def load_chunks(chunks_dir: str) -> tuple[str, list[str]]:
    """
    Load all chunk files and combine into context

    Returns:
        Tuple of (combined_context, list_of_chunk_filenames)
    """
    if not os.path.exists(chunks_dir):
        return "", []

    chunks = []
    chunk_files = []

    for root, dirs, files in os.walk(chunks_dir):
        for file in sorted(files):
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        chunks.append(f"=== {file} ===\n{content}\n")
                        chunk_files.append(file)
                except Exception as e:
                    st.warning(f"Could not read {file}: {e}")

    return "\n".join(chunks), chunk_files


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "config" not in st.session_state:
        try:
            st.session_state.config = GeminiConfig.from_yaml()
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            st.stop()

    if "client" not in st.session_state:
        try:
            st.session_state.client = genai.Client(
                api_key=st.session_state.config.api_key
            )
        except Exception as e:
            st.error(f"Failed to connect to Gemini API: {e}")
            st.stop()

    if "registry" not in st.session_state:
        try:
            st.session_state.registry = StoreRegistry(
                st.session_state.config.registry_path
            )
        except Exception as e:
            st.error(f"Failed to load registry: {e}")
            st.stop()

    if "logger" not in st.session_state:
        log_path = os.path.join(
            os.path.dirname(st.session_state.config.registry_path), "query_log.jsonl"
        )
        st.session_state.logger = QueryLogger(
            log_path, area="", site=""
        )  # Will be updated per query

    if "tracker" not in st.session_state:
        st.session_state.tracker = UploadTracker(
            st.session_state.config.upload_tracking_path
        )

    if "upload_manager" not in st.session_state:
        st.session_state.upload_manager = UploadManager(
            st.session_state.config,
            st.session_state.client,
            st.session_state.registry,
            st.session_state.tracker,
        )

    if "selected_area" not in st.session_state:
        all_stores = st.session_state.registry.list_all()
        if all_stores:
            (area, site), _ = list(all_stores.items())[0]
            st.session_state.selected_area = area
            st.session_state.selected_site = site
        else:
            st.session_state.selected_area = None
            st.session_state.selected_site = None

    if "context" not in st.session_state:
        st.session_state.context = ""
        st.session_state.chunk_files = []


def get_response(question: str, area: str, site: str) -> tuple[str, float]:
    """
    Get response from Gemini API with RAG context

    Returns:
        Tuple of (response_text, response_time_seconds)
    """
    config = st.session_state.config
    client = st.session_state.client
    context = st.session_state.context

    model_name = config.model_name
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    system_instruction = f"""You are a helpful tourism guide assistant for the {area} region,
specifically for the {site} area.

Use ONLY the following source material to answer questions. If the answer is not in the source material,
say so honestly. Always respond in the same language as the question.

SOURCE MATERIAL:
{context}

Answer questions based only on this source material."""

    start_time = time.time()

    response = client.models.generate_content(
        model=model_name,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction, temperature=config.temperature
        ),
    )

    response_time = time.time() - start_time

    return response.text, response_time


def main():
    st.set_page_config(
        page_title="Tourism Guide Q&A",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🗺️ Tourism Guide")
        st.markdown("---")

        # Area/Site Selection
        st.subheader("📍 Location")

        all_stores = st.session_state.registry.list_all()
        if not all_stores:
            st.error("No stores found in registry. Run upload first.")
            st.stop()

        # Create dropdown options
        location_options = {
            f"{area} / {site}": (area, site) for (area, site) in all_stores.keys()
        }
        selected_location = st.selectbox(
            "Select Area / Site",
            options=list(location_options.keys()),
            index=0,
        )

        # Update selected area/site
        area, site = location_options[selected_location]
        if (
            area != st.session_state.selected_area
            or site != st.session_state.selected_site
        ):
            st.session_state.selected_area = area
            st.session_state.selected_site = site
            st.session_state.messages = []  # Clear chat history on location change

            # Load chunks for selected location
            chunks_dir = os.path.join(st.session_state.config.chunks_dir, area, site)
            context, chunk_files = load_chunks(chunks_dir)
            st.session_state.context = context
            st.session_state.chunk_files = chunk_files

        # Display location info
        store_id = st.session_state.registry.get_store(area, site)
        registry_data = st.session_state.registry.registry.get(f"{area}:{site}", {})
        metadata = registry_data.get("metadata", {})

        st.info(
            f"""
            **Area:** {area}
            **Site:** {site}
            **Documents:** {metadata.get('file_count', 'N/A')}
            **Chunks:** {len(st.session_state.chunk_files)}
            """
        )

        st.markdown("---")

        # Settings
        st.subheader("⚙️ Settings")
        config = st.session_state.config

        # Model selection
        model_options = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
            "gemini-2.5-flash",
        ]
        current_model = config.model_name.replace("models/", "")
        selected_model = st.selectbox(
            "Model",
            options=model_options,
            index=(
                model_options.index(current_model)
                if current_model in model_options
                else 0
            ),
        )
        config.model_name = selected_model

        # Temperature
        config.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=config.temperature,
            step=0.1,
        )

        st.markdown("---")

        # Statistics
        if st.button("📊 Show Statistics"):
            stats = st.session_state.logger.get_stats()
            st.write("**Query Statistics**")
            st.metric("Total Queries", stats["total_queries"])
            st.metric("Avg Response Time", f"{stats['avg_response_time_seconds']:.2f}s")

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Main content area
    st.title("🗺️ Tourism Guide Q&A")
    st.markdown(f"**Current Location:** {area} / {site}")

    # Create tabs
    tab_chat, tab_manage = st.tabs(["💬 Chat", "⚙️ Manage Content"])

    # ===== CHAT TAB =====
    with tab_chat:
        # Check if context is loaded
        if not st.session_state.context:
            st.warning(
                f"⚠️ No content found for {area} / {site}. Please upload documents first."
            )

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "time" in message:
                    st.caption(f"⏱️ {message['time']:.2f}s")

        # Chat input
        if question := st.chat_input("Ask a question about this location..."):
            # Display user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching content..."):
                    try:
                        answer, response_time = get_response(question, area, site)

                        st.markdown(answer)
                        st.caption(f"⏱️ {response_time:.2f}s")

                        # Save to messages
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "time": response_time,
                            }
                        )

                        # Log the query
                        st.session_state.logger.area = area
                        st.session_state.logger.site = site
                        st.session_state.logger.log_query(
                            query=question,
                            answer=answer,
                            model=config.model_name,
                            context_chars=len(st.session_state.context),
                            response_time_seconds=response_time,
                            chunks_used=st.session_state.chunk_files,
                        )

                    except Exception as e:
                        st.error(f"Error: {e}")

    # ===== MANAGE CONTENT TAB =====
    with tab_manage:
        st.subheader("📂 Uploaded Content")

        # View uploaded content
        summary = st.session_state.upload_manager.get_uploaded_content_summary()

        if not summary:
            st.info("No content uploaded yet.")
        else:
            # Display as dataframe
            import pandas as pd

            df = pd.DataFrame(summary)
            st.dataframe(
                df,
                column_config={
                    "area": "Area",
                    "site": "Site",
                    "store_id": "Store ID",
                    "file_count": "Files",
                    "chunk_count": "Chunks",
                    "created_at": "Created",
                    "last_updated": "Updated",
                },
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("---")

        # Remove content section
        st.subheader("🗑️ Remove Content")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            remove_area = st.selectbox(
                "Select Area to Remove",
                options=[s["area"] for s in summary],
                key="remove_area",
            )

        with col2:
            # Filter sites for selected area
            available_sites = [s["site"] for s in summary if s["area"] == remove_area]
            remove_site = st.selectbox(
                "Select Site to Remove", options=available_sites, key="remove_site"
            )

        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("🗑️ Remove", type="primary"):
                with st.spinner("Removing content..."):
                    success, message = st.session_state.upload_manager.remove_location(
                        remove_area, remove_site
                    )
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        st.markdown("---")

        # Upload new content section
        st.subheader("📤 Upload Content")

        # Get available locations from content directory
        available_locations = st.session_state.upload_manager.get_available_locations()

        if not available_locations:
            st.warning(
                f"No content found in {st.session_state.config.content_root}. Add files first."
            )
        else:
            upload_col1, upload_col2, upload_col3, upload_col4 = st.columns(
                [2, 2, 1, 1]
            )

            with upload_col1:
                upload_option = st.radio(
                    "Upload Scope",
                    ["All Locations", "Specific Location"],
                    key="upload_scope",
                )

            with upload_col2:
                if upload_option == "Specific Location":
                    location_strs = [f"{a} / {s}" for a, s in available_locations]
                    selected_loc = st.selectbox(
                        "Select Location", options=location_strs, key="upload_location"
                    )
                    upload_area, upload_site = selected_loc.split(" / ")
                else:
                    upload_area, upload_site = None, None

            with upload_col3:
                force_upload = st.checkbox("Force Re-upload", key="force_upload")

            with upload_col4:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("📤 Upload", type="primary"):
                    with st.spinner("Uploading content..."):
                        success, message, stats = (
                            st.session_state.upload_manager.upload_content(
                                area=upload_area, site=upload_site, force=force_upload
                            )
                        )

                        if success:
                            st.success(message)
                            if stats:
                                st.json(stats)
                            st.rerun()
                        else:
                            st.error(message)


if __name__ == "__main__":
    main()
