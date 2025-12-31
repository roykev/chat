#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Tourism Guide RAG Chat with Gemini

Usage:
    python gemini/main_interactive.py
    python gemini/main_interactive.py --area sharon --site bridge
    python gemini/main_interactive.py --model gemini-2.0-flash-exp
"""

import argparse
import sys
import os
import time
import locale

# Set UTF-8 encoding
if sys.stdout.encoding != 'UTF-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr.reconfigure(encoding='utf-8')

try:
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

import google.genai as genai
from google.genai import types

from config import GeminiConfig
from query_logger import QueryLogger
from store_registry import StoreRegistry
from store_manager import StoreManager


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
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        chunks.append(f"=== {file} ===\n{content}\n")
                        chunk_files.append(file)
                except Exception as e:
                    print(f"Warning: Could not read {file}: {e}")

    return "\n".join(chunks), chunk_files


def run_rag_chat(
    client: genai.Client,
    model_name: str,
    context: str,
    chunk_files: list[str],
    area: str,
    site: str,
    logger: QueryLogger,
    store_name: str,
    temperature: float = 0.7
):
    """
    Run interactive RAG chat session with manual context loading

    Args:
        client: Gemini API client
        model_name: Model to use
        context: RAG context from chunks
        chunk_files: List of chunk filenames used
        area: Tourism area
        site: Site name
        logger: Query logger instance
        store_name: File Search Store name (not used - for future compatibility)
        temperature: Generation temperature
    """
    print("\n" + "=" * 70)
    print("🗺️  Tourism Guide - Interactive Q&A")
    print("=" * 70)
    print(f"Area: {area} | Site: {site}")
    print(f"Model: {model_name}")
    print(f"Context: {len(context):,} characters from {len(chunk_files)} chunks")
    print("=" * 70)
    print("Type your questions below. Commands:")
    print("  - 'quit' or 'exit' to end session")
    print("  - 'stats' to show query statistics")
    print("=" * 70 + "\n")

    # System instruction with RAG context
    system_instruction = f"""You are a helpful tourism guide assistant for the {area} region,
specifically for the {site} area.

Use ONLY the following source material to answer questions. If the answer is not in the source material,
say so honestly. Always respond in the same language as the question.

SOURCE MATERIAL:
{context}

Answer questions based only on this source material."""

    while True:
        try:
            user_input = input("❓ שאלה / Question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\n👋 להתראות / Goodbye!")
                break

            if user_input.lower() == 'stats':
                stats = logger.get_stats()
                print("\n" + "=" * 70)
                print("📊 Query Statistics")
                print("=" * 70)
                print(f"Total queries: {stats['total_queries']}")
                print(f"Average response time: {stats['avg_response_time_seconds']}s")
                print(f"Areas: {', '.join(stats['areas'])}")
                print(f"Sites: {', '.join(stats['sites'])}")
                print("=" * 70 + "\n")
                continue

            # Query using manual context (File Search not supported in google-generativeai SDK)
            print("\n-> מחפש בתכנים / Searching context...")
            start_time = time.time()

            response = client.models.generate_content(
                model=model_name,
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature
                )
            )

            response_time = time.time() - start_time

            # Display response
            print("\n" + "-" * 70)
            print("🤖 תשובה / Answer:")
            print("-" * 70)
            print(response.text)
            print("-" * 70)
            print(f"⏱️  Response time: {response_time:.2f}s")
            print("-" * 70 + "\n")

            # Log the query
            logger.log_query(
                query=user_input,
                answer=response.text,
                model=model_name,
                context_chars=len(context),
                response_time_seconds=response_time,
                chunks_used=chunk_files
            )

        except KeyboardInterrupt:
            print("\n\n👋 להתראות / Goodbye!")
            break
        except EOFError:
            print("\n\n👋 להתראות / Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ שגיאה / Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive RAG chat for Tourism Guide"
    )

    parser.add_argument(
        '--area',
        help='Tourism area (default: auto-detect from registry)'
    )
    parser.add_argument(
        '--site',
        help='Site name (default: auto-detect from registry)'
    )
    parser.add_argument(
        '--model',
        help='Gemini model to use (default from config.yaml)'
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 70)
    print("📚 Tourism Guide - RAG Chat System")
    print("=" * 70)

    try:
        config = GeminiConfig.from_yaml()

        # Override with command-line arguments
        if args.model:
            config.model_name = args.model

    except FileNotFoundError as e:
        print(f"\n❌ Configuration error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    # Get store_id from registry
    area = args.area
    site = args.site
    store_id = None

    try:
        registry = StoreRegistry(config.registry_path)
        all_stores = registry.list_all()

        if not all_stores:
            print(f"\n❌ No stores found in registry.")
            print("Please run main_upload.py first to create and upload to a store.")
            sys.exit(1)


        # If area/site not specified, use first available
        if not area or not site:
            (area, site), store_id = list(all_stores.items())[0]
            print(f"\n-> Auto-detected: {area} / {site}")
            print(f"-> Store ID: {store_id}")
        else:
            # Look up specific area/site
            store_id = registry.get_store(area, site)
            if not store_id:
                print(f"\n❌ No store found for {area}/{site}")
                print(f"Available stores:")
                for (a, s), sid in all_stores.items():
                    print(f"  - {a}/{s}: {sid}")
                sys.exit(1)
            print(f"\n-> Using store: {area} / {site}")
            print(f"-> Store ID: {store_id}")

    except Exception as e:
        print(f"\n❌ Error loading registry: {e}")
        print("Please run main_upload.py first to create and upload to a store.")
        sys.exit(1)

    # Connect to Gemini
    print("\n-> Connecting to Gemini...")

    try:
        client = genai.Client(api_key=config.api_key)
        print("✓ Connected to Gemini API")
    except Exception as e:
        print(f"❌ Error connecting to Gemini: {e}")
        sys.exit(1)

    # Initialize store manager with store_id from registry
    print("\n-> Initializing File Search Store...")
    try:
        store_manager = StoreManager(
            client,
            f"{area}_{site}_Tourism_RAG",
            store_id=store_id
        )
        store_name = store_manager.store_name
        print(f"✓ Connected to store: {store_name}")
    except Exception as e:
        print(f"❌ Error connecting to store: {e}")
        sys.exit(1)

    # Load chunks for the area/site (used as fallback)
    chunks_dir = os.path.join(config.chunks_dir, area, site)
    print(f"\n-> Loading chunks from: {chunks_dir}")

    context, chunk_files = load_chunks(chunks_dir)

    if not context:
        print(f"\n❌ No chunks found for {area}/{site}")
        print(f"Please run main_upload.py first to create chunks.")
        sys.exit(1)

    print(f"✓ Loaded {len(chunk_files)} chunk files ({len(context):,} characters)")

    # Initialize logger
    log_path = os.path.join(os.path.dirname(config.registry_path), "query_log.jsonl")
    logger = QueryLogger(log_path, area=area, site=site)
    print(f"✓ Query logger initialized: {log_path}")

    # Start RAG chat session
    print(f"\n✓ Using model: {config.model_name} (temperature: {config.temperature})")
    print(f"✓ RAG mode: Manual context loading")
    print(f"✓ Store ID: {store_id} (for reference only)")

    run_rag_chat(
        client,
        config.model_name,
        context,
        chunk_files,
        area,
        site,
        logger,
        store_name,
        config.temperature
    )


if __name__ == "__main__":
    main()