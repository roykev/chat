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
import locale
import os
import sys
import time

# Add parent directory to path if running as script
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    # Change to parent directory so relative paths in config work
    os.chdir(parent_dir)

# Set UTF-8 encoding
if sys.stdout.encoding != "UTF-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "UTF-8":
    sys.stderr.reconfigure(encoding="utf-8")

try:
    locale.setlocale(locale.LC_ALL, "")
except:
    pass

import google.genai as genai
from google.genai import types

from gemini.config import GeminiConfig
from gemini.query_logger import QueryLogger
from gemini.store_registry import StoreRegistry


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
    store_id: str,
    temperature: float = 0.7,
):
    """
    Run interactive RAG chat session with logging

    Args:
        client: Gemini API client
        model_name: Model to use
        context: RAG context from chunks
        chunk_files: List of chunk filenames used
        area: Tourism area
        site: Site name
        logger: Query logger instance
        store_id: Gemini store ID
        temperature: Generation temperature
    """
    # Get document count from uploaded files
    doc_count = 0
    try:
        # Count files uploaded to Files API
        files_list = list(client.files.list())
        doc_count = len(files_list)
    except Exception as e:
        print(f"Warning: Could not get file count: {e}")

    print("\n" + "=" * 70)
    print("🗺️  Tourism Guide - Interactive Q&A")
    print("=" * 70)
    print(f"Area: {area} | Site: {site}")
    print(f"Model: {model_name}")
    print(f"Store: {store_id}")
    print(f"Documents in storage: {doc_count}")
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

            if user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 להתראות / Goodbye!")
                break

            if user_input.lower() == "stats":
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

            # Query with RAG context and timing
            print("\n-> מחפש בתכנים / Searching content...")
            start_time = time.time()

            response = client.models.generate_content(
                model=model_name,
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction, temperature=temperature
                ),
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
                chunks_used=chunk_files,
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
        "--area", help="Tourism area (default: auto-detect from registry)"
    )
    parser.add_argument("--site", help="Site name (default: auto-detect from registry)")
    parser.add_argument(
        "--chunks-dir", help="Directory containing chunk files (default: from config)"
    )
    parser.add_argument(
        "--model", help="Gemini model to use (default from config.yaml)"
    )
    parser.add_argument(
        "--test",
        help="Non-interactive test mode: provide a single question to test",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show diagnostics: storage info, file counts, configuration",
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 70)
    print("📚 Tourism Guide - RAG System")
    print("=" * 70)

    try:
        config = GeminiConfig.from_yaml()

        # Assertion 1: Verify API key is not None
        assert (
            config.api_key is not None
        ), "API key is None - check GOOGLE_API_KEY in .env file"
        assert (
            len(config.api_key.strip()) > 0
        ), "API key is empty - check GOOGLE_API_KEY in .env file"
        print(f"\n✓ API key loaded (length: {len(config.api_key)})")

        # Override with command-line arguments
        if args.model:
            config.model_name = args.model

        chunks_dir = args.chunks_dir if args.chunks_dir else config.chunks_dir

    except FileNotFoundError as e:
        print(f"\n❌ Configuration error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        sys.exit(1)

    # Determine area/site from registry if not specified
    area = args.area
    site = args.site

    if not area or not site:
        try:
            registry = StoreRegistry(config.registry_path)
            all_stores = registry.list_all()

            if all_stores:
                # Use first available area/site
                (area, site), store_id = list(all_stores.items())[0]
                print(f"\n-> Auto-detected: {area} / {site}")
            else:
                area = area or "unknown"
                site = site or "unknown"
        except Exception as e:
            print(f"Warning: Could not load registry: {e}")
            area = area or "unknown"
            site = site or "unknown"

    # Assertion 2: Verify store exists in registry
    try:
        registry = StoreRegistry(config.registry_path)
        store_id = registry.get_store(area, site)
        assert (
            store_id is not None
        ), f"Store not found in registry for area='{area}', site='{site}'. Run main_upload.py first."
        print(f"✓ Store verified in registry: {store_id}")
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error verifying store: {e}")
        sys.exit(1)

    # Connect to Gemini
    print("\n-> Connecting to Gemini...")

    try:
        client = genai.Client(api_key=config.api_key)
        print("✓ Connected to Gemini API")
    except Exception as e:
        print(f"❌ Error connecting to Gemini: {e}")
        sys.exit(1)

    # Diagnostics mode
    if args.diagnostics:
        print("\n" + "=" * 70)
        print("🔍 DIAGNOSTICS")
        print("=" * 70)

        # Configuration
        print("\n📋 Configuration:")
        print(f"  Config file: config.yaml")
        print(f"  API key: {config.api_key[:20]}... (length: {len(config.api_key)})")
        print(f"  Model: {config.model_name}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Content root: {config.content_root}")
        print(f"  Chunks directory: {config.chunks_dir}")
        print(f"  Registry path: {config.registry_path}")

        # Registry info
        print("\n📂 Registry Info:")
        registry = StoreRegistry(config.registry_path)
        all_stores = registry.list_all()
        print(f"  Total registered stores: {len(all_stores)}")
        for (reg_area, reg_site), reg_store_id in all_stores.items():
            metadata = registry.registry.get(f"{reg_area}:{reg_site}", {}).get(
                "metadata", {}
            )
            print(f"\n  [{reg_area} / {reg_site}]")
            print(f"    Store ID: {reg_store_id}")
            print(f"    Files: {metadata.get('file_count', 'N/A')}")
            print(f"    Chunks: {metadata.get('chunk_count', 'N/A')}")
            print(f"    Created: {metadata.get('created_at', 'N/A')}")
            print(f"    Updated: {metadata.get('last_updated', 'N/A')}")

        # Current area/site
        print(f"\n🎯 Current Selection:")
        print(f"  Area: {area}")
        print(f"  Site: {site}")
        print(f"  Store ID: {store_id}")

        # Check chunks directory
        area_site_chunks_dir = os.path.join(config.chunks_dir, area, site)
        print(f"\n📁 Chunks Directory:")
        print(f"  Path: {area_site_chunks_dir}")
        if os.path.exists(area_site_chunks_dir):
            chunk_files = [
                f for f in os.listdir(area_site_chunks_dir) if f.endswith(".txt")
            ]
            print(f"  Exists: Yes")
            print(f"  Chunk files: {len(chunk_files)}")
            total_size = sum(
                os.path.getsize(os.path.join(area_site_chunks_dir, f))
                for f in chunk_files
            )
            print(f"  Total size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
            if chunk_files:
                print(f"  Files:")
                for cf in sorted(chunk_files)[:5]:
                    size = os.path.getsize(os.path.join(area_site_chunks_dir, cf))
                    print(f"    - {cf} ({size} bytes)")
                if len(chunk_files) > 5:
                    print(f"    ... and {len(chunk_files) - 5} more")
        else:
            print(f"  Exists: No")

        # Files API info
        print(f"\n☁️  Gemini Files API:")
        try:
            uploaded_files = list(client.files.list())
            print(f"  Total uploaded files: {len(uploaded_files)}")
            if uploaded_files:
                print(f"  Files:")
                for uf in uploaded_files[:10]:
                    print(f"    - {uf.name}")
                    print(f"      Display name: {uf.display_name}")
                    print(f"      State: {uf.state}")
                    print(f"      Created: {uf.create_time}")
                    print(f"      Expires: {uf.expiration_time}")
                if len(uploaded_files) > 10:
                    print(f"    ... and {len(uploaded_files) - 10} more")
        except Exception as e:
            print(f"  Error listing files: {e}")

        print("\n" + "=" * 70)
        sys.exit(0)

    # Initialize logger
    log_path = os.path.join(os.path.dirname(config.registry_path), "query_log.jsonl")
    logger = QueryLogger(log_path, area=area, site=site)
    print(f"✓ Query logger initialized: {log_path}")

    # Ensure model name has models/ prefix
    model_name = config.model_name
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    print(f"✓ Using model: {model_name} (temperature: {config.temperature})")

    # Load chunks from disk for context
    area_site_chunks_dir = os.path.join(chunks_dir, area, site)
    context, chunk_files = load_chunks(area_site_chunks_dir)

    if not context:
        print(f"\n⚠️  Warning: No chunks found in {area_site_chunks_dir}")
        print("    The assistant will have no context to answer questions.")
    else:
        print(f"✓ Loaded {len(chunk_files)} chunks ({len(context)} characters)")

    # Test mode - single question
    if args.test:
        print("\n" + "=" * 70)
        print("🧪 TEST MODE - Single Question")
        print("=" * 70)
        print(f"Question: {args.test}\n")

        system_instruction = f"""You are a helpful tourism guide assistant for the {area} region,
specifically for the {site} area.

Use ONLY the following source material to answer questions. If the answer is not in the source material,
say so honestly. Always respond in the same language as the question.

SOURCE MATERIAL:
{context}

Answer questions based only on this source material."""

        try:
            import time

            start_time = time.time()

            response = client.models.generate_content(
                model=model_name,
                contents=args.test,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=config.temperature,
                ),
            )

            response_time = time.time() - start_time

            print("Answer:")
            print("-" * 70)
            print(response.text)
            print("-" * 70)
            print(f"Response time: {response_time:.2f}s")

        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

        sys.exit(0)

    # Interactive mode
    run_rag_chat(
        client,
        model_name,
        context,
        chunk_files,
        area,
        site,
        logger,
        store_id,
        config.temperature,
    )


if __name__ == "__main__":
    main()
