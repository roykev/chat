"""
Gemini File Search Store management (create, upload, list)
"""

import os
import sys
import time
from typing import List

import google.genai as genai
from google.genai import types


class StoreManager:
    """Manages Gemini File Search Store operations"""

    def __init__(
        self, client: genai.Client, store_display_name: str, store_id: str = None
    ):
        """
        Initialize store manager

        Args:
            client: Gemini API client
            store_display_name: Display name for the store
            store_id: Specific store ID to use (optional)
        """
        self.client = client
        self.store_display_name = store_display_name
        self.store_id = store_id
        self._store = None

    def get_or_create_store(self) -> types.FileSearchStore:
        """
        Get existing store or create new one

        Returns:
            FileSearchStore instance
        """
        if self._store:
            return self._store

        # If specific store ID is provided, use that directly
        if self.store_id:
            print(f"\n-> Using specified store ID: {self.store_id}")
            try:
                store = self.client.file_search_stores.get(name=self.store_id)
                print(f"-> Successfully connected to store: {store.name}")
                self._store = store
                return store
            except Exception as e:
                print(f"-> Error connecting to store {self.store_id}: {e}")
                print(f"-> Falling back to search by display name...")

        print(
            f"\n-> Checking for existing File Search Store: '{self.store_display_name}'..."
        )

        # List stores and check for display name match
        for store in self.client.file_search_stores.list():
            if store.display_name == self.store_display_name:
                print(f"-> Found existing store: {store.name}")
                self._store = store
                return store

        # Create new store if not found
        print(f"-> Store not found. Creating new store...")

        try:
            store = self.client.file_search_stores.create(
                display_name=self.store_display_name
            )
            print(f"-> Successfully created new store: {store.name}")
            print(f"   Display name: {self.store_display_name}")
        except (TypeError, Exception):
            print(f"-> Note: display_name not supported by API, creating without it")
            store = self.client.file_search_stores.create()
            print(f"-> Successfully created new store: {store.name}")

        self._store = store
        return store

    def upload_files(self, file_paths: List[str], max_wait_seconds: int = 300) -> List:
        """
        Upload multiple files to the store

        Args:
            file_paths: List of file paths to upload
            max_wait_seconds: Maximum time to wait for uploads to complete

        Returns:
            List of upload operations
        """
        store = self.get_or_create_store()

        print(f"\n-> Uploading {len(file_paths)} files to store '{store.name}'...")

        operations = []
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                # Handle Unicode filenames safely
                safe_filename = filename.encode("utf-8", errors="replace").decode(
                    "utf-8"
                )
                print(f"   Uploading: {safe_filename}")
            except Exception:
                print(f"   Uploading: {file_path}")

            try:
                # Ensure file path is properly encoded
                if isinstance(file_path, str):
                    file_path_encoded = file_path.encode(
                        "utf-8", errors="replace"
                    ).decode("utf-8")
                else:
                    file_path_encoded = file_path

                op = self.client.file_search_stores.upload_to_file_search_store(
                    file_search_store_name=store.name, file=file_path
                )
                operations.append(op)
            except Exception as e:
                print(f"   ❌ Error uploading file: {e}")
                # Re-raise to be caught by the outer handler
                raise

        print(f"-> Successfully submitted {len(operations)} files for upload.")

        # Wait for operations to complete
        print("-> Waiting for uploads to complete...")
        start_time = time.time()
        last_status_time = start_time

        while time.time() - start_time < max_wait_seconds:
            # Refresh operation status from server
            for i in range(len(operations)):
                operations[i] = self.client.operations.get(operations[i])

            all_done = all(op.done for op in operations)
            if all_done:
                break

            # Print progress every 30 seconds
            current_time = time.time()
            if current_time - last_status_time >= 30:
                elapsed = int(current_time - start_time)
                done_count = sum(1 for op in operations if op.done)
                print(
                    f"   [{elapsed}s] {done_count}/{len(operations)} operations completed..."
                )
                last_status_time = current_time

            time.sleep(2)

        # Check results
        succeeded = 0
        failed = 0
        incomplete = 0

        for op in operations:
            if op.done:
                if hasattr(op, "error") and op.error:
                    print(f"   ✗ Upload failed: {op.error}")
                    failed += 1
                else:
                    succeeded += 1
            else:
                incomplete += 1

        print(f"\n-> Upload results:")
        print(f"   ✓ Succeeded: {succeeded}")
        if failed > 0:
            print(f"   ✗ Failed: {failed}")
        if incomplete > 0:
            print(f"   ⚠ Incomplete: {incomplete}")

        if failed > 0 or incomplete > 0:
            raise Exception(
                f"Upload failed: {succeeded} succeeded, {failed} failed, {incomplete} incomplete"
            )

        print("-> All files successfully uploaded to store.")
        return operations

    def list_files(self) -> int:
        """
        Get the count of active documents in the store

        Returns:
            Number of active documents
        """
        store = self.get_or_create_store()

        # Refresh store info to get latest document counts
        store = self.client.file_search_stores.get(name=store.name)

        active_count = store.active_documents_count or 0
        print(f"-> Store has {active_count} active documents")

        return active_count

    @property
    def store_name(self) -> str:
        """Get the store name (creates store if needed)"""
        store = self.get_or_create_store()
        return store.name

    def list_all_stores(self) -> list:
        """
        List all available stores

        Returns:
            List of store objects
        """
        try:
            stores = list(self.client.file_search_stores.list())
            print(f"\n-> Found {len(stores)} total stores:")
            for i, store in enumerate(stores, 1):
                display_name = getattr(store, "display_name", "N/A")
                print(f"   [{i}] {store.name}")
                print(f"       Display Name: {display_name}")
            return stores
        except Exception as e:
            print(f"-> Error listing stores: {e}")
            return []
