"""
File chunking functionality for creating text segments
Supports various file formats via file_parser
"""

import os
import hashlib
import unicodedata
from typing import List
from file_parser import parse_file


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to use only ASCII characters

    Args:
        filename: Original filename (may contain non-ASCII characters)

    Returns:
        ASCII-safe filename
    """
    # Try to transliterate Unicode characters to ASCII
    try:
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', filename)
        # Encode to ASCII, ignoring characters that can't be converted
        ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')

        # If we lost too many characters, use a hash instead
        if len(ascii_name) < len(filename) * 0.3:  # Lost more than 70%
            # Create a hash of the original filename
            file_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
            return f"file_{file_hash}"

        # Remove any remaining problematic characters
        safe_name = ''.join(c if c.isalnum() or c in '_- ' else '_' for c in ascii_name)
        # Clean up multiple underscores and spaces
        safe_name = ' '.join(safe_name.split())
        safe_name = '_'.join(safe_name.split('_'))

        return safe_name if safe_name else f"file_{hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]}"

    except Exception:
        # Fallback to hash if anything goes wrong
        file_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
        return f"file_{file_hash}"


def chunk_text_file(
    file_path: str,
    file_id: str,
    chunk_size: int = 1000,
    output_dir: str = "chunks"
) -> List[str]:
    """
    Parse and split a file into chunks, save as separate text files

    Supports: .txt, .md, .docx, .pdf

    Args:
        file_path: Path to the input file
        file_id: Identifier for the file (used in chunk filenames)
        chunk_size: Number of characters per chunk
        output_dir: Directory to save chunk files

    Returns:
        List of file paths for created chunks
    """
    # Sanitize file_id to ensure ASCII-safe filenames
    safe_file_id = sanitize_filename(file_id)

    # Parse the file to extract text
    try:
        content = parse_file(file_path)
    except ValueError as e:
        print(f"   Warning: {e}")
        return []

    if not content.strip():
        print(f"   Warning: No text content extracted from {file_path}")
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split into chunks
    chunks = []
    current_pos = 0
    chunk_num = 0

    while current_pos < len(content):
        # Get chunk of text
        chunk_text = content[current_pos:current_pos + chunk_size]

        # Try to break at sentence/paragraph boundary if possible
        if current_pos + chunk_size < len(content):
            # Look for paragraph break
            last_newline = chunk_text.rfind('\n\n')
            if last_newline > chunk_size * 0.5:  # At least 50% through chunk
                chunk_text = chunk_text[:last_newline]
            else:
                # Look for sentence break
                last_period = max(
                    chunk_text.rfind('. '),
                    chunk_text.rfind('.\n'),
                    chunk_text.rfind('! '),
                    chunk_text.rfind('? ')
                )
                if last_period > chunk_size * 0.5:
                    chunk_text = chunk_text[:last_period + 1]

        # Save chunk to file
        chunk_num += 1
        chunk_filename = f"{safe_file_id}_chunk_{chunk_num:03d}.txt"
        chunk_filepath = os.path.join(output_dir, chunk_filename)

        with open(chunk_filepath, 'w', encoding='utf-8') as f:
            f.write(f"--- {file_id}: Chunk {chunk_num} ---\n")
            f.write(f"Source: {os.path.basename(file_path)}\n\n")
            f.write(chunk_text)

        chunks.append(chunk_filepath)

        # Move position forward by actual chunk size
        actual_chunk_size = len(chunk_text)
        current_pos += actual_chunk_size

        # Skip whitespace at the beginning of next chunk
        while current_pos < len(content) and content[current_pos].isspace():
            current_pos += 1

    return chunks


def chunk_text_smart(
    text: str,
    file_id: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    output_dir: str = "chunks"
) -> List[str]:
    """
    Split text into overlapping chunks with smart boundary detection

    Args:
        text: Text content to chunk
        file_id: Identifier for the content
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        output_dir: Directory to save chunk files

    Returns:
        List of file paths for created chunks
    """
    # Sanitize file_id to ensure ASCII-safe filenames
    safe_file_id = sanitize_filename(file_id)

    os.makedirs(output_dir, exist_ok=True)

    chunks = []
    current_pos = 0
    chunk_num = 0

    while current_pos < len(text):
        # Calculate end position
        end_pos = min(current_pos + chunk_size, len(text))

        # Extract chunk
        chunk_text = text[current_pos:end_pos]

        # Find good break point (if not at end)
        if end_pos < len(text):
            # Try to break at paragraph
            last_para = chunk_text.rfind('\n\n')
            if last_para > chunk_size * 0.6:
                chunk_text = chunk_text[:last_para]
                end_pos = current_pos + last_para
            else:
                # Try to break at sentence
                for delimiter in ['. ', '.\n', '! ', '? ']:
                    last_sent = chunk_text.rfind(delimiter)
                    if last_sent > chunk_size * 0.6:
                        chunk_text = chunk_text[:last_sent + len(delimiter)]
                        end_pos = current_pos + last_sent + len(delimiter)
                        break

        # Save chunk
        chunk_num += 1
        chunk_filename = f"{safe_file_id}_chunk_{chunk_num:03d}.txt"
        chunk_filepath = os.path.join(output_dir, chunk_filename)

        with open(chunk_filepath, 'w', encoding='utf-8') as f:
            f.write(f"--- {file_id}: Chunk {chunk_num} ---\n\n")
            f.write(chunk_text.strip())

        chunks.append(chunk_filepath)

        # Move to next chunk with overlap
        current_pos = end_pos - overlap
        if current_pos >= len(text):
            break

    return chunks