"""
Utility functions for the Gemini Tourism RAG system
"""

import os


def source_key(param="OPENAI_API_KEY"):
    """
    Load API key from ~/.bashrc environment variables

    Args:
        param: Name of the environment variable to retrieve (default: "OPENAI_API_KEY")

    Returns:
        The value of the requested environment variable

    Raises:
        KeyError: If the environment variable is not found in ~/.bashrc
    """
    # Load the contents of ~/.bashrc into environment variables
    bashrc_path = os.path.expanduser("~/.bashrc")
    with open(bashrc_path, "r") as f:
        bashrc_contents = f.read()

    # Split the contents into lines and process each line
    for line in bashrc_contents.split("\n"):
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Parse lines in the format: export VARIABLE=value
        if line.startswith("export "):
            parts = line.split(" ", 1)[1].split("=", 1)
            if len(parts) == 2:
                variable, value = parts
                os.environ[variable] = value.strip('"')

    # Now you can access the environment variables as if they were set in the shell
    return os.environ[param]