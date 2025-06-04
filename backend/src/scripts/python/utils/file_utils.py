# backend/src/scripts/python/utils/file_utils.py
import json
import os
import logging

logger = logging.getLogger(__name__)

def json_to_file(data: dict, filepath: str) -> bool:
    """Saves dictionary data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved JSON data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to file {filepath}: {e}", exc_info=True)
        return False 