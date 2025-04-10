#!/usr/bin/env python3
"""
Test script for unit tests.
This script simply echoes its arguments as JSON.
"""

import sys
import json

if __name__ == "__main__":
    # Get the arguments
    args = sys.argv[1:]
    
    # Return as JSON
    result = {"result": "success", "args": args}
    print(json.dumps(result)) 