#!/bin/bash

# Start Python environment
echo "Activating Python environment..."
conda activate geo-py310

# Print environment info
echo "Python version: $(python --version)"
echo "Node.js version: $(node --version)"
echo "NPM version: $(npm --version)"
echo ""
echo "Development environment ready!"
echo "Run 'npm run start:frontend' for frontend"
echo "Run 'npm run start:backend' for backend"
