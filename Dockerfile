# Use a base image with conda pre-installed
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies (including GDAL for ogr2ogr)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gdal-bin \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Copy the entire project
COPY . .

# Install Node.js dependencies
RUN npm run install:all

# Build the frontend
RUN npm run build

# Copy and run the conda environment setup script
RUN chmod +x backend/src/scripts/python/create_conda_env_strict.sh
RUN cd backend/src/scripts/python && ./create_conda_env_strict.sh

# Create directory for Python data (will be mounted or populated)
RUN mkdir -p backend/src/scripts/python/data/gis

# Run database migrations (will be overridden by env vars in production)
# This is just to ensure the migration files are valid
RUN cd backend && npm run db:migrate || echo "Database migration failed (expected in build phase)"

# Create a startup script that activates conda and starts the server
RUN echo '#!/bin/bash\n\
eval "$(conda shell.bash hook)"\n\
conda activate glacial-erratics\n\
cd /app\n\
npm run db:migrate\n\
cd backend && npm start' > /app/start.sh \
    && chmod +x /app/start.sh

# Expose the port
EXPOSE 3001

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/api/health || exit 1

# Start the application
CMD ["/app/start.sh"] 