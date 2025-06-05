#!/bin/bash
# Run all tests for the backend

# Set the conda environment name
CONDA_ENV="glacial-erratics"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get the backend directory path (parent of the tests directory)
BACKEND_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
echo -e "${YELLOW}Backend directory: ${BACKEND_DIR}${NC}"

echo -e "${YELLOW}Starting test suite for Glacial Erratics Backend...${NC}"

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
    echo -e "${YELLOW}Conda environment $CONDA_ENV is not activated.${NC}"
    echo -e "${YELLOW}Attempting to activate...${NC}"
    
    # Try to source conda.sh
    CONDA_SH_PATH="$(conda info --base)/etc/profile.d/conda.sh"
    if [[ -f "$CONDA_SH_PATH" ]]; then
        source "$CONDA_SH_PATH"
        conda activate $CONDA_ENV
    else
        echo -e "${RED}Failed to find conda.sh. Please activate conda environment manually:${NC}"
        echo -e "${YELLOW}conda activate $CONDA_ENV${NC}"
        exit 1
    fi
    
    # Check if activation was successful
    if [[ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]]; then
        echo -e "${RED}Failed to activate conda environment $CONDA_ENV.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Successfully activated conda environment $CONDA_ENV.${NC}"
fi

# Change to the backend directory to run the tests
cd "$BACKEND_DIR"

# Load environment variables for database connection
TEST_ENV_FILE="$BACKEND_DIR/tests/.env.test"
ENV_FILE="$BACKEND_DIR/.env"

if [[ -f "$TEST_ENV_FILE" ]]; then
    echo -e "${YELLOW}Loading environment variables from test .env file...${NC}"
    # Export variables from .env.test file (simple version without handling quoted values)
    export $(grep -v '^#' "$TEST_ENV_FILE" | xargs)
elif [[ -f "$ENV_FILE" ]]; then
    echo -e "${YELLOW}Loading environment variables from .env file...${NC}"
    # Export variables from .env file (simple version without handling quoted values)
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo -e "${RED}No .env or .env.test file found${NC}"
fi

echo -e "${YELLOW}Running JavaScript tests...${NC}"
npm test

JS_RESULT=$?

echo -e "${YELLOW}Running Python tests...${NC}"
if [[ -d "$BACKEND_DIR/src/scripts/python" ]]; then
    cd "$BACKEND_DIR/src/scripts/python"
    
    # Create a test environment variables file for Python tests
    echo "# Environment variables for Python tests" > .env.test
    echo "DB_HOST=$DB_HOST" >> .env.test
    echo "DB_PORT=$DB_PORT" >> .env.test
    echo "DB_NAME=$DB_NAME" >> .env.test
    echo "DB_USER=$DB_USER" >> .env.test
    echo "DB_PASSWORD=$DB_PASSWORD" >> .env.test
    
    python -m unittest discover -s tests
    PY_RESULT=$?
    
    # Remove the temporary file
    rm -f .env.test
else
    echo -e "${RED}Python scripts directory not found: $BACKEND_DIR/src/scripts/python${NC}"
    PY_RESULT=1
fi

# Return to the backend directory
cd "$BACKEND_DIR"

# Check the installation of Python packages
echo -e "${YELLOW}Checking Python package installations...${NC}"
pip list | grep -E "psycopg2|geopandas|numpy|pandas" || echo -e "${RED}Some required packages may be missing.${NC}"

echo -e "${YELLOW}Verifying database configuration...${NC}"
echo -e "${YELLOW}DB_HOST: $DB_HOST, DB_USER: $DB_USER, DB_NAME: $DB_NAME${NC}"

# Check database connectivity
if [[ -z "$DB_PASSWORD" || -z "$DB_HOST" || -z "$DB_USER" || -z "$DB_NAME" ]]; then
    echo -e "${RED}Database environment variables not properly set${NC}"
    DB_RESULT=1
else
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT count(*) FROM \"Erratics\";" >/dev/null 2>&1
    DB_RESULT=$?
fi

# Print test results
echo -e "\n${YELLOW}Test Results:${NC}"
if [ $JS_RESULT -eq 0 ]; then
    echo -e "JavaScript tests: ${GREEN}PASSED${NC}"
else
    echo -e "JavaScript tests: ${RED}FAILED${NC}"
fi

if [ $PY_RESULT -eq 0 ]; then
    echo -e "Python tests: ${GREEN}PASSED${NC}"
else
    echo -e "Python tests: ${RED}FAILED${NC}"
fi

if [ $DB_RESULT -eq 0 ]; then
    echo -e "Database connectivity: ${GREEN}OK${NC}"
else
    echo -e "Database connectivity: ${RED}FAILED${NC}"
    echo -e "${YELLOW}Please check your database configuration in .env file.${NC}"
fi

# Final result
if [ $JS_RESULT -eq 0 ] && [ $PY_RESULT -eq 0 ] && [ $DB_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed successfully!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Please check the output above.${NC}"
    exit 1
fi 