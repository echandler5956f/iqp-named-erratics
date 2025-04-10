const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs');

// Set NODE_ENV to test
process.env.NODE_ENV = 'test';

// Try to load environment variables from .env.test first, then fall back to .env
const testEnvPath = path.resolve(__dirname, '.env.test');
const regularEnvPath = path.resolve(__dirname, '../.env');

if (fs.existsSync(testEnvPath)) {
  console.log(`Loading test environment from: ${testEnvPath}`);
  dotenv.config({ path: testEnvPath });
} else if (fs.existsSync(regularEnvPath)) {
  console.log(`Loading environment from: ${regularEnvPath}`);
  dotenv.config({ path: regularEnvPath });
} else {
  console.warn('No .env or .env.test file found. Using existing environment variables.');
}

// Log the database connection details
console.log(`Database connection: ${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_NAME}`); 