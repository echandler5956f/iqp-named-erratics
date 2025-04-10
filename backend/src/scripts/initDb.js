#!/usr/bin/env node

const path = require('path');
const dotenv = require('dotenv');

// Load environment variables from .env file
const envPath = path.resolve(__dirname, '../../.env');
const result = dotenv.config({ path: envPath });
if (result.error) {
  console.error('Error loading .env file:', result.error);
  process.exit(1);
}

// Validate required environment variables
const requiredEnvVars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'];
const missingEnvVars = requiredEnvVars.filter(varName => !process.env[varName]);
if (missingEnvVars.length > 0) {
  console.error('Missing required environment variables:', missingEnvVars.join(', '));
  console.error('Please check your .env file at:', envPath);
  process.exit(1);
}

// Now we can import other modules that depend on environment variables
const { initializeDatabase } = require('../utils/dbInit');

async function main() {
  try {
    console.log('Initializing database...');
    const result = await initializeDatabase();
    
    if (result) {
      console.log('Database initialization successful!');
      console.log('You can now start the application or import data.');
    } else {
      console.error('Database initialization failed.');
      process.exit(1);
    }
  } catch (error) {
    console.error('Error initializing database:', error);
    process.exit(1);
  }
}

main(); 