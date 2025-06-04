#!/usr/bin/env node

const path = require('path');
const fs = require('fs');
const dotenv = require('dotenv');

// Load environment variables from .env file in the project root
// __dirname for this script is backend/src/scripts/
const envPath = path.resolve(__dirname, '../../../.env'); 
console.log(`[importData.js] Attempting to load environment variables from: ${envPath}`);
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

// Now we can import modules that depend on environment variables
const { importErraticsFromCSV } = require('../utils/importData');
const { initializeDatabase } = require('../utils/dbInit');

const DEFAULT_DATA_PATH = path.resolve(__dirname, '../data/erratics.csv');

async function main() {
  try {
    // Parse command line arguments
    const args = process.argv.slice(2);
    const dataPath = args.length > 0 ? args[0] : DEFAULT_DATA_PATH;
    
    // Check if file exists
    if (!fs.existsSync(dataPath)) {
      console.error(`Error: File not found at path: ${dataPath}`);
      console.log('Usage: node importData.js [path/to/csv]');
      process.exit(1);
    }
    
    console.log(`Initializing database...`);
    const dbInitialized = await initializeDatabase();
    if (!dbInitialized) {
      console.error('Failed to initialize database.');
      process.exit(1);
    }
    
    console.log(`Importing data from: ${dataPath}`);
    const count = await importErraticsFromCSV(dataPath);
    
    if (count > 0) {
      console.log(`Successfully imported ${count} erratic records`);
    } else {
      console.warn('No data was imported. Check your CSV file format.');
    }
    
    process.exit(0);
  } catch (error) {
    console.error('Error importing data:', error);
    process.exit(1);
  }
}

main(); 