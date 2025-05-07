#!/usr/bin/env node

/**
 * Script to install pgvector from source
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const dotenv = require('dotenv');

// Load environment variables
const envPath = path.join(__dirname, '../../.env');
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
}

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

console.log(`${colors.blue}Installing pgvector from source...${colors.reset}`);

try {
  // Detect PostgreSQL version
  let pgVersion = '14'; // Default version
  
  try {
    // Try to run psql and get version
    const psqlVersionOutput = execSync('psql --version', { encoding: 'utf8' });
    // Extract version using regex
    const versionMatch = psqlVersionOutput.match(/PostgreSQL\s+(\d+)/i);
    
    if (versionMatch && versionMatch[1]) {
      pgVersion = versionMatch[1].trim();
      console.log(`${colors.green}Detected PostgreSQL version: ${pgVersion}${colors.reset}`);
    } else {
      console.log(`${colors.yellow}Could not parse PostgreSQL version, using default: ${pgVersion}${colors.reset}`);
    }
  } catch (err) {
    console.log(`${colors.yellow}Could not detect PostgreSQL version: ${err.message}${colors.reset}`);
    console.log(`${colors.yellow}Using default version: ${pgVersion}${colors.reset}`);
  }
  
  // Install required build dependencies
  console.log(`${colors.yellow}Installing build dependencies...${colors.reset}`);
  execSync('sudo apt-get update', { stdio: 'inherit' });
  execSync(`sudo apt-get install -y git make gcc postgresql-server-dev-${pgVersion}`, { stdio: 'inherit' });
  
  // Create a temporary directory for the build
  const tempDir = path.join(os.tmpdir(), 'pgvector-build-' + Date.now());
  console.log(`${colors.yellow}Creating temp directory: ${tempDir}${colors.reset}`);
  
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  
  process.chdir(tempDir);
  
  // Clone pgvector repository
  console.log(`${colors.yellow}Cloning pgvector repository...${colors.reset}`);
  execSync('git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git .', { stdio: 'inherit' });
  
  // Build and install
  console.log(`${colors.yellow}Building pgvector...${colors.reset}`);
  execSync('make', { stdio: 'inherit' });
  
  console.log(`${colors.yellow}Installing pgvector...${colors.reset}`);
  execSync('sudo make install', { stdio: 'inherit' });
  
  console.log(`${colors.green}pgvector has been successfully built and installed from source!${colors.reset}`);
  
  // Clean up
  console.log(`${colors.yellow}Cleaning up temporary files...${colors.reset}`);
  process.chdir(os.homedir());
  execSync(`rm -rf ${tempDir}`, { stdio: 'inherit' });
  
  // Get database name from environment variables
  const dbName = process.env.DB_NAME || 'glacial_erratics';
  const dbUser = process.env.DB_USER || 'quant';
  
  console.log(`${colors.green}Using database connection info:${colors.reset}`);
  console.log(`  ${colors.yellow}Database: ${dbName}${colors.reset}`);
  console.log(`  ${colors.yellow}User: ${dbUser}${colors.reset}`);
  
  // Check if database exists, create if it doesn't
  try {
    console.log(`${colors.yellow}Checking if database ${dbName} exists...${colors.reset}`);
    execSync(`psql -lqt | cut -d \\| -f 1 | grep -qw ${dbName}`);
    console.log(`${colors.green}Database ${dbName} exists!${colors.reset}`);
  } catch (error) {
    console.log(`${colors.yellow}Database ${dbName} does not exist. Creating it...${colors.reset}`);
    try {
      execSync(`createdb ${dbName}`, { stdio: 'inherit' });
      console.log(`${colors.green}Database ${dbName} created successfully!${colors.reset}`);
    } catch (createError) {
      console.error(`${colors.red}Failed to create database: ${createError.message}${colors.reset}`);
      console.log(`${colors.yellow}Please create the database manually:${colors.reset}`);
      console.log(`  ${colors.cyan}createdb ${dbName}${colors.reset}`);
    }
  }
  
  // Provide instructions for next steps
  console.log(`\n${colors.blue}==== Next Steps ====${colors.reset}`);
  console.log(`${colors.green}1. Make sure the database exists:${colors.reset}`);
  console.log(`   createdb ${dbName}`);
  console.log(`${colors.green}2. Enable pgvector in your database:${colors.reset}`);
  console.log(`   psql -d ${dbName} -c "CREATE EXTENSION vector;"`);
  console.log(`${colors.green}3. Run database migrations:${colors.reset}`);
  console.log(`   npm run db:migrate`);
  
} catch (error) {
  console.error(`${colors.red}Error: ${error.message}${colors.reset}`);
  process.exit(1);
} 