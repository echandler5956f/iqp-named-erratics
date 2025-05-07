#!/usr/bin/env node

/**
 * Script to install Python dependencies required for spatial analysis
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

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

// Path to requirements.txt
const pythonDir = path.join(__dirname, 'python');
const requirementsPath = path.join(pythonDir, 'requirements.txt');

console.log(`${colors.blue}Checking for Python dependencies...${colors.reset}`);

// Check if requirements.txt exists
if (!fs.existsSync(requirementsPath)) {
  console.error(`${colors.red}Error: requirements.txt not found at ${requirementsPath}${colors.reset}`);
  process.exit(1);
}

try {
  // Check if we're in a conda environment
  const condaEnvName = process.env.CONDA_DEFAULT_ENV;
  let useCondaPip = !!condaEnvName;
  
  // Check if conda is available
  try {
    const condaInfo = execSync('conda info', { encoding: 'utf8' });
    if (!condaEnvName) {
      console.log(`${colors.yellow}Warning: Not in an active conda environment. Using system pip.${colors.reset}`);
    } else {
      console.log(`${colors.green}Using conda environment: ${condaEnvName}${colors.reset}`);
    }
  } catch (error) {
    useCondaPip = false;
    console.log(`${colors.yellow}Conda not found. Using system pip.${colors.reset}`);
  }
  
  // Install dependencies
  const pipCmd = useCondaPip ? 'conda run -n ' + condaEnvName + ' pip' : 'pip';
  
  console.log(`${colors.blue}Installing Python dependencies using ${pipCmd}...${colors.reset}`);
  execSync(`${pipCmd} install -r "${requirementsPath}"`, { 
    stdio: 'inherit',
    encoding: 'utf8'
  });
  
  // Install spacy model
  console.log(`${colors.blue}Installing spaCy model...${colors.reset}`);
  execSync(`${pipCmd} install -U spacy==3.7.2`, { stdio: 'inherit' });
  
  // The correct way to run spacy download command (no 'pip run')
  if (useCondaPip) {
    execSync(`conda run -n ${condaEnvName} python -m spacy download en_core_web_md`, { stdio: 'inherit' });
  } else {
    execSync(`python -m spacy download en_core_web_md`, { stdio: 'inherit' });
  }
  
  console.log(`${colors.green}Python dependencies installed successfully!${colors.reset}`);
} catch (error) {
  console.error(`${colors.red}Error installing Python dependencies: ${error.message}${colors.reset}`);
  process.exit(1);
}

// Check if we need to create any directories for data storage
const dataDir = path.join(pythonDir, 'data');
if (!fs.existsSync(dataDir)) {
  console.log(`${colors.blue}Creating data directory at ${dataDir}${colors.reset}`);
  fs.mkdirSync(dataDir, { recursive: true });
}

console.log(`${colors.green}Setup complete! Python environment is ready for spatial analysis.${colors.reset}`); 