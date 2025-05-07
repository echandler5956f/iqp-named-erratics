#!/usr/bin/env node

/**
 * Script to run spatial analysis on all erratics
 * This runs both proximity analysis and classification on all erratics
 */

const path = require('path');
const db = require('../models');
const pythonService = require('../services/pythonService');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  bright: '\x1b[1m'
};

// Feature layers to analyze
const FEATURE_LAYERS = ['water_bodies', 'settlements', 'trails', 'roads'];

// Process a batch of erratics
async function processBatch(erratics, updateDb = true) {
  const results = {
    total: erratics.length,
    proximity: { successful: 0, failed: 0 },
    classification: { successful: 0, failed: 0 }
  };
  
  for (let i = 0; i < erratics.length; i++) {
    const erratic = erratics[i];
    const id = erratic.id;
    const name = erratic.name || `ID: ${id}`;
    
    // Log progress
    process.stdout.write(`\r${colors.bright}Processing ${i+1}/${erratics.length}: ${colors.cyan}${name}${colors.reset}${' '.repeat(30)}`);
    
    try {
      // Run proximity analysis
      const proximityResult = await pythonService.runProximityAnalysis(id, FEATURE_LAYERS, updateDb);
      if (proximityResult.error) {
        console.error(`\n${colors.red}Error in proximity analysis for ${name}: ${proximityResult.error}${colors.reset}`);
        results.proximity.failed++;
      } else {
        results.proximity.successful++;
      }
      
      // Run classification
      const classifyResult = await pythonService.runClassification(id, updateDb);
      if (classifyResult.error) {
        console.error(`\n${colors.red}Error in classification for ${name}: ${classifyResult.error}${colors.reset}`);
        results.classification.failed++;
      } else {
        results.classification.successful++;
      }
    } catch (error) {
      console.error(`\n${colors.red}Error processing ${name}: ${error.message}${colors.reset}`);
      results.proximity.failed++;
      results.classification.failed++;
    }
  }
  
  // Clear the progress line and return results
  process.stdout.write('\r' + ' '.repeat(80) + '\r');
  return results;
}

// Main function
async function main() {
  try {
    console.log(`${colors.blue}${colors.bright}Starting spatial analysis on all erratics...${colors.reset}`);
    
    // Get all erratics from the database
    const erratics = await db.Erratic.findAll({
      attributes: ['id', 'name']
    });
    
    if (erratics.length === 0) {
      console.error(`${colors.red}No erratics found in database. Aborting.${colors.reset}`);
      process.exit(1);
    }
    
    console.log(`${colors.green}Found ${erratics.length} erratics to process.${colors.reset}`);
    console.log(`${colors.yellow}This may take a while depending on the number of erratics.${colors.reset}`);
    
    // Process all erratics
    const startTime = new Date();
    const results = await processBatch(erratics, true);
    const endTime = new Date();
    const runtime = (endTime - startTime) / 1000; // in seconds
    
    // Display results
    console.log(`\n${colors.bright}${colors.green}Spatial analysis complete!${colors.reset}`);
    console.log(`${colors.blue}Runtime: ${runtime.toFixed(2)} seconds${colors.reset}`);
    console.log(`${colors.blue}Total erratics processed: ${results.total}${colors.reset}`);
    console.log(`${colors.blue}Proximity analysis: ${colors.green}${results.proximity.successful} successful${colors.reset}, ${colors.red}${results.proximity.failed} failed${colors.reset}`);
    console.log(`${colors.blue}Classification: ${colors.green}${results.classification.successful} successful${colors.reset}, ${colors.red}${results.classification.failed} failed${colors.reset}`);
    
    process.exit(0);
  } catch (error) {
    console.error(`${colors.red}Error running spatial analysis: ${error.message}${colors.reset}`);
    console.error(error.stack);
    process.exit(1);
  }
}

// Call the main function
main(); 