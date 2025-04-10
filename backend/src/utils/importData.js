const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { Sequelize } = require('sequelize');
const db = require('../models');

/**
 * Import erratic data from a CSV file
 * @param {string} filePath Path to the CSV file
 * @returns {Promise<number>} Number of records imported
 */
async function importErraticsFromCSV(filePath) {
  return new Promise((resolve, reject) => {
    const erratics = [];
    
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => {
        // Validate required fields
        if (!row.name || !row.latitude || !row.longitude) {
          console.warn('Skipping row with missing required fields:', row);
          return;
        }
        
        // Parse numeric values
        const latitude = parseFloat(row.latitude);
        const longitude = parseFloat(row.longitude);
        const elevation = row.elevation ? parseFloat(row.elevation) : null;
        const size_meters = row.size_meters ? parseFloat(row.size_meters) : null;
        
        // Skip rows with invalid coordinates
        if (isNaN(latitude) || isNaN(longitude)) {
          console.warn('Skipping row with invalid coordinates:', row);
          return;
        }
        
        // Create point geometry using PostGIS
        const location = Sequelize.literal(
          `ST_SetSRID(ST_MakePoint(${longitude}, ${latitude}), 4326)`
        );
        
        // Prepare discovery date if present
        let discovery_date = null;
        if (row.discovery_date) {
          try {
            discovery_date = new Date(row.discovery_date);
            // Check if date is valid
            if (isNaN(discovery_date.getTime())) {
              discovery_date = null;
            }
          } catch (error) {
            console.warn('Invalid discovery date format:', row.discovery_date);
          }
        }
        
        // Add to the batch
        erratics.push({
          name: row.name,
          location,
          elevation,
          size_meters,
          rock_type: row.rock_type || null,
          estimated_age: row.estimated_age || null,
          discovery_date,
          description: row.description || null,
          cultural_significance: row.cultural_significance || null,
          historical_notes: row.historical_notes || null,
          image_url: row.image_url || null
        });
      })
      .on('end', async () => {
        try {
          // Insert into database in batches
          if (erratics.length > 0) {
            await db.Erratic.bulkCreate(erratics);
            console.log(`Successfully imported ${erratics.length} erratics`);
            resolve(erratics.length);
          } else {
            console.warn('No valid data found in CSV file');
            resolve(0);
          }
        } catch (error) {
          console.error('Error importing data:', error);
          reject(error);
        }
      })
      .on('error', (error) => {
        console.error('Error reading CSV file:', error);
        reject(error);
      });
  });
}

module.exports = { importErraticsFromCSV }; 