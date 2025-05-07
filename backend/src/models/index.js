const fs = require('fs');
const path = require('path');
const Sequelize = require('sequelize');
const process = require('process');
const basename = path.basename(__filename);
const db = {};

// Get connection parameters from environment variables
const dbName = process.env.DB_NAME;
const dbUser = process.env.DB_USER;
const dbPassword = String(process.env.DB_PASSWORD); // Ensure password is a string
const dbHost = process.env.DB_HOST;
const dbPort = process.env.DB_PORT;

// Debug output (remove in production)
console.log('Connecting to database with the following parameters:');
console.log(`Database: ${dbName}`);
console.log(`User: ${dbUser}`);
console.log(`Host: ${dbHost}:${dbPort}`);
console.log(`Password length: ${dbPassword ? dbPassword.length : 0}`);

// Restore original Sequelize initialization using env vars directly
const sequelize = new Sequelize(
  dbName,
  dbUser,
  dbPassword,
  {
    host: dbHost,
    port: dbPort,
    dialect: 'postgres',
    dialectOptions: {
      // ssl: process.env.NODE_ENV === 'production' ? { require: true, rejectUnauthorized: false } : false // Example SSL config if needed
    },
    logging: process.env.NODE_ENV === 'development' ? console.log : false // Keep logging for dev
  }
);

// Load all model files in this directory
fs
  .readdirSync(__dirname)
  .filter(file => {
    return (
      file.indexOf('.') !== 0 &&
      file !== basename &&
      file.slice(-3) === '.js' &&
      file.indexOf('.test.js') === -1
    );
  })
  .forEach(file => {
    const model = require(path.join(__dirname, file))(sequelize, Sequelize.DataTypes);
    db[model.name] = model;
  });

// Associate models if they have associations
Object.keys(db).forEach(modelName => {
  if (db[modelName].associate) {
    db[modelName].associate(db);
  }
});

db.sequelize = sequelize;
db.Sequelize = Sequelize;

module.exports = db; 