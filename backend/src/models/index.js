const fs = require('fs');
const path = require('path');
const Sequelize = require('sequelize');
const basename = path.basename(__filename);
const env = process.env.NODE_ENV || 'development';
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

// Create Sequelize instance with database config from environment variables
const sequelize = new Sequelize(
  dbName,
  dbUser,
  dbPassword,
  {
    host: dbHost,
    port: dbPort,
    dialect: 'postgres',
    dialectOptions: {
      ssl: env === 'production'
    },
    logging: env === 'development' ? console.log : false
  }
);

// Load all model files in this directory
fs.readdirSync(__dirname)
  .filter(file => {
    return (
      file.indexOf('.') !== 0 &&
      file !== basename &&
      file.slice(-3) === '.js'
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