require('dotenv').config({ path: require('path').resolve(__dirname, '../../.env') });

module.exports = {
  development: {
    username: process.env.DB_USER,
    password: String(process.env.DB_PASSWORD || ''),
    database: process.env.DB_NAME,
    host: process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT, 10),
    dialect: 'postgres',
    dialectOptions: {
      // ssl: { require: true, rejectUnauthorized: false } // Example SSL config
    }
  },
  test: {
    username: process.env.DB_USER_TEST || process.env.DB_USER,
    password: String(process.env.DB_PASSWORD_TEST || process.env.DB_PASSWORD || ''),
    database: process.env.DB_NAME_TEST || process.env.DB_NAME + '_test',
    host: process.env.DB_HOST_TEST || process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT_TEST || process.env.DB_PORT, 10),
    dialect: 'postgres',
    logging: false,
  },
  production: {
    username: process.env.DB_USER_PROD || process.env.DB_USER,
    password: String(process.env.DB_PASSWORD_PROD || process.env.DB_PASSWORD || ''),
    database: process.env.DB_NAME_PROD || process.env.DB_NAME,
    host: process.env.DB_HOST_PROD || process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT_PROD || process.env.DB_PORT, 10),
    dialect: 'postgres',
    logging: false,
    dialectOptions: {
      // ssl: { require: true, rejectUnauthorized: false } // Production SSL config
    }
  }
}; 