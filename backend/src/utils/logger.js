const winston = require('winston');
const path = require('path');
require('winston-daily-rotate-file'); // For log rotation

// Define log directory
const logDir = path.join(__dirname, '../../logs'); // Store logs in project_root/logs or backend/logs

// Define log levels
const levels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3, // For HTTP request logging (e.g., with Morgan)
  verbose: 4,
  debug: 5,
  silly: 6
};

// Determine log level based on NODE_ENV
const level = () => {
  const env = process.env.NODE_ENV || 'development';
  const isDevelopment = env === 'development';
  return isDevelopment ? 'debug' : 'warn'; // More verbose in dev, less in prod
};

// Define colors for development console logging
const colors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  http: 'magenta',
  debug: 'white'
};
winston.addColors(colors);

// Define log format
const consoleFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss:ms' }),
  winston.format.colorize({ all: true }), // Colorize the entire log message
  winston.format.printf(
    (info) => `${info.timestamp} ${info.level}: ${info.message}`
  )
);

const fileFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.json() // Log in JSON format to files
);

// Define transports
const transports = [
  // Console transport (for development or general visibility)
  new winston.transports.Console({
    format: consoleFormat,
    level: process.env.NODE_ENV === 'production' ? 'info' : 'debug', // Adjust level for console in prod vs dev
  }),
  
  // File transport for all logs (rotated daily)
  new winston.transports.DailyRotateFile({
    filename: path.join(logDir, 'app-%DATE%.log'),
    datePattern: 'YYYY-MM-DD',
    zippedArchive: true, // Compress old log files
    maxSize: '20m',      // Rotate if file size exceeds 20MB
    maxFiles: '14d',     // Keep logs for 14 days
    format: fileFormat,
    level: 'info', // Log info and above to the general app log
  }),
  
  // File transport for error logs only (rotated daily)
  new winston.transports.DailyRotateFile({
    filename: path.join(logDir, 'error-%DATE%.log'),
    datePattern: 'YYYY-MM-DD',
    zippedArchive: true,
    maxSize: '20m',
    maxFiles: '30d',     // Keep error logs for 30 days
    format: fileFormat,
    level: 'error', // Only log errors to this file
  })
];

// Create the logger instance
const logger = winston.createLogger({
  level: level(), // Use the dynamic level function
  levels,          // Use custom defined levels
  transports
});

// Create a stream object with a 'write' function that will be used by Morgan (for HTTP request logging)
logger.stream = {
  write: (message) => {
    logger.http(message.substring(0, message.lastIndexOf('\n')));
  },
};

module.exports = logger; 