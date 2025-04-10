const express = require('express');
const cors = require('cors');
const path = require('path');
const dotenv = require('dotenv');

// Load environment variables from .env file
const envPath = path.resolve(__dirname, '../.env');
console.log('Loading environment from:', envPath);
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

// Now import modules that depend on environment variables
const { initializeDatabase } = require('./utils/dbInit');
const erraticRoutes = require('./routes/erraticRoutes');
const authRoutes = require('./routes/authRoutes');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/erratics', erraticRoutes);
app.use('/api/auth', authRoutes);

// Health check route
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' });
});

// Root path handler - helps users find the frontend
app.get('/', (req, res) => {
  // In development: redirect to the frontend URL
  if (process.env.NODE_ENV === 'development') {
    res.send(`
      <html>
        <head>
          <title>Glacial Erratics API Server</title>
          <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }
            h1 { color: #2d3748; }
            a { color: #4299e1; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .endpoints { background: #f7fafc; padding: 15px; border-radius: 5px; margin: 20px 0; }
            code { background: #edf2f7; padding: 2px 5px; border-radius: 3px; }
          </style>
        </head>
        <body>
          <h1>Glacial Erratics - API Server</h1>
          <p>This is the backend API server for the Glacial Erratics Map application.</p>
          <p>To view the frontend application, go to: <a href="http://localhost:5173">http://localhost:5173</a></p>
          
          <div class="endpoints">
            <h2>Available API Endpoints:</h2>
            <ul>
              <li><code>GET /api/health</code> - Check if the API is running</li>
              <li><code>GET /api/erratics</code> - Get all erratics</li>
              <li><code>GET /api/erratics/:id</code> - Get a specific erratic by ID</li>
              <li><code>GET /api/erratics/nearby?lat=X&lng=Y&radius=Z</code> - Get erratics near a location</li>
            </ul>
          </div>
        </body>
      </html>
    `);
  } else {
    // In production, we could serve the static frontend files here
    res.redirect('/api/health');
  }
});

// Catch-all route for undefined API routes
app.use('/api/*', (req, res) => {
  res.status(404).json({ message: 'API endpoint not found' });
});

// Initialize database and start server
async function startServer() {
  try {
    // Initialize database
    const dbInitialized = await initializeDatabase();
    if (!dbInitialized) {
      console.error('Failed to initialize database. Exiting...');
      process.exit(1);
    }
    
    // Start server
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error('Error starting server:', error);
    process.exit(1);
  }
}

startServer();
