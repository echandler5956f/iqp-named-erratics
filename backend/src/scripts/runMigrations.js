const path = require('path');
const fs = require('fs');
const dotenv = require('dotenv');
const { Sequelize } = require('sequelize');

// Load environment variables
const envPath = path.resolve(__dirname, '../../.env');
dotenv.config({ path: envPath });

// Configure database connection
const sequelize = new Sequelize(
  process.env.DB_NAME,
  process.env.DB_USER,
  process.env.DB_PASSWORD,
  {
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    dialect: 'postgres',
    logging: console.log
  }
);

// Get migration directory
const migrationsDir = path.join(__dirname, '../migrations');

// Function to apply migrations
async function runMigrations() {
  try {
    // Test database connection
    await sequelize.authenticate();
    console.log('Connected to the database successfully.');
    
    // Create migrations table if it doesn't exist
    await sequelize.query(`
      CREATE TABLE IF NOT EXISTS "Migrations" (
        "id" SERIAL PRIMARY KEY,
        "name" VARCHAR(255) NOT NULL UNIQUE,
        "appliedAt" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
    `);
    
    // Get list of migration files sorted by name
    const migrationFiles = fs.readdirSync(migrationsDir)
      .filter(file => file.endsWith('.js'))
      .sort();
    
    if (migrationFiles.length === 0) {
      console.log('No migration files found.');
      return;
    }
    
    // Get list of applied migrations
    const [appliedMigrations] = await sequelize.query(
      'SELECT name FROM "Migrations" ORDER BY name;'
    );
    
    const appliedMigrationNames = appliedMigrations.map(m => m.name);
    
    // Filter out already applied migrations
    const pendingMigrations = migrationFiles.filter(
      file => !appliedMigrationNames.includes(file)
    );
    
    if (pendingMigrations.length === 0) {
      console.log('No pending migrations.');
      return;
    }
    
    console.log(`Found ${pendingMigrations.length} pending migrations.`);
    
    // Apply each migration in a transaction
    for (const migrationFile of pendingMigrations) {
      const migration = require(path.join(migrationsDir, migrationFile));
      
      console.log(`Applying migration: ${migrationFile}`);
      
      const transaction = await sequelize.transaction();
      
      try {
        // Apply the migration
        await migration.up(sequelize.getQueryInterface(), Sequelize);
        
        // Record the migration
        await sequelize.query(
          'INSERT INTO "Migrations" (name) VALUES (?);',
          {
            replacements: [migrationFile],
            transaction
          }
        );
        
        await transaction.commit();
        console.log(`Migration ${migrationFile} applied successfully.`);
      } catch (error) {
        await transaction.rollback();
        console.error(`Error applying migration ${migrationFile}:`, error);
        process.exit(1);
      }
    }
    
    console.log('All migrations applied successfully.');
  } catch (error) {
    console.error('Error running migrations:', error);
    process.exit(1);
  } finally {
    await sequelize.close();
  }
}

// Run migrations
runMigrations(); 