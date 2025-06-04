# Database Migrations

This directory contains Sequelize migrations for managing the database schema across different environments and versions of the application.

## Migration Files

Migrations are JavaScript files that define how to change the database schema. Each migration has two functions:
- `up()`: Applied when running the migration
- `down()`: Applied when rolling back the migration

Migration files are named with a sequential prefix to ensure they're run in the correct order.

## Creating New Migrations

### Using the Template

For most schema changes, you can use the template file `../migration_templates/migration-template.js` as a starting point:

1. Copy the template file into the `migrations` directory and rename it with the next sequential number (e.g., `YYYYMMDDHHMMSS-descriptive-name.js`).
2. Update the file's description comment
3. Modify the `up()` and `down()` functions to implement your schema changes
4. Test both applying and rolling back your migration

### Best Practices

1. **Always check if columns/tables exist before modifying them** to make migrations rerunnable and idempotent
2. **Include detailed comments** explaining the purpose of each change
3. **Use explicit Sequelize data types** with constraints where applicable
4. **Test both `up()` and `down()` functions** to ensure they work correctly
5. **Handle errors gracefully** with try/catch blocks and descriptive console output

## Running Migrations

### Using Node Scripts

The project includes scripts in `package.json` for running migrations:

```bash
# Apply all pending migrations
npm run db:migrate

# Undo the most recent migration
npm run db:migrate:undo

# Undo all migrations
npm run db:migrate:undo:all
```

### Using Sequelize CLI Directly

You can also use Sequelize CLI commands directly:

```bash
# Apply all pending migrations
npx sequelize-cli db:migrate

# Undo the most recent migration
npx sequelize-cli db:migrate:undo

# Undo all migrations
npx sequelize-cli db:migrate:undo:all
```

### Configuration

Sequelize CLI uses the configuration specified in your project's `.sequelizerc` file, which should point to a configuration file (e.g., `backend/config/database.js`) for database connections. Make sure this setup is correct for your environment.

## Verifying Migrations

After running migrations, you can verify the changes using:

```bash
# Connect to your database and describe the table structure
psql -h localhost -U your_username -d your_database -c "\d \"ErraticAnalyses\""
```

## Troubleshooting

If migrations fail:

1. Check the console output for specific error messages
2. Verify that your database connection is working
3. Ensure all tables and columns referenced in the migration exist
4. Check for syntax errors in your migration file

## Additional Resources

- [Sequelize Migrations Documentation](https://sequelize.org/master/manual/migrations.html)
- [Project Database Structure](../../README.md#database-structure) 