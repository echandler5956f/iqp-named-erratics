require('dotenv').config({ path: require('path').resolve(__dirname, '../../../.env') });

// Purpose: Script to create an admin user.
// Usage: node backend/src/scripts/createAdminUser.js <username> <email> <password> [--admin]

const { User, sequelize } = require('../models'); // Adjust path if your models/index.js is elsewhere
const bcrypt = require('bcrypt'); // Not strictly needed here due to hook, but good for awareness

async function createAdmin() {
  const args = process.argv.slice(2);
  if (args.length < 3) {
    console.error('Usage: node createAdminUser.js <username> <email> <password> [--admin]');
    process.exit(1);
  }

  const username = args[0];
  const email = args[1];
  const password = args[2];
  const isAdmin = args.includes('--admin');

  try {
    // await sequelize.sync(); // Sync can be dangerous, models should be synced by migrations. Assuming DB is set up.
    // Initialize DB connection (Sequelize does this implicitly on first query if configured)
    // For a standalone script, it's good to ensure connection is established.
    // However, '../models' should export the sequelize instance that's already configured & connected by the main app.

    const existingUser = await User.findOne({ where: { username } });
    if (existingUser) {
      console.log(`User "${username}" already exists.`);
      let updated = false;
      if (!existingUser.is_admin && isAdmin) {
        existingUser.is_admin = true;
        updated = true;
      }
      // This script will not update password or email for existing user to keep it simple.
      // For password changes, the beforeUpdate hook would re-hash if password field is dirty.
      if (updated) {
        await existingUser.save();
        console.log(`User "${username}" updated. is_admin: ${existingUser.is_admin}.`);
      } else {
        console.log(`No updates applied to existing user "${username}". is_admin is already: ${existingUser.is_admin}`);
      }
      process.exit(0);
      return;
    }
    
    console.log(`Creating user: ${username}, Email: ${email}, Is Admin: ${isAdmin}`);

    const newUser = await User.create({
      username,
      email,
      password, // The 'beforeCreate' hook in User.js will hash this
      is_admin: isAdmin,
    });

    console.log(`User "${newUser.username}" created successfully with ID: ${newUser.id}.`);
    console.log(`is_admin flag set to: ${newUser.is_admin}`);

  } catch (error) {
    console.error('Error creating/updating admin user:', error);
    if (error.errors && Array.isArray(error.errors)) {
        error.errors.forEach(err => console.error(`- ${err.message}`));
    }
    process.exit(1);
  } finally {
    await sequelize.close();
    console.log('Database connection closed.');
  }
}

createAdmin(); 