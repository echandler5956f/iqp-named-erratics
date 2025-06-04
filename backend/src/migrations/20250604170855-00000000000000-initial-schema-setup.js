'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up (queryInterface, Sequelize) {
    // Create Erratics Table
    await queryInterface.createTable('Erratics', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      name: {
        type: Sequelize.STRING,
        allowNull: false
      },
      location: {
        type: Sequelize.GEOMETRY('POINT'), // Assuming PostGIS is enabled
        allowNull: false
      },
      elevation: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      size_meters: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      rock_type: {
        type: Sequelize.STRING(100),
        allowNull: true
      },
      estimated_age: {
        type: Sequelize.STRING(100),
        allowNull: true
      },
      discovery_date: {
        type: Sequelize.DATE,
        allowNull: true
      },
      description: {
        type: Sequelize.TEXT,
        allowNull: true
      },
      cultural_significance: {
        type: Sequelize.TEXT,
        allowNull: true
      },
      historical_notes: {
        type: Sequelize.TEXT,
        allowNull: true
      },
      image_url: {
        type: Sequelize.STRING,
        allowNull: true
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });

    // Create Users Table
    await queryInterface.createTable('Users', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      username: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: true
      },
      email: {
        type: Sequelize.STRING,
        allowNull: false,
        unique: true,
        validate: {
          isEmail: true
        }
      },
      password: {
        type: Sequelize.STRING,
        allowNull: false
      },
      is_admin: {
        type: Sequelize.BOOLEAN,
        defaultValue: false
      },
      last_login: {
        type: Sequelize.DATE,
        allowNull: true
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });

    // Create ErraticAnalyses Table
    await queryInterface.createTable('ErraticAnalyses', {
      erraticId: {
        type: Sequelize.INTEGER,
        primaryKey: true,
        allowNull: false,
        references: {
          model: 'Erratics', // Name of the target table
          key: 'id'
        },
        onUpdate: 'CASCADE',
        onDelete: 'CASCADE'
      },
      usage_type: {
        type: Sequelize.ARRAY(Sequelize.STRING(100)),
        allowNull: true
      },
      cultural_significance_score: {
        type: Sequelize.INTEGER,
        allowNull: true
      },
      has_inscriptions: {
        type: Sequelize.BOOLEAN,
        allowNull: true
      },
      accessibility_score: {
        type: Sequelize.INTEGER,
        allowNull: true
      },
      size_category: {
        type: Sequelize.STRING(50),
        allowNull: true
      },
      nearest_water_body_dist: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      nearest_settlement_dist: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      nearest_colonial_settlement_dist: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      nearest_road_dist: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      nearest_colonial_road_dist: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      nearest_native_territory_dist: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      elevation_category: {
        type: Sequelize.STRING(50),
        allowNull: true
      },
      geological_type: {
        type: Sequelize.STRING(100),
        allowNull: true
      },
      estimated_displacement_dist: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      ruggedness_tri: {
        type: Sequelize.FLOAT,
        allowNull: true
      },
      terrain_landform: {
        type: Sequelize.STRING(100),
        allowNull: true
      },
      terrain_slope_position: {
        type: Sequelize.STRING(100),
        allowNull: true
      },
      vector_embedding: {
        type: 'VECTOR(384)', // Raw type for pgvector
        allowNull: true
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });

    // Create ErraticMedia Table
    await queryInterface.createTable('ErraticMedia', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      erraticId: {
        type: Sequelize.INTEGER,
        allowNull: false,
        references: {
          model: 'Erratics',
          key: 'id'
        },
        onUpdate: 'CASCADE',
        onDelete: 'CASCADE' // Added for consistency
      },
      media_type: {
        type: Sequelize.ENUM('image', 'video', 'document', 'other'),
        allowNull: false,
        defaultValue: 'image'
      },
      url: {
        type: Sequelize.STRING,
        allowNull: false
      },
      title: {
        type: Sequelize.STRING,
        allowNull: true
      },
      description: {
        type: Sequelize.TEXT,
        allowNull: true
      },
      credit: {
        type: Sequelize.STRING,
        allowNull: true
      },
      capture_date: {
        type: Sequelize.DATE,
        allowNull: true
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });

    // Create ErraticReferences Table
    await queryInterface.createTable('ErraticReferences', {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      erraticId: {
        type: Sequelize.INTEGER,
        allowNull: false,
        references: {
          model: 'Erratics',
          key: 'id'
        },
        onUpdate: 'CASCADE',
        onDelete: 'CASCADE' // Added for consistency
      },
      reference_type: {
        type: Sequelize.ENUM('article', 'book', 'paper', 'website', 'other'),
        allowNull: false
      },
      title: {
        type: Sequelize.STRING,
        allowNull: false
      },
      authors: {
        type: Sequelize.STRING,
        allowNull: true
      },
      publication: {
        type: Sequelize.STRING,
        allowNull: true
      },
      year: {
        type: Sequelize.INTEGER,
        allowNull: true
      },
      url: {
        type: Sequelize.STRING,
        allowNull: true
      },
      doi: {
        type: Sequelize.STRING,
        allowNull: true
      },
      description: {
        type: Sequelize.TEXT,
        allowNull: true
      },
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE
      }
    });

    // Add any necessary indexes here if not automatically created by references/PKs
    // Example: await queryInterface.addIndex('Erratics', ['rock_type']);
  },

  async down (queryInterface, Sequelize) {
    // Drop tables in reverse order of creation due to foreign key constraints
    await queryInterface.dropTable('ErraticReferences');
    await queryInterface.dropTable('ErraticMedia');
    await queryInterface.dropTable('ErraticAnalyses');
    await queryInterface.dropTable('Users');
    await queryInterface.dropTable('Erratics');
    // If you created custom ENUM types manually for PG (less common with Sequelize 6+ which handles it better),
    // you might need to drop them here too, e.g.:
    // await queryInterface.sequelize.query('DROP TYPE IF EXISTS "public"."enum_ErraticMedia_media_type";');
    // await queryInterface.sequelize.query('DROP TYPE IF EXISTS "public"."enum_ErraticReferences_reference_type";');
  }
}; 