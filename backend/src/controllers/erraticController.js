const { Op } = require('sequelize');
const db = require('../models');

// Get all erratics with optional filtering
exports.getAllErratics = async (req, res) => {
  try {
    const { rockType, minSize, maxSize, search } = req.query;
    
    // Build filter conditions
    const whereConditions = {};
    
    if (rockType) {
      whereConditions.rock_type = rockType;
    }
    
    if (minSize) {
      whereConditions.size_meters = {
        ...whereConditions.size_meters,
        [Op.gte]: parseFloat(minSize)
      };
    }
    
    if (maxSize) {
      whereConditions.size_meters = {
        ...whereConditions.size_meters,
        [Op.lte]: parseFloat(maxSize)
      };
    }
    
    if (search) {
      whereConditions[Op.or] = [
        { name: { [Op.iLike]: `%${search}%` } },
        { description: { [Op.iLike]: `%${search}%` } },
        { rock_type: { [Op.iLike]: `%${search}%` } }
      ];
    }
    
    const erratics = await db.Erratic.findAll({
      where: whereConditions,
      attributes: [
        'id', 
        'name', 
        'rock_type', 
        'size_meters',
        'estimated_age',
        'elevation',
        'description',
        'cultural_significance',
        'image_url',
        [db.sequelize.fn('ST_AsGeoJSON', db.sequelize.col('location')), 'location']
      ]
    });
    
    // Parse the GeoJSON string returned by PostGIS
    const formattedErratics = erratics.map(erratic => {
      const plainErratic = erratic.get({ plain: true });
      plainErratic.location = JSON.parse(plainErratic.location);
      return plainErratic;
    });
    
    res.json(formattedErratics);
  } catch (error) {
    console.error('Error fetching erratics:', error);
    res.status(500).json({ message: 'Error fetching erratics', error: error.message });
  }
};

// Get a single erratic by ID
exports.getErraticById = async (req, res) => {
  try {
    const { id } = req.params;
    
    const erratic = await db.Erratic.findByPk(id, {
      attributes: {
        include: [[db.sequelize.fn('ST_AsGeoJSON', db.sequelize.col('location')), 'location']]
      },
      include: [
        {
          model: db.ErraticMedia,
          as: 'media',
          attributes: ['id', 'media_type', 'url', 'title', 'description', 'credit']
        },
        {
          model: db.ErraticReference,
          as: 'references',
          attributes: ['id', 'reference_type', 'title', 'authors', 'publication', 'year', 'url']
        }
      ]
    });
    
    if (!erratic) {
      return res.status(404).json({ message: 'Erratic not found' });
    }
    
    // Parse the GeoJSON location
    const formattedErratic = erratic.get({ plain: true });
    formattedErratic.location = JSON.parse(formattedErratic.location);
    
    res.json(formattedErratic);
  } catch (error) {
    console.error('Error fetching erratic:', error);
    res.status(500).json({ message: 'Error fetching erratic', error: error.message });
  }
};

// Get erratics within a certain radius
exports.getNearbyErratics = async (req, res) => {
  try {
    const { lat, lng, radius = 50 } = req.query; // radius in kilometers
    
    if (!lat || !lng) {
      return res.status(400).json({ message: 'Latitude and longitude are required' });
    }
    
    const erratics = await db.Erratic.findAll({
      attributes: [
        'id', 
        'name',
        'rock_type',
        'size_meters',
        'image_url',
        [db.sequelize.fn('ST_AsGeoJSON', db.sequelize.col('location')), 'location'],
        [
          db.sequelize.fn(
            'ST_DistanceSphere', 
            db.sequelize.col('location'), 
            db.sequelize.fn('ST_MakePoint', parseFloat(lng), parseFloat(lat))
          ),
          'distance'
        ]
      ],
      where: db.sequelize.where(
        db.sequelize.fn(
          'ST_DistanceSphere', 
          db.sequelize.col('location'), 
          db.sequelize.fn('ST_MakePoint', parseFloat(lng), parseFloat(lat))
        ),
        {
          [Op.lte]: parseFloat(radius) * 1000 // Convert kilometers to meters
        }
      ),
      order: [
        [db.sequelize.literal('distance'), 'ASC']
      ]
    });
    
    const formattedErratics = erratics.map(erratic => {
      const plainErratic = erratic.get({ plain: true });
      plainErratic.location = JSON.parse(plainErratic.location);
      plainErratic.distance = parseFloat(plainErratic.distance) / 1000; // Convert to kilometers
      return plainErratic;
    });
    
    res.json(formattedErratics);
  } catch (error) {
    console.error('Error fetching nearby erratics:', error);
    res.status(500).json({ message: 'Error fetching nearby erratics', error: error.message });
  }
};

// Create a new erratic
exports.createErratic = async (req, res) => {
  try {
    const { 
      name, latitude, longitude, elevation, size_meters, rock_type,
      estimated_age, discovery_date, description, cultural_significance,
      historical_notes, image_url
    } = req.body;
    
    // Validate required fields
    if (!name || !latitude || !longitude) {
      return res.status(400).json({ message: 'Name, latitude, and longitude are required' });
    }
    
    // Create PostGIS point from coordinates
    const location = db.sequelize.literal(
      `ST_SetSRID(ST_MakePoint(${parseFloat(longitude)}, ${parseFloat(latitude)}), 4326)`
    );
    
    // Create the erratic
    const erratic = await db.Erratic.create({
      name,
      location,
      elevation: elevation ? parseFloat(elevation) : null,
      size_meters: size_meters ? parseFloat(size_meters) : null,
      rock_type,
      estimated_age,
      discovery_date: discovery_date ? new Date(discovery_date) : null,
      description,
      cultural_significance,
      historical_notes,
      image_url
    });
    
    res.status(201).json({
      message: 'Erratic created successfully',
      id: erratic.id
    });
  } catch (error) {
    console.error('Error creating erratic:', error);
    res.status(500).json({ message: 'Error creating erratic', error: error.message });
  }
};

// Update an existing erratic
exports.updateErratic = async (req, res) => {
  try {
    const { id } = req.params;
    const { 
      name, latitude, longitude, elevation, size_meters, rock_type,
      estimated_age, discovery_date, description, cultural_significance,
      historical_notes, image_url
    } = req.body;
    
    // Find the erratic first
    const erratic = await db.Erratic.findByPk(id);
    if (!erratic) {
      return res.status(404).json({ message: 'Erratic not found' });
    }
    
    // Prepare update data
    const updateData = {};
    
    if (name) updateData.name = name;
    if (latitude && longitude) {
      updateData.location = db.sequelize.literal(
        `ST_SetSRID(ST_MakePoint(${parseFloat(longitude)}, ${parseFloat(latitude)}), 4326)`
      );
    }
    if (elevation !== undefined) updateData.elevation = elevation ? parseFloat(elevation) : null;
    if (size_meters !== undefined) updateData.size_meters = size_meters ? parseFloat(size_meters) : null;
    if (rock_type !== undefined) updateData.rock_type = rock_type;
    if (estimated_age !== undefined) updateData.estimated_age = estimated_age;
    if (discovery_date !== undefined) updateData.discovery_date = discovery_date ? new Date(discovery_date) : null;
    if (description !== undefined) updateData.description = description;
    if (cultural_significance !== undefined) updateData.cultural_significance = cultural_significance;
    if (historical_notes !== undefined) updateData.historical_notes = historical_notes;
    if (image_url !== undefined) updateData.image_url = image_url;
    
    // Update the erratic
    await erratic.update(updateData);
    
    res.json({ message: 'Erratic updated successfully' });
  } catch (error) {
    console.error('Error updating erratic:', error);
    res.status(500).json({ message: 'Error updating erratic', error: error.message });
  }
};

// Delete an erratic
exports.deleteErratic = async (req, res) => {
  try {
    const { id } = req.params;
    
    // Find the erratic first
    const erratic = await db.Erratic.findByPk(id);
    if (!erratic) {
      return res.status(404).json({ message: 'Erratic not found' });
    }
    
    // Delete associated media and references
    await db.ErraticMedia.destroy({ where: { erraticId: id } });
    await db.ErraticReference.destroy({ where: { erraticId: id } });
    
    // Delete the erratic
    await erratic.destroy();
    
    res.json({ message: 'Erratic deleted successfully' });
  } catch (error) {
    console.error('Error deleting erratic:', error);
    res.status(500).json({ message: 'Error deleting erratic', error: error.message });
  }
}; 