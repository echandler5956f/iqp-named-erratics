const { Op } = require('sequelize');
const db = require('../models');
const { Erratic, ErraticAnalysis, ErraticMedia, ErraticReference, sequelize } = db;

class ErraticService {
  _formatErratic(erraticInstance) {
    if (!erraticInstance) return null;
    const plainErratic = erraticInstance.get({ plain: true });

    // Flatten analysis fields into the main erratic object
    if (plainErratic.analysis) {
      for (const key in plainErratic.analysis) {
        if (plainErratic.analysis.hasOwnProperty(key)) {
          plainErratic[key] = plainErratic.analysis[key];
        }
      }
      delete plainErratic.analysis;
    }

    // Parse location from GeoJSON string to object
    if (plainErratic.location && typeof plainErratic.location === 'string') {
      try {
        plainErratic.location = JSON.parse(plainErratic.location);
      } catch (parseError) {
        console.error(`[ErraticService] Failed to parse location for erratic ${plainErratic.id}:`, parseError);
        plainErratic.location = null; // Or keep as string, or handle error differently
      }
    }
    return plainErratic;
  }

  async getAllErratics(queryParams) {
    const { rockType, minSize, maxSize, search } = queryParams;
    const whereConditions = {};

    if (rockType) whereConditions.rock_type = rockType;
    if (minSize) whereConditions.size_meters = { ...whereConditions.size_meters, [Op.gte]: parseFloat(minSize) };
    if (maxSize) whereConditions.size_meters = { ...whereConditions.size_meters, [Op.lte]: parseFloat(maxSize) };

    if (search) {
      const searchPattern = `%${search}%`;
      whereConditions[Op.or] = [
        { name: { [Op.iLike]: searchPattern } },
        { description: { [Op.iLike]: searchPattern } },
        { rock_type: { [Op.iLike]: searchPattern } },
      ];
    }

    const erratics = await Erratic.findAll({
      where: whereConditions,
      attributes: [
        'id', 'name', 'rock_type', 'size_meters', 'elevation', 'image_url',
        'estimated_age', 'description', 'cultural_significance',
        [sequelize.fn('ST_AsGeoJSON', sequelize.col('Erratic.location')), 'location']
      ],
      include: [{
        model: ErraticAnalysis,
        as: 'analysis',
        attributes: { exclude: ['erraticId', 'createdAt', 'updatedAt'] },
      }],
      // Add ordering, pagination options here if needed in the future
    });

    return erratics.map(this._formatErratic);
  }

  async getErraticById(id) {
    const erraticInstance = await Erratic.findByPk(id, {
      attributes: {
        include: [[sequelize.fn('ST_AsGeoJSON', sequelize.col('Erratic.location')), 'location']]
      },
      include: [
        { model: ErraticMedia, as: 'media', attributes: ['id', 'media_type', 'url', 'title', 'description', 'credit'] },
        { model: ErraticReference, as: 'references', attributes: ['id', 'reference_type', 'title', 'authors', 'publication', 'year', 'url'] },
        { model: ErraticAnalysis, as: 'analysis', attributes: { exclude: ['erraticId', 'createdAt', 'updatedAt'] } },
      ],
    });

    if (!erraticInstance) {
      const error = new Error('Erratic not found');
      error.statusCode = 404;
      throw error;
    }
    return this._formatErratic(erraticInstance);
  }

  async getNearbyErratics(lat, lng, radiusKm = 50) {
    if (!lat || !lng) {
      const error = new Error('Latitude and longitude are required');
      error.statusCode = 400;
      throw error;
    }

    const latitude = parseFloat(lat);
    const longitude = parseFloat(lng);
    const radiusMeters = parseFloat(radiusKm) * 1000;

    const point = sequelize.fn('ST_MakePoint', longitude, latitude);
    const distanceSphere = sequelize.fn('ST_DistanceSphere', sequelize.col('location'), point);

    const erratics = await Erratic.findAll({
      attributes: [
        'id', 'name', 'rock_type', 'size_meters', 'image_url',
        [sequelize.fn('ST_AsGeoJSON', sequelize.col('location')), 'location'],
        [distanceSphere, 'distance']
      ],
      where: sequelize.where(distanceSphere, { [Op.lte]: radiusMeters }),
      order: [[sequelize.literal('distance'), 'ASC']],
    });

    return erratics.map(erraticInstance => {
      const plainErratic = this._formatErratic(erraticInstance);
      if (plainErratic && plainErratic.distance !== undefined) {
        plainErratic.distance = parseFloat(plainErratic.distance) / 1000; // Convert meters to kilometers
      }
      return plainErratic;
    });
  }

  async createErratic(data) {
    const { name, latitude, longitude, ...otherData } = data;
    if (!name || latitude === undefined || longitude === undefined) {
      const error = new Error('Name, latitude, and longitude are required');
      error.statusCode = 400;
      throw error;
    }

    const location = sequelize.literal(`ST_SetSRID(ST_MakePoint(${parseFloat(longitude)}, ${parseFloat(latitude)}), 4326)`);
    
    const erraticData = {
      name,
      location,
      ...otherData,
      elevation: otherData.elevation ? parseFloat(otherData.elevation) : null,
      size_meters: otherData.size_meters ? parseFloat(otherData.size_meters) : null,
      discovery_date: otherData.discovery_date ? new Date(otherData.discovery_date) : null,
    };

    const erratic = await Erratic.create(erraticData);
    return { id: erratic.id, message: 'Erratic created successfully' };
  }

  async updateErratic(id, data) {
    const erratic = await Erratic.findByPk(id);
    if (!erratic) {
      const error = new Error('Erratic not found');
      error.statusCode = 404;
      throw error;
    }

    const { latitude, longitude, ...otherData } = data;
    const coreUpdateData = {};
    const analysisUpdateData = {};

    // Define model fields to separate updates
    const coreFields = Object.keys(Erratic.getAttributes());
    const analysisFields = ErraticAnalysis.getAttributes ? Object.keys(ErraticAnalysis.getAttributes()) : [];

    for (const key in otherData) {
      if (coreFields.includes(key) && key !== 'id' && key !== 'createdAt' && key !== 'updatedAt' && key !== 'location') {
        coreUpdateData[key] = otherData[key];
      } else if (analysisFields.includes(key) && key !== 'erraticId' && key !== 'createdAt' && key !== 'updatedAt') {
        analysisUpdateData[key] = otherData[key];
      }
    }
    
    // Handle location update if latitude and longitude are provided
    if (latitude !== undefined && longitude !== undefined) {
      coreUpdateData.location = sequelize.literal(`ST_SetSRID(ST_MakePoint(${parseFloat(longitude)}, ${parseFloat(latitude)}), 4326)`);
    }

    // Perform updates in a transaction for atomicity
    const transaction = await sequelize.transaction();
    try {
      if (Object.keys(coreUpdateData).length > 0) {
        await Erratic.update(coreUpdateData, { where: { id }, transaction });
      }

      if (Object.keys(analysisUpdateData).length > 0) {
        const [analysisRecord, created] = await ErraticAnalysis.findOrCreate({
          where: { erraticId: id },
          defaults: { ...analysisUpdateData, erraticId: id }, // Ensure erraticId is part of defaults for creation
          transaction,
        });
        if (!created) {
          await analysisRecord.update(analysisUpdateData, { transaction });
        }
      }
      await transaction.commit();
      return { message: 'Erratic updated successfully' };
    } catch (err) {
      await transaction.rollback();
      console.error('[ErraticService] Error updating erratic:', err);
      const error = new Error('Error updating erratic');
      error.statusCode = 500;
      throw error;
    }
  }

  async deleteErratic(id) {
    const erratic = await Erratic.findByPk(id);
    if (!erratic) {
      const error = new Error('Erratic not found');
      error.statusCode = 404;
      throw error;
    }

    // Assuming onDelete: 'CASCADE' is set in model associations for ErraticAnalysis, ErraticMedia, ErraticReference
    // If not, manual deletion would be needed here within a transaction:
    // await ErraticMedia.destroy({ where: { erraticId: id }, transaction });
    // await ErraticReference.destroy({ where: { erraticId: id }, transaction });
    // await ErraticAnalysis.destroy({ where: { erraticId: id }, transaction });
    
    await erratic.destroy(); // This will trigger CASCADE deletes if associations are configured correctly.
    return { message: 'Erratic deleted successfully' };
  }
}

module.exports = new ErraticService(); 