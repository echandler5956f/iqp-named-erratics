const request = require('supertest');
const { expect } = require('chai');
const sinon = require('sinon');
const proxyquire = require('proxyquire').noCallThru();
const express = require('express');

// Import mock services
const pythonServiceMock = require('../mocks/pythonService.mock');

// Create a custom AnalysisController class for testing
class AnalysisControllerTest {
  constructor() {
    // Bind methods to the instance
    this._processBatchAnalysis = this._processBatchAnalysis.bind(this);
    this.getProximityAnalysis = this.getProximityAnalysis.bind(this);
    this.batchProximityAnalysis = this.batchProximityAnalysis.bind(this);
  }

  async getProximityAnalysis(req, res) {
    const erraticId = parseInt(req.params.id);
    
    if (isNaN(erraticId)) {
      return res.status(400).json({ error: 'Invalid erratic ID' });
    }
    
    const featureLayers = req.query.features ? req.query.features.split(',') : [];
    const updateDb = req.query.update === 'true';
    
    try {
      const results = await pythonServiceMock.runProximityAnalysis(erraticId, featureLayers, updateDb);
      
      if (results.error) {
        return res.status(404).json({ error: results.error });
      }
      
      res.json(results);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  async batchProximityAnalysis(req, res) {
    const { erraticIds, featureLayers, updateDb } = req.body;
    
    if (!Array.isArray(erraticIds) || erraticIds.length === 0) {
      return res.status(400).json({ error: 'Invalid or empty erratic ID list' });
    }
    
    res.json({ 
      message: 'Batch analysis started', 
      job_id: `batch_${Date.now()}`,
      erratics_count: erraticIds.length
    });
    
    // Process in background
    this._processBatchAnalysis(erraticIds, featureLayers, updateDb)
      .catch(error => console.error('Error in batch processing:', error));
  }

  async _processBatchAnalysis(erraticIds, featureLayers = [], updateDb = true) {
    const results = {
      successful: [],
      failed: []
    };
    
    for (const erraticId of erraticIds) {
      try {
        const result = await pythonServiceMock.runProximityAnalysis(erraticId, featureLayers, updateDb);
        if (result.error) {
          results.failed.push({ id: erraticId, error: result.error });
        } else {
          results.successful.push(erraticId);
        }
      } catch (error) {
        results.failed.push({ id: erraticId, error: error.message });
      }
    }
    
    console.log(`Batch analysis completed. Successful: ${results.successful.length}, Failed: ${results.failed.length}`);
    return results;
  }
}

describe('Analysis Routes Integration Tests', function() {
  let app, server;
  
  // Set timeout for integration tests
  this.timeout(5000);
  
  before(function() {
    // Create a mock app for testing
    app = express();
    app.use(express.json());
    
    // Mock JWT auth middleware for protected routes
    const authMiddleware = {
      authenticateToken: (req, res, next) => next(),
      requireAdmin: (req, res, next) => next()
    };
    
    // Create controller directly
    const analysisController = new AnalysisControllerTest();
    
    // Create router with mocked dependencies
    const router = proxyquire('../../src/routes/analysisRoutes', {
      '../controllers/analysisController': analysisController,
      '../utils/auth': authMiddleware
    });
    
    // Mount the router
    app.use('/api/analysis', router);
    
    // Add error handler
    app.use((err, req, res, next) => {
      res.status(500).json({ error: err.message });
    });
  });
  
  after(function() {
    if (server) {
      server.close();
    }
  });
  
  describe('GET /api/analysis/proximity/:id', function() {
    it('should return 200 and proximity analysis for valid erratic ID', function(done) {
      request(app)
        .get('/api/analysis/proximity/1')
        .expect('Content-Type', /json/)
        .expect(200)
        .end((err, res) => {
          if (err) return done(err);
          
          expect(res.body).to.have.property('erratic_id', 1);
          expect(res.body).to.have.property('erratic_name', 'Plymouth Rock');
          expect(res.body).to.have.nested.property('proximity_analysis.elevation_category', 'lowland');
          done();
        });
    });
    
    it('should return 400 for invalid erratic ID', function(done) {
      request(app)
        .get('/api/analysis/proximity/invalid')
        .expect('Content-Type', /json/)
        .expect(400)
        .end((err, res) => {
          if (err) return done(err);
          
          expect(res.body).to.have.property('error', 'Invalid erratic ID');
          done();
        });
    });
    
    it('should return 404 for non-existent erratic ID', function(done) {
      request(app)
        .get('/api/analysis/proximity/999')
        .expect('Content-Type', /json/)
        .expect(404)
        .end((err, res) => {
          if (err) return done(err);
          
          expect(res.body).to.have.property('error', 'Erratic with ID 999 not found');
          done();
        });
    });
    
    it('should accept feature layers as query parameter', function(done) {
      // Spy on the pythonService
      const runProximityAnalysisSpy = sinon.spy(pythonServiceMock, 'runProximityAnalysis');
      
      request(app)
        .get('/api/analysis/proximity/1?features=water_bodies,settlements')
        .expect(200)
        .end((err, res) => {
          if (err) {
            runProximityAnalysisSpy.restore();
            return done(err);
          }
          
          expect(runProximityAnalysisSpy.calledOnce).to.be.true;
          expect(runProximityAnalysisSpy.firstCall.args[1]).to.deep.equal(['water_bodies', 'settlements']);
          
          runProximityAnalysisSpy.restore();
          done();
        });
    });
  });
  
  describe('POST /api/analysis/proximity/batch', function() {
    it('should return 200 and start batch analysis for valid request', function(done) {
      request(app)
        .post('/api/analysis/proximity/batch')
        .send({
          erraticIds: [1, 2],
          featureLayers: ['water_bodies', 'settlements'],
          updateDb: true
        })
        .expect('Content-Type', /json/)
        .expect(200)
        .end((err, res) => {
          if (err) return done(err);
          
          expect(res.body).to.have.property('message', 'Batch analysis started');
          expect(res.body).to.have.property('erratics_count', 2);
          done();
        });
    });
    
    it('should return 400 for invalid erratic IDs', function(done) {
      request(app)
        .post('/api/analysis/proximity/batch')
        .send({
          erraticIds: 'invalid',
          featureLayers: ['water_bodies', 'settlements']
        })
        .expect('Content-Type', /json/)
        .expect(400)
        .end((err, res) => {
          if (err) return done(err);
          
          expect(res.body).to.have.property('error', 'Invalid or empty erratic ID list');
          done();
        });
    });
    
    it('should return 400 for empty erratic IDs array', function(done) {
      request(app)
        .post('/api/analysis/proximity/batch')
        .send({
          erraticIds: [],
          featureLayers: ['water_bodies', 'settlements']
        })
        .expect('Content-Type', /json/)
        .expect(400)
        .end((err, res) => {
          if (err) return done(err);
          
          expect(res.body).to.have.property('error', 'Invalid or empty erratic ID list');
          done();
        });
    });
  });
}); 