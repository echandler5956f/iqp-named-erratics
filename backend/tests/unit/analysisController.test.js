const { expect } = require('chai');
const sinon = require('sinon');
const proxyquire = require('proxyquire').noCallThru();

// Import mock services
const pythonServiceMock = require('../mocks/pythonService.mock');

// Use proxyquire to inject our mocks
const analysisController = proxyquire('../../src/controllers/analysisController', {
  '../services/pythonService': pythonServiceMock
});

describe('Analysis Controller', function() {
  let req, res, next;
  
  beforeEach(function() {
    // Setup request and response objects
    req = {
      params: {},
      query: {},
      body: {}
    };
    
    res = {
      status: sinon.stub().returns({ json: sinon.spy() }),
      json: sinon.spy()
    };
    
    next = sinon.spy();
  });
  
  afterEach(function() {
    sinon.restore();
  });
  
  describe('getProximityAnalysis', function() {
    it('should return proximity analysis for a valid erratic ID', async function() {
      // Arrange
      req.params.id = '1';
      
      // Act
      await analysisController.getProximityAnalysis(req, res);
      
      // Assert
      expect(res.json.calledOnce).to.be.true;
      expect(res.json.firstCall.args[0]).to.have.property('erratic_id', 1);
      expect(res.json.firstCall.args[0]).to.have.property('erratic_name', 'Plymouth Rock');
      expect(res.json.firstCall.args[0]).to.have.nested.property('proximity_analysis.elevation_category', 'lowland');
    });
    
    it('should handle invalid erratic ID', async function() {
      // Arrange
      req.params.id = 'invalid';
      
      // Act
      await analysisController.getProximityAnalysis(req, res);
      
      // Assert
      expect(res.status.calledWith(400)).to.be.true;
      expect(res.status().json.calledWith(sinon.match({ error: 'Invalid erratic ID' }))).to.be.true;
    });
    
    it('should handle non-existent erratic ID', async function() {
      // Arrange
      req.params.id = '999';
      
      // Act
      await analysisController.getProximityAnalysis(req, res);
      
      // Assert
      expect(res.status.calledWith(404)).to.be.true;
      expect(res.status().json.calledWith(sinon.match({ error: 'Erratic with ID 999 not found' }))).to.be.true;
    });
    
    it('should extract feature layers from query parameters', async function() {
      // Arrange
      req.params.id = '1';
      req.query.features = 'water_bodies,settlements';
      
      // Spy on the pythonService
      const runProximityAnalysisSpy = sinon.spy(pythonServiceMock, 'runProximityAnalysis');
      
      // Act
      await analysisController.getProximityAnalysis(req, res);
      
      // Assert
      expect(runProximityAnalysisSpy.calledOnce).to.be.true;
      expect(runProximityAnalysisSpy.firstCall.args[1]).to.deep.equal(['water_bodies', 'settlements']);
    });
    
    it('should handle updateDb parameter', async function() {
      // Arrange
      req.params.id = '1';
      req.query.update = 'true';
      
      // Spy on the pythonService
      const runProximityAnalysisSpy = sinon.spy(pythonServiceMock, 'runProximityAnalysis');
      
      // Act
      await analysisController.getProximityAnalysis(req, res);
      
      // Assert
      expect(runProximityAnalysisSpy.calledOnce).to.be.true;
      expect(runProximityAnalysisSpy.firstCall.args[2]).to.be.true;
    });
  });
  
  describe('batchProximityAnalysis', function() {
    it('should start a batch analysis job for multiple erratics', async function() {
      // Arrange
      req.body = {
        erraticIds: [1, 2],
        featureLayers: ['water_bodies', 'settlements'],
        updateDb: true
      };
      
      // Act
      await analysisController.batchProximityAnalysis(req, res);
      
      // Assert
      expect(res.json.calledOnce).to.be.true;
      expect(res.json.firstCall.args[0]).to.have.property('message', 'Batch analysis started');
      expect(res.json.firstCall.args[0]).to.have.property('erratics_count', 2);
    });
    
    it('should handle invalid erratic IDs array', async function() {
      // Arrange
      req.body = {
        erraticIds: 'invalid',
        featureLayers: ['water_bodies', 'settlements']
      };
      
      // Act
      await analysisController.batchProximityAnalysis(req, res);
      
      // Assert
      expect(res.status.calledWith(400)).to.be.true;
      expect(res.status().json.calledWith(sinon.match({ error: 'Invalid or empty erratic ID list' }))).to.be.true;
    });
    
    it('should handle empty erratic IDs array', async function() {
      // Arrange
      req.body = {
        erraticIds: [],
        featureLayers: ['water_bodies', 'settlements']
      };
      
      // Act
      await analysisController.batchProximityAnalysis(req, res);
      
      // Assert
      expect(res.status.calledWith(400)).to.be.true;
      expect(res.status().json.calledWith(sinon.match({ error: 'Invalid or empty erratic ID list' }))).to.be.true;
    });
  });
}); 