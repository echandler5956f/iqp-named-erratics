const request = require('supertest');
const { expect } = require('chai');
const sinon = require('sinon');
const proxyquire = require('proxyquire').noCallThru();
const express = require('express');

// Mock services and utilities
const pythonServiceMock = {
  runProximityAnalysis: sinon.stub(),
  runClassification: sinon.stub(),
  // Stub other pythonService methods if their routes are tested
};

// We will use the actual job store logic but can spy on it
const { jobStore, generateJobId } = require('../../src/utils/jobStore');
const jobStoreSpy = {
  addJob: sinon.spy(jobStore, 'addJob'),
  updateJobStatus: sinon.spy(jobStore, 'updateJobStatus'),
  getJob: sinon.spy(jobStore, 'getJob'),
};

describe('Analysis Routes - Integration Tests', function() {
  let app;
  this.timeout(5000);

  before(function() {
    // Reset stubs and spies before tests
    sinon.reset();

    // Mock the python service in the controller
    const analysisController = proxyquire('../../src/controllers/analysisController', {
      '../services/pythonService': pythonServiceMock,
      '../utils/jobStore': { jobStore, generateJobId }, // Use real job store
      '../utils/logger': { info: () => {}, warn: () => {}, error: () => {}, debug: () => {} }
    });

    // Mock auth middleware
    const authMiddleware = {
      authenticateToken: (req, res, next) => next(),
      requireAdmin: (req, res, next) => next()
    };
    
    // Create router with the controller that has mocked dependencies
    const analysisRoutes = proxyquire('../../src/routes/analysisRoutes', {
      '../controllers/analysisController': analysisController,
      '../utils/auth': authMiddleware
    });
    
    // Setup express app
    app = express();
    app.use(express.json());
    app.use('/api/analysis', analysisRoutes);
  });

  afterEach(function() {
    // Clear spies history after each test
    sinon.resetHistory();
  });

  after(function() {
    // Restore original methods
    sinon.restore();
  });
  
  describe('GET /api/analysis/proximity/:id', function() {
    it('should return 200 and analysis for a valid ID', async () => {
      const mockResult = { erratic_id: 1, analysis: 'data' };
      pythonServiceMock.runProximityAnalysis.resolves(mockResult);

      const res = await request(app)
        .get('/api/analysis/proximity/1')
        .expect('Content-Type', /json/)
        .expect(200);
        
      expect(res.body).to.deep.equal(mockResult);
      expect(pythonServiceMock.runProximityAnalysis.calledOnceWith(1)).to.be.true;
    });
    
    it('should return 404 for a non-existent ID', async () => {
      pythonServiceMock.runProximityAnalysis.resolves({ error: 'Not found', statusCode: 404 });

      const res = await request(app)
        .get('/api/analysis/proximity/999')
        .expect('Content-Type', /json/)
        .expect(404);
        
      expect(res.body).to.have.property('error', 'Not found');
    });

    it('should return 400 for an invalid ID', async () => {
      const res = await request(app)
        .get('/api/analysis/proximity/invalid')
        .expect('Content-Type', /json/)
        .expect(400);
      
      expect(res.body).to.have.property('error', 'Invalid erratic ID');
    });
  });
  
  describe('POST /api/analysis/proximity/batch', function() {
    it('should return 202 Accepted and create a job', async () => {
      // Prevent the background process from running and throwing errors
      pythonServiceMock.runProximityAnalysis.resolves({ success: true });

      const res = await request(app)
        .post('/api/analysis/proximity/batch')
        .send({ erraticIds: [1, 2] })
        .expect('Content-Type', /json/)
        .expect(202);
      
      expect(res.body).to.have.property('message', 'Batch proximity analysis accepted');
      expect(res.body).to.have.property('job_id');
      
      const jobId = res.body.job_id;
      expect(jobStoreSpy.addJob.calledOnce).to.be.true;
      
      const job = jobStore.getJob(jobId);
      expect(job).to.not.be.null;
      expect(job.status).to.equal('pending');
      expect(job.type).to.equal('batch_proximity');
    });

    it('should return 400 for invalid request body', async () => {
      const res = await request(app)
        .post('/api/analysis/proximity/batch')
        .send({ erraticIds: [] }) // Empty array
        .expect('Content-Type', /json/)
        .expect(400);

      expect(res.body).to.have.property('error', 'Invalid or empty erratic ID list');
    });
  });

  describe('GET /api/analysis/jobs/:jobId', function() {
    it('should return the status of a created job', async () => {
      pythonServiceMock.runProximityAnalysis.resolves({ success: true });

      // First, create a job
      const batchRes = await request(app)
        .post('/api/analysis/proximity/batch')
        .send({ erraticIds: [5] })
        .expect(202);
      
      const jobId = batchRes.body.job_id;

      // Now, poll the job status endpoint
      const statusRes = await request(app)
        .get(`/api/analysis/jobs/${jobId}`)
        .expect('Content-Type', /json/)
        .expect(200);

      expect(statusRes.body.id).to.equal(jobId);
      expect(statusRes.body.status).to.be.oneOf(['pending', 'running']); // It will be 'pending' as the background task is stubbed
    });

    it('should return 404 for a non-existent job ID', async () => {
      const res = await request(app)
        .get('/api/analysis/jobs/nonexistent_job_123')
        .expect('Content-Type', /json/)
        .expect(404);
        
      expect(res.body).to.have.property('message', 'Job not found');
    });
  });
}); 