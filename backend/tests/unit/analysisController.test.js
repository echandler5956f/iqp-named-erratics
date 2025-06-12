const { expect } = require('chai');
const sinon = require('sinon');
const proxyquire = require('proxyquire').noCallThru();

// Mock dependencies
const pythonServiceMock = {
  runProximityAnalysis: sinon.stub(),
  runClassification: sinon.stub(),
  runClusteringAnalysis: sinon.stub(),
  runBuildTopicModels: sinon.stub(),
};

const jobStoreMock = {
  addJob: sinon.stub(),
  updateJobStatus: sinon.stub(),
  getJob: sinon.stub(),
};

const generateJobIdMock = sinon.stub();

// Use proxyquire to inject our mocks
const analysisController = proxyquire('../../src/controllers/analysisController', {
  '../services/pythonService': pythonServiceMock,
  '../utils/jobStore': { 
    jobStore: jobStoreMock,
    generateJobId: generateJobIdMock
  },
  '../utils/logger': {
    info: () => {},
    warn: () => {},
    error: () => {},
    debug: () => {},
  }
});

describe('Analysis Controller - Unit Tests', function() {
  let req, res;
  
  beforeEach(function() {
    // Reset stubs before each test
    sinon.reset();
    
    // Setup fresh request and response objects
    req = {
      params: {},
      query: {},
      body: {}
    };
    
    res = {
      status: sinon.stub().returnsThis(),
      json: sinon.spy()
    };
  });
  
  describe('getProximityAnalysis', function() {
    it('should return proximity analysis for a valid ID', async function() {
      req.params.id = '1';
      const mockResult = { success: true, data: 'some_analysis' };
      pythonServiceMock.runProximityAnalysis.resolves(mockResult);
      
      await analysisController.getProximityAnalysis(req, res);
      
      expect(pythonServiceMock.runProximityAnalysis.calledOnceWith(1, [], false)).to.be.true;
      expect(res.json.calledOnceWith(mockResult)).to.be.true;
    });

    it('should handle Python service error', async function() {
      req.params.id = '999';
      const mockError = { error: 'Erratic not found', statusCode: 404 };
      pythonServiceMock.runProximityAnalysis.resolves(mockError);

      await analysisController.getProximityAnalysis(req, res);

      expect(res.status.calledOnceWith(404)).to.be.true;
      expect(res.json.calledOnceWith({ error: 'Erratic not found' })).to.be.true;
    });

    it('should return 400 for an invalid ID', async function() {
      req.params.id = 'invalid';
      await analysisController.getProximityAnalysis(req, res);
      expect(res.status.calledOnceWith(400)).to.be.true;
      expect(res.json.calledOnceWith({ error: 'Invalid erratic ID' })).to.be.true;
    });
  });

  describe('batchProximityAnalysis', function() {
    it('should accept a batch job and return 202 with a job ID', async function() {
      req.body = { erraticIds: [1, 2, 3] };
      const mockJobId = 'batch_proximity_123';
      generateJobIdMock.returns(mockJobId);

      // We need to stub the private method call to prevent it from actually running
      const processBatchStub = sinon.stub(analysisController, '_processBatchAnalysis').resolves();
      
      await analysisController.batchProximityAnalysis(req, res);
      
      expect(generateJobIdMock.calledOnceWith('batch_proximity')).to.be.true;
      expect(jobStoreMock.addJob.calledOnceWith(mockJobId, 'batch_proximity', { count: 3, featureLayers: undefined, updateDb: undefined })).to.be.true;
      expect(res.status.calledOnceWith(202)).to.be.true;
      expect(res.json.calledOnceWith({ message: 'Batch proximity analysis accepted', job_id: mockJobId })).to.be.true;
      expect(processBatchStub.calledOnce).to.be.true;

      processBatchStub.restore(); // Clean up the stub on the controller itself
    });

    it('should return 400 if erraticIds is not a valid array', async function() {
      req.body = { erraticIds: 'not-an-array' };
      await analysisController.batchProximityAnalysis(req, res);
      expect(res.status.calledOnceWith(400)).to.be.true;
      expect(res.json.calledOnceWith({ error: 'Invalid or empty erratic ID list' })).to.be.true;
    });
  });

  describe('_processBatchAnalysis', function() {
    it('should process all items and update job status correctly on success', async function() {
      const mockJobId = 'batch_123';
      const erraticIds = [1, 2];
      pythonServiceMock.runProximityAnalysis.resolves({ success: true });

      const summary = await analysisController._processBatchAnalysis(mockJobId, erraticIds, [], true);

      expect(jobStoreMock.updateJobStatus.calledWith(mockJobId, 'running')).to.be.true;
      expect(pythonServiceMock.runProximityAnalysis.callCount).to.equal(2);
      expect(summary.successful).to.equal(2);
      expect(summary.failed).to.equal(0);
      expect(jobStoreMock.updateJobStatus.calledWith(mockJobId, 'completed')).to.be.true;
    });

    it('should handle failures and update job status with errors', async function() {
        const mockJobId = 'batch_456';
        const erraticIds = [1, 999];
        pythonServiceMock.runProximityAnalysis.withArgs(1).resolves({ success: true });
        pythonServiceMock.runProximityAnalysis.withArgs(999).resolves({ error: 'Not Found' });
  
        const summary = await analysisController._processBatchAnalysis(mockJobId, erraticIds, [], true);
  
        expect(jobStoreMock.updateJobStatus.calledWith(mockJobId, 'running')).to.be.true;
        expect(summary.successful).to.equal(1);
        expect(summary.failed).to.equal(1);
        expect(summary.errors[0]).to.deep.equal({ id: 999, error: 'Not Found' });
        expect(jobStoreMock.updateJobStatus.calledWith(mockJobId, 'completed_with_errors')).to.be.true;
      });
  });

  describe('getClusterAnalysis', function() {
    it('should accept a cluster job and return 202', async function() {
        req.query = { algorithm: 'dbscan' };
        const mockJobId = 'cluster_abc';
        generateJobIdMock.returns(mockJobId);
        pythonServiceMock.runClusteringAnalysis.resolves({ success: true }); // Prevent unhandled promise rejection

        await analysisController.getClusterAnalysis(req, res);

        expect(generateJobIdMock.calledOnceWith('clustering')).to.be.true;
        expect(jobStoreMock.addJob.calledOnce).to.be.true;
        expect(res.status.calledOnceWith(202)).to.be.true;
        expect(res.json.calledOnceWith({ message: 'Clustering analysis job accepted', job_id: mockJobId })).to.be.true;
    });
  });

  describe('getJobStatus', function() {
    it('should return job status if job exists', async function() {
      req.params.jobId = 'job_123';
      const mockJob = { id: 'job_123', status: 'running' };
      jobStoreMock.getJob.returns(mockJob);

      await analysisController.getJobStatus(req, res);

      expect(jobStoreMock.getJob.calledOnceWith('job_123')).to.be.true;
      expect(res.json.calledOnceWith(mockJob)).to.be.true;
    });

    it('should return 404 if job does not exist', async function() {
      req.params.jobId = 'job_456';
      jobStoreMock.getJob.returns(null);

      await analysisController.getJobStatus(req, res);

      expect(jobStoreMock.getJob.calledOnceWith('job_456')).to.be.true;
      expect(res.status.calledOnceWith(404)).to.be.true;
      expect(res.json.calledOnceWith({ message: 'Job not found' })).to.be.true;
    });
  });
}); 