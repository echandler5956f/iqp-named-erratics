const { expect } = require('chai');
const sinon = require('sinon');
const proxyquire = require('proxyquire').noCallThru();

// Mock dependencies
const childProcessMock = {
  exec: sinon.stub(),
};
const fsMock = {
  existsSync: sinon.stub(),
};
const loggerMock = {
  info: () => {},
  warn: () => {},
  error: () => {},
  debug: () => {},
};

// Use proxyquire to inject mocks
const pythonService = proxyquire('../../src/services/pythonService', {
  'child_process': childProcessMock,
  'fs': fsMock,
  '../utils/logger': loggerMock,
});

describe('Python Service - Unit Tests', function() {
  
  beforeEach(function() {
    sinon.reset();
    // Default mock behavior
    fsMock.existsSync.returns(true);
    childProcessMock.exec.yields(null, '{"success": true}', ''); // Simulate successful execution
  });

  describe('runScript', function() {
    it('should execute a Python script with correct parameters', async function() {
      const scriptName = 'test_script.py';
      const args = ['arg1', 'arg2'];
      
      await pythonService.runScript(scriptName, args);

      expect(childProcessMock.exec.calledOnce).to.be.true;
      const command = childProcessMock.exec.firstCall.args[0];
      expect(command).to.include(scriptName);
      expect(command).to.include('arg1 arg2');
    });

    it('should throw an error if script does not exist', async function() {
      fsMock.existsSync.returns(false);
      try {
        await pythonService.runScript('non_existent.py', []);
        // Fail the test if no error is thrown
        expect.fail('Expected runScript to throw an error but it did not.');
      } catch (error) {
        expect(error.message).to.include('Python script not found');
      }
    });

    it('should handle script execution errors', async function() {
      const execError = new Error('command failed');
      childProcessMock.exec.yields(execError, '', 'stderr output');

      try {
        await pythonService.runScript('fail_script.py', []);
        expect.fail('Expected runScript to throw an error but it did not.');
      } catch (error) {
        expect(error.message).to.include('Execution failed for script');
      }
    });

    it('should handle invalid JSON output from script', async function() {
      childProcessMock.exec.yields(null, 'this is not json', '');

      try {
        await pythonService.runScript('bad_json_script.py', []);
        expect.fail('Expected runScript to throw an error but it did not.');
      } catch (error) {
        expect(error.message).to.include('Invalid JSON output');
      }
    });
  });

  describe('runProximityAnalysis', function() {
    it('should call runScript with correct arguments', async function() {
      const runScriptStub = sinon.stub(pythonService, 'runScript').resolves();
      await pythonService.runProximityAnalysis(1, ['water', 'roads'], true);
      
      expect(runScriptStub.calledOnce).to.be.true;
      const [scriptName, args] = runScriptStub.firstCall.args;
      expect(scriptName).to.equal('proximity_analysis.py');
      expect(args).to.deep.equal(['1', '--features', 'water', 'roads', '--update-db']);
      
      runScriptStub.restore();
    });
  });

  describe('runClassification', function() {
    it('should call runScript with correct arguments', async function() {
        const runScriptStub = sinon.stub(pythonService, 'runScript').resolves();
        await pythonService.runClassification(5, true);
        
        expect(runScriptStub.calledOnce).to.be.true;
        const [scriptName, args] = runScriptStub.firstCall.args;
        expect(scriptName).to.equal('classify_erratic.py');
        expect(args).to.deep.equal(['5', '--update-db']);
        
        runScriptStub.restore();
    });
  });

  describe('runClusteringAnalysis', function() {
    it('should call runScript with correct arguments including JSON params', async function() {
        const runScriptStub = sinon.stub(pythonService, 'runScript').resolves();
        const algoParams = { eps: 0.5, min_samples: 5 };
        await pythonService.runClusteringAnalysis('dbscan', ['lat', 'lon'], algoParams, true, 'results.json');
        
        expect(runScriptStub.calledOnce).to.be.true;
        const [scriptName, args] = runScriptStub.firstCall.args;
        expect(scriptName).to.equal('clustering.py');
        expect(args).to.deep.equal([
            '--algorithm', 'dbscan',
            '--features', 'lat', 'lon',
            '--algo_params', '{"eps":0.5,"min_samples":5}',
            '--output', 'results.json'
        ]);
        
        runScriptStub.restore();
    });
  });

  describe('runBuildTopicModels', function() {
    it('should call runScript with correct arguments and placeholder ID', async function() {
        const runScriptStub = sinon.stub(pythonService, 'runScript').resolves();
        await pythonService.runBuildTopicModels('topics.json');
        
        expect(runScriptStub.calledOnce).to.be.true;
        const [scriptName, args] = runScriptStub.firstCall.args;
        expect(scriptName).to.equal('classify_erratic.py');
        expect(args).to.deep.equal(['1', '--build-topics', '--output', 'topics.json']);
        
        runScriptStub.restore();
    });
  });
}); 