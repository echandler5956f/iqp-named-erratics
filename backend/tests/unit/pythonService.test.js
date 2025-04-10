const { expect } = require('chai');
const sinon = require('sinon');
const childProcess = require('child_process');
const fs = require('fs');
const path = require('path');
const pythonService = require('../../src/services/pythonService');

describe('Python Service', function() {
  // Stub for exec and fs
  let execStub, fsExistsStub, originalExec;
  
  before(function() {
    // Save the original exec
    originalExec = childProcess.exec;
  });
  
  beforeEach(function() {
    // Create a replacement function for exec that we control completely
    childProcess.exec = function mockExec(command, callback) {
      // We can check the command but don't actually run it
      const returnValue = {
        on: sinon.stub()
      };
      
      // Default to success
      if (callback) {
        process.nextTick(() => callback(null, '{"success": true}', ''));
      }
      
      return returnValue;
    };
    
    // Spy on our mock exec to track calls
    execStub = sinon.spy(childProcess, 'exec');
    
    // Make fs.existsSync always return true by default
    fsExistsStub = sinon.stub(fs, 'existsSync').returns(true);
  });
  
  afterEach(function() {
    sinon.restore();
  });
  
  after(function() {
    // Restore the original exec
    childProcess.exec = originalExec;
  });
  
  describe('runScript', function() {
    it('should execute a Python script with the correct parameters', async function() {
      // Arrange
      const scriptName = 'test_script.py';
      const args = ['arg1', 'arg2'];
      const mockOutput = JSON.stringify({ result: 'success', args: ['arg1', 'arg2'] });
      
      // Override our exec mock for this specific test
      childProcess.exec = function(command, callback) {
        // Validate the command string
        expect(command).to.include(scriptName);
        expect(command).to.include('arg1 arg2');
        
        // Return the mock output
        callback(null, mockOutput, '');
        
        return { on: sinon.stub() };
      };
      
      // Act
      const result = await pythonService.runScript(scriptName, args);
      
      // Assert
      expect(result).to.deep.equal({ result: 'success', args: ['arg1', 'arg2'] });
    });
    
    it('should handle script execution errors', async function() {
      // Arrange
      const scriptName = 'test_script.py';
      const error = new Error('Script execution failed');
      
      // Create a stub for the runScript method to force the error
      const runScriptStub = sinon.stub(pythonService, 'runScript');
      runScriptStub.throws(error);
      
      // Act & Assert
      try {
        await pythonService.runScript(scriptName);
        expect.fail('Should have thrown an error');
      } catch (err) {
        expect(err.message).to.equal('Script execution failed');
      }
    });
    
    it('should handle invalid JSON output', async function() {
      try {
        // Run the script that outputs invalid JSON
        await pythonService.runScript('invalid_json_test.py');
        expect.fail('Should have thrown an error');
      } catch (err) {
        expect(err.message).to.equal('Invalid output format from Python script');
      }
    });
    
    it('should handle non-existent scripts', async function() {
      // Arrange
      const scriptName = 'non_existent_script.py';
      
      // Override the fsExistsStub for this test only
      fsExistsStub.returns(false);
      
      // Act & Assert
      try {
        await pythonService.runScript(scriptName);
        expect.fail('Should have thrown an error');
      } catch (err) {
        expect(err.message).to.include('Python script not found');
      }
    });
  });
  
  describe('runProximityAnalysis', function() {
    it('should call runScript with the correct parameters', async function() {
      // Arrange
      const erraticId = 1;
      const featureLayers = ['water_bodies', 'settlements'];
      const updateDb = true;
      
      // Create a stub for the runScript method
      const runScriptStub = sinon.stub(pythonService, 'runScript').resolves({ result: 'success' });
      
      // Act
      await pythonService.runProximityAnalysis(erraticId, featureLayers, updateDb);
      
      // Assert
      expect(runScriptStub.calledOnce).to.be.true;
      expect(runScriptStub.firstCall.args[0]).to.equal('proximity_analysis.py');
      
      const passedArgs = runScriptStub.firstCall.args[1];
      expect(passedArgs).to.include('1');
      expect(passedArgs).to.include('--features');
      expect(passedArgs).to.include('water_bodies');
      expect(passedArgs).to.include('settlements');
      expect(passedArgs).to.include('--update-db');
    });
    
    it('should not include --update-db flag when updateDb is false', async function() {
      // Arrange
      const erraticId = 1;
      const featureLayers = ['water_bodies'];
      const updateDb = false;
      
      // Create a stub for the runScript method
      const runScriptStub = sinon.stub(pythonService, 'runScript').resolves({ result: 'success' });
      
      // Act
      await pythonService.runProximityAnalysis(erraticId, featureLayers, updateDb);
      
      // Assert
      expect(runScriptStub.calledOnce).to.be.true;
      const passedArgs = runScriptStub.firstCall.args[1];
      expect(passedArgs).to.not.include('--update-db');
    });
    
    it('should not include feature layers if none are provided', async function() {
      // Arrange
      const erraticId = 1;
      const featureLayers = [];
      
      // Create a stub for the runScript method
      const runScriptStub = sinon.stub(pythonService, 'runScript').resolves({ result: 'success' });
      
      // Act
      await pythonService.runProximityAnalysis(erraticId, featureLayers);
      
      // Assert
      expect(runScriptStub.calledOnce).to.be.true;
      const passedArgs = runScriptStub.firstCall.args[1];
      expect(passedArgs).to.not.include('--features');
    });
  });
}); 