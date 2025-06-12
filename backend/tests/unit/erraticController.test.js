const { expect } = require('chai');
const sinon = require('sinon');
const proxyquire = require('proxyquire').noCallThru();

// Mock dependencies
const erraticServiceMock = {
  getAllErratics: sinon.stub(),
  getErraticById: sinon.stub(),
  getNearbyErratics: sinon.stub(),
  createErratic: sinon.stub(),
  updateErratic: sinon.stub(),
  deleteErratic: sinon.stub(),
};

const loggerMock = {
  info: () => {},
  warn: () => {},
  error: () => {},
  debug: () => {},
};

// Use proxyquire to inject mocks
const erraticController = proxyquire('../../src/controllers/erraticController', {
  '../services/erraticService': erraticServiceMock,
  '../utils/logger': loggerMock,
});

describe('Erratic Controller - Unit Tests', function() {
  let req, res;
  
  beforeEach(function() {
    sinon.reset();
    req = { params: {}, query: {}, body: {} };
    res = {
      status: sinon.stub().returnsThis(),
      json: sinon.spy(),
    };
  });

  describe('getAllErratics', function() {
    it('should fetch all erratics and return them', async function() {
      const mockErratics = [{ id: 1, name: 'Test Erratic' }];
      erraticServiceMock.getAllErratics.resolves(mockErratics);

      await erraticController.getAllErratics(req, res);

      expect(erraticServiceMock.getAllErratics.calledOnceWith(req.query)).to.be.true;
      expect(res.json.calledOnceWith(mockErratics)).to.be.true;
    });

    it('should handle service errors', async function() {
        const error = new Error('Service failed');
        error.statusCode = 500;
        erraticServiceMock.getAllErratics.rejects(error);

        await erraticController.getAllErratics(req, res);

        expect(res.status.calledOnceWith(500)).to.be.true;
        expect(res.json.calledOnceWith({ message: 'Service failed' })).to.be.true;
    });
  });

  describe('getErraticById', function() {
    it('should return a single erratic for a valid ID', async function() {
      req.params.id = '1';
      const mockErratic = { id: 1, name: 'Found Erratic' };
      erraticServiceMock.getErraticById.resolves(mockErratic);

      await erraticController.getErraticById(req, res);

      expect(erraticServiceMock.getErraticById.calledOnceWith('1')).to.be.true;
      expect(res.json.calledOnceWith(mockErratic)).to.be.true;
    });

    it('should return 404 if erratic is not found', async function() {
        req.params.id = '999';
        const error = new Error('Not found');
        error.statusCode = 404;
        erraticServiceMock.getErraticById.rejects(error);

        await erraticController.getErraticById(req, res);

        expect(res.status.calledOnceWith(404)).to.be.true;
        expect(res.json.calledOnceWith({ message: 'Not found' })).to.be.true;
    });
  });

  describe('createErratic', function() {
    it('should create an erratic and return 201 status', async function() {
        req.body = { name: 'New Erratic', location: 'POINT(1 1)' };
        const createdErratic = { id: 3, ...req.body };
        erraticServiceMock.createErratic.resolves(createdErratic);

        await erraticController.createErratic(req, res);

        expect(erraticServiceMock.createErratic.calledOnceWith(req.body)).to.be.true;
        expect(res.status.calledOnceWith(201)).to.be.true;
        expect(res.json.calledOnceWith(createdErratic)).to.be.true;
    });
  });

  describe('updateErratic', function() {
    it('should update an erratic and return it', async function() {
        req.params.id = '1';
        req.body = { name: 'Updated Name' };
        const updatedErratic = { id: 1, name: 'Updated Name' };
        erraticServiceMock.updateErratic.resolves(updatedErratic);

        await erraticController.updateErratic(req, res);

        expect(erraticServiceMock.updateErratic.calledOnceWith('1', req.body)).to.be.true;
        expect(res.json.calledOnceWith(updatedErratic)).to.be.true;
    });
  });

  describe('deleteErratic', function() {
    it('should delete an erratic and return a success message', async function() {
        req.params.id = '1';
        const result = { message: 'Erratic deleted successfully' };
        erraticServiceMock.deleteErratic.resolves(result);

        await erraticController.deleteErratic(req, res);

        expect(erraticServiceMock.deleteErratic.calledOnceWith('1')).to.be.true;
        expect(res.json.calledOnceWith(result)).to.be.true;
    });
  });

  describe('getNearbyErratics', function() {
    it('should fetch nearby erratics and return them', async function() {
        req.query = { lat: '40', lng: '-70', radius: '1000' };
        const mockErratics = [{ id: 2, name: 'Nearby Erratic' }];
        erraticServiceMock.getNearbyErratics.resolves(mockErratics);

        await erraticController.getNearbyErratics(req, res);

        expect(erraticServiceMock.getNearbyErratics.calledOnceWith('40', '-70', '1000')).to.be.true;
        expect(res.json.calledOnceWith(mockErratics)).to.be.true;
    });
  });
}); 