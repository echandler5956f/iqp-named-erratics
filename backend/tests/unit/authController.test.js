const { expect } = require('chai');
const sinon = require('sinon');
const proxyquire = require('proxyquire').noCallThru();

// Mock dependencies
const authServiceMock = {
  register: sinon.stub(),
  login: sinon.stub(),
  getProfile: sinon.stub(),
};

const loggerMock = {
  info: () => {},
  warn: () => {},
  error: () => {},
  debug: () => {},
};

// Use proxyquire to inject mocks
const authController = proxyquire('../../src/controllers/authController', {
  '../services/authService': authServiceMock,
  '../utils/logger': loggerMock,
});

describe('Auth Controller - Unit Tests', function() {
  let req, res;
  
  beforeEach(function() {
    sinon.reset();
    req = { params: {}, query: {}, body: {}, user: {} };
    res = {
      status: sinon.stub().returnsThis(),
      json: sinon.spy(),
    };
  });

  describe('register', function() {
    it('should register a user and return 201', async function() {
      req.body = { username: 'testuser', password: 'password123' };
      const result = { user: { id: 1, username: 'testuser' }, token: 'a.b.c' };
      authServiceMock.register.resolves(result);

      await authController.register(req, res);

      expect(authServiceMock.register.calledOnceWith(req.body)).to.be.true;
      expect(res.status.calledOnceWith(201)).to.be.true;
      expect(res.json.calledOnceWith(result)).to.be.true;
    });

    it('should handle registration errors', async function() {
        const error = new Error('User already exists');
        error.statusCode = 409;
        authServiceMock.register.rejects(error);

        await authController.register(req, res);

        expect(res.status.calledOnceWith(409)).to.be.true;
        expect(res.json.calledOnceWith({ message: 'User already exists' })).to.be.true;
    });
  });

  describe('login', function() {
    it('should log in a user and return a token', async function() {
      req.body = { username: 'testuser', password: 'password123' };
      const result = { user: { id: 1, username: 'testuser' }, token: 'a.b.c' };
      authServiceMock.login.resolves(result);

      await authController.login(req, res);

      expect(authServiceMock.login.calledOnceWith(req.body)).to.be.true;
      expect(res.json.calledOnceWith(result)).to.be.true;
    });

    it('should handle login failure', async function() {
        req.body = { username: 'testuser', password: 'wrongpassword' };
        const error = new Error('Invalid credentials');
        error.statusCode = 401;
        authServiceMock.login.rejects(error);

        await authController.login(req, res);

        expect(res.status.calledOnceWith(401)).to.be.true;
        expect(res.json.calledOnceWith({ message: 'Invalid credentials' })).to.be.true;
    });
  });

  describe('getProfile', function() {
    it('should return the user profile for an authenticated user', async function() {
        req.user = { id: 1 }; // This would be attached by auth middleware
        const profile = { id: 1, username: 'testuser', email: 'test@test.com' };
        authServiceMock.getProfile.resolves(profile);

        await authController.getProfile(req, res);

        expect(authServiceMock.getProfile.calledOnceWith(1)).to.be.true;
        expect(res.json.calledOnceWith(profile)).to.be.true;
    });

    it('should handle profile not found error', async function() {
        req.user = { id: 999 };
        const error = new Error('User not found');
        error.statusCode = 404;
        authServiceMock.getProfile.rejects(error);

        await authController.getProfile(req, res);
        
        expect(res.status.calledOnceWith(404)).to.be.true;
        expect(res.json.calledOnceWith({ message: 'User not found' })).to.be.true;
    });
  });
}); 