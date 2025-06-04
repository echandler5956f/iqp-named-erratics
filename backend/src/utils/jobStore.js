const logger = require('./logger');

// Simple in-memory store for job statuses
const jobs = new Map();

// Simple job ID generator (can be made more robust if needed, e.g., using uuid)
const generateJobId = (prefix = 'job') => `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

const jobStore = {
  addJob: (jobId, type, initialData = {}) => {
    if (jobs.has(jobId)) {
      logger.warn(`[JobStore] Job ID ${jobId} already exists. Overwriting is generally not expected.`);
    }
    const job = {
      id: jobId,
      type, // e.g., 'batch_proximity', 'clustering'
      status: 'pending', // Possible statuses: pending, running, completed, failed
      createdAt: new Date(),
      updatedAt: new Date(),
      params: initialData, // Parameters used to start the job
      result: null,
      error: null,
    };
    jobs.set(jobId, job);
    logger.info(`[JobStore] Added job: ${jobId}`, { type, initialData });
    return job;
  },

  updateJobStatus: (jobId, status, data = {}) => {
    if (jobs.has(jobId)) {
      const job = jobs.get(jobId);
      job.status = status;
      job.updatedAt = new Date();
      if (data.result !== undefined) job.result = data.result;
      if (data.error !== undefined) job.error = data.error;
      jobs.set(jobId, job);
      logger.info(`[JobStore] Updated job ${jobId} to status: ${status}`, { resultExists: data.result !== undefined, errorExists: data.error !== undefined });
    } else {
      logger.warn(`[JobStore] Attempted to update non-existent job ID: ${jobId}`);
    }
  },

  getJob: (jobId) => {
    if (!jobs.has(jobId)) {
      logger.warn(`[JobStore] Attempted to get non-existent job ID: ${jobId}`);
      return null;
    }
    return jobs.get(jobId);
  },

  // Optional: A way to list jobs (for debugging, could be an admin endpoint)
  listJobs: (limit = 20) => {
    // Return a limited list of recent jobs, for example
    const allJobs = Array.from(jobs.values());
    allJobs.sort((a, b) => b.createdAt - a.createdAt); // Sort by newest first
    return allJobs.slice(0, limit);
  },

  // Optional: A simple cleanup for very old jobs to prevent memory leaks if server runs for a very long time
  // This is a very basic implementation. For production, a more robust TTL or LRU cache might be better.
  cleanupOldJobs: (maxAgeMinutes = 60 * 24) => {
    const now = new Date();
    const maxAgeMs = maxAgeMinutes * 60 * 1000;
    let cleanedCount = 0;
    for (const [jobId, job] of jobs.entries()) {
      if (job.status === 'completed' || job.status === 'failed') {
        if (now - job.updatedAt > maxAgeMs) {
          jobs.delete(jobId);
          cleanedCount++;
        }
      }
    }
    if (cleanedCount > 0) {
      logger.info(`[JobStore] Cleaned up ${cleanedCount} old jobs.`);
    }
  }
};

// Periodically cleanup old jobs (e.g., every hour)
setInterval(() => {
  jobStore.cleanupOldJobs();
}, 60 * 60 * 1000);

module.exports = { jobStore, generateJobId }; 