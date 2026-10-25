/**
 * @fileoverview Lightweight structured logger for SimpleRAG.
 * 
 * Provides consistent log levels, structured JSON outputs, and environment-based
 * filtering without adding heavy runtime dependencies.
 * 
 * @module lib/logger
 */

const LOG_LEVELS = Object.freeze({
    silent: 0,
    error: 1,
    warn: 2,
    info: 3,
    debug: 4
});

/**
 * Resolves current active log level from environment or default.
 * @returns {number}
 */
function getCurrentLogLevel() {
    const envLevel = (process.env.LOG_LEVEL || '').toLowerCase();
    if (Object.prototype.hasOwnProperty.call(LOG_LEVELS, envLevel)) {
        return LOG_LEVELS[envLevel];
    }
    return process.env.NODE_ENV === 'test' ? LOG_LEVELS.silent : LOG_LEVELS.info;
}

/**
 * Emits a structured log record.
 * 
 * @param {'error'|'warn'|'info'|'debug'} level - Severity level.
 * @param {string} message - Human-readable log message.
 * @param {Object} [meta={}] - Structured context metadata.
 */
function log(level, message, meta = {}) {
    const currentThreshold = getCurrentLogLevel();
    const targetPriority = LOG_LEVELS[level] || LOG_LEVELS.info;

    if (targetPriority > currentThreshold) {
        return;
    }

    const timestamp = new Date().toISOString();
    const record = {
        timestamp,
        level: level.toUpperCase(),
        message,
        ...meta
    };

    if (process.env.LOG_FORMAT === 'json') {
        const stream = level === 'error' ? process.stderr : process.stdout;
        stream.write(JSON.stringify(record) + '\n');
    } else {
        const stream = level === 'error' ? process.stderr : process.stdout;
        const metaStr = Object.keys(meta).length > 0 ? ` ${JSON.stringify(meta)}` : '';
        stream.write(`[${timestamp}] [${level.toUpperCase()}] ${message}${metaStr}\n`);
    }
}

const logger = {
    debug: (msg, meta) => log('debug', msg, meta),
    info: (msg, meta) => log('info', msg, meta),
    warn: (msg, meta) => log('warn', msg, meta),
    error: (msg, meta) => log('error', msg, meta)
};

module.exports = {
    logger,
    LOG_LEVELS
};
