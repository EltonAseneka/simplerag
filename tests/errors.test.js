/**
 * @fileoverview Unit tests for lib/errors.js and lib/logger.js.
 */

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');

const {
    BaseRAGError,
    ValidationError,
    DocumentNotFoundError,
    UnsupportedFormatError,
    EmptyDocumentError,
    EmbeddingProviderError,
    LLMProviderError,
    VectorStoreError
} = require('../lib/errors');

const { logger } = require('../lib/logger');

describe('lib/errors.js - Typed Error Classes', () => {
    it('BaseRAGError should inherit from Error with code and details', () => {
        const err = new BaseRAGError('General issue', 'ERR_CODE', { info: 1 });
        assert.ok(err instanceof Error);
        assert.ok(err instanceof BaseRAGError);
        assert.equal(err.name, 'BaseRAGError');
        assert.equal(err.code, 'ERR_CODE');
        assert.deepEqual(err.details, { info: 1 });
    });

    it('ValidationError should set default code VALIDATION_ERROR', () => {
        const err = new ValidationError('Bad input', { field: 'name' });
        assert.ok(err instanceof BaseRAGError);
        assert.equal(err.code, 'VALIDATION_ERROR');
    });

    it('DocumentNotFoundError should include docPath in details', () => {
        const err = new DocumentNotFoundError('/docs/missing.pdf');
        assert.ok(err instanceof BaseRAGError);
        assert.equal(err.code, 'DOCUMENT_NOT_FOUND');
        assert.equal(err.details.docPath, '/docs/missing.pdf');
    });

    it('UnsupportedFormatError should format message with dot and list supported', () => {
        const err = new UnsupportedFormatError('exe', ['pdf', 'txt']);
        assert.ok(err instanceof BaseRAGError);
        assert.equal(err.code, 'UNSUPPORTED_FORMAT');
        assert.ok(err.message.includes('.exe'));
        assert.ok(err.message.includes('.pdf, .txt'));
    });

    it('EmptyDocumentError should store docPath', () => {
        const err = new EmptyDocumentError('empty.txt');
        assert.ok(err instanceof BaseRAGError);
        assert.equal(err.code, 'EMPTY_DOCUMENT');
        assert.equal(err.details.docPath, 'empty.txt');
    });

    it('EmbeddingProviderError should store provider in details', () => {
        const err = new EmbeddingProviderError('Missing token', 'openai');
        assert.ok(err instanceof BaseRAGError);
        assert.equal(err.code, 'EMBEDDING_PROVIDER_ERROR');
        assert.equal(err.details.provider, 'openai');
    });

    it('LLMProviderError should store model in details', () => {
        const err = new LLMProviderError('Quota exceeded', 'gemini-1.5-flash');
        assert.ok(err instanceof BaseRAGError);
        assert.equal(err.code, 'LLM_PROVIDER_ERROR');
        assert.equal(err.details.model, 'gemini-1.5-flash');
    });

    it('VectorStoreError should store indexPath in details', () => {
        const err = new VectorStoreError('Failed to save', '/path/to/db');
        assert.ok(err instanceof BaseRAGError);
        assert.equal(err.code, 'VECTOR_STORE_ERROR');
        assert.equal(err.details.indexPath, '/path/to/db');
    });
});

describe('lib/logger.js - Structured Logger', () => {
    it('logger should expose debug, info, warn, error methods without throwing', () => {
        assert.doesNotThrow(() => logger.debug('debug message', { key: 'val' }));
        assert.doesNotThrow(() => logger.info('info message', { key: 'val' }));
        assert.doesNotThrow(() => logger.warn('warn message', { key: 'val' }));
        assert.doesNotThrow(() => logger.error('error message', { key: 'val' }));
    });
});
