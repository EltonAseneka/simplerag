/**
 * @fileoverview Custom typed error hierarchy for SimpleRAG.
 * 
 * Provides specific error classifications with machine-readable error codes
 * and structured metadata to facilitate debugging and robust upstream error handling.
 * 
 * @module lib/errors
 */

/**
 * Base error class for all SimpleRAG operational errors.
 */
class BaseRAGError extends Error {
    /**
     * @param {string} message - Human-readable error message.
     * @param {string} code - Machine-readable error code.
     * @param {Object} [details={}] - Additional context or metadata.
     */
    constructor(message, code = 'RAG_GENERAL_ERROR', details = {}) {
        super(message);
        this.name = this.constructor.name;
        this.code = code;
        this.details = details;
        if (Error.captureStackTrace) {
            Error.captureStackTrace(this, this.constructor);
        }
    }
}

/**
 * Thrown when an input argument fails validation constraints.
 */
class ValidationError extends BaseRAGError {
    constructor(message, details = {}) {
        super(message, 'VALIDATION_ERROR', details);
    }
}

/**
 * Thrown when a requested document file cannot be found on the filesystem.
 */
class DocumentNotFoundError extends BaseRAGError {
    /**
     * @param {string} docPath - Path to the missing document.
     */
    constructor(docPath) {
        super(`Document file not found at path: "${docPath}"`, 'DOCUMENT_NOT_FOUND', { docPath });
    }
}

/**
 * Thrown when a document has an unsupported file format or extension.
 */
class UnsupportedFormatError extends BaseRAGError {
    /**
     * @param {string} extension - The unsupported extension encountered.
     * @param {string[]} supported - List of supported extensions.
     */
    constructor(extension, supported = ['pdf', 'docx', 'txt', 'md']) {
        const extWithDot = extension.startsWith('.') ? extension : `.${extension}`;
        super(
            `Unsupported document format: "${extWithDot}". Supported formats are: ${supported.map(s => `.${s}`).join(', ')}`,
            'UNSUPPORTED_FORMAT',
            { extension, supported }
        );
    }
}

/**
 * Thrown when a document file yields no text content upon parsing.
 */
class EmptyDocumentError extends BaseRAGError {
    /**
     * @param {string} docPath - Path to the empty document.
     */
    constructor(docPath) {
        super(`No content could be extracted from document: "${docPath}"`, 'EMPTY_DOCUMENT', { docPath });
    }
}

/**
 * Thrown when embedding initialization or execution fails.
 */
class EmbeddingProviderError extends BaseRAGError {
    /**
     * @param {string} message - Specific embedding error description.
     * @param {string} provider - Provider name ('openai', 'google', etc.).
     */
    constructor(message, provider) {
        super(message, 'EMBEDDING_PROVIDER_ERROR', { provider });
    }
}

/**
 * Thrown when LLM initialization or invocation fails.
 */
class LLMProviderError extends BaseRAGError {
    /**
     * @param {string} message - Specific LLM error description.
     * @param {string} model - Model identifier.
     */
    constructor(message, model) {
        super(message, 'LLM_PROVIDER_ERROR', { model });
    }
}

/**
 * Thrown when vector store creation, loading, or persistence fails.
 */
class VectorStoreError extends BaseRAGError {
    /**
     * @param {string} message - Specific vector store error message.
     * @param {string} [indexPath] - Path to vector store index.
     */
    constructor(message, indexPath = '') {
        super(message, 'VECTOR_STORE_ERROR', { indexPath });
    }
}

module.exports = {
    BaseRAGError,
    ValidationError,
    DocumentNotFoundError,
    UnsupportedFormatError,
    EmptyDocumentError,
    EmbeddingProviderError,
    LLMProviderError,
    VectorStoreError
};
