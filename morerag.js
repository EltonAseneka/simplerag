/**
 * @fileoverview morerag.js - Document Path Utilities and Multi-Document RAG Operations.
 * 
 * Provides robust file system utilities, document format detection, directory ingestion,
 * structured logging, and multi-document retrieval-augmented generation (RAG) capabilities.
 * 
 * @module morerag
 */

const path = require('path');
const fs = require('fs');
const { ValidationError } = require('./lib/errors');
const { logger } = require('./lib/logger');

/**
 * Supported document file extensions for ingestion and vector indexing.
 * @type {ReadonlyArray<string>}
 */
const SUPPORTED_EXTENSIONS = Object.freeze(['pdf', 'docx', 'txt', 'md']);

/**
 * Extracts the file extension in normalized, lowercase form without the leading dot.
 * Works consistently across both Windows and POSIX paths.
 * 
 * @param {string} filePath - Path or filename to inspect.
 * @returns {string} Lowercase extension (e.g., 'pdf', 'docx') or empty string if none.
 * 
 * @example
 * getFileExtension('documents/report.PDF'); // returns 'pdf'
 * getFileExtension('archive.tar.gz');      // returns 'gz'
 * getFileExtension('README');              // returns ''
 */
function getFileExtension(filePath) {
    if (!filePath || typeof filePath !== 'string') {
        return '';
    }
    const clean = filePath.trim();
    if (!clean) return '';
    const filename = clean.split(/[\\/]/).filter(Boolean).pop() || '';
    const ext = path.extname(filename);
    return ext ? ext.replace(/^\./, '').toLowerCase() : '';
}

/**
 * Validates whether a given file path has a supported extension.
 * 
 * @param {string} filePath - Path to the file.
 * @param {string[]} [allowed=SUPPORTED_EXTENSIONS] - Array of allowed extensions.
 * @returns {boolean} True if file extension is supported, false otherwise.
 * 
 * @example
 * isSupportedFileType('doc.pdf'); // returns true
 * isSupportedFileType('notes.exe'); // returns false
 */
function isSupportedFileType(filePath, allowed = SUPPORTED_EXTENSIONS) {
    const ext = getFileExtension(filePath);
    if (!ext) return false;
    const normalizedAllowed = allowed.map(e => e.toLowerCase().replace(/^\./, ''));
    return normalizedAllowed.includes(ext);
}

/**
 * Extracts a safe, cross-platform base filename without directory path or extension.
 * Correctly handles both POSIX (/) and Windows (\) separators regardless of the host OS.
 * 
 * @param {string} filePath - Full or relative file path.
 * @returns {string} Clean base filename without extension.
 * 
 * @example
 * getSafeBasename('files/how_to_create_pdf_Gresham_Norman_CV_Clean.docx');
 * // returns 'how_to_create_pdf_Gresham_Norman_CV_Clean'
 * getSafeBasename('C:\\Users\\docs\\report.final.pdf');
 * // returns 'report.final'
 */
function getSafeBasename(filePath) {
    if (!filePath || typeof filePath !== 'string') {
        return '';
    }
    const cleanPath = filePath.trim();
    if (!cleanPath) return '';
    const filename = cleanPath.split(/[\\/]/).filter(Boolean).pop() || '';
    const ext = path.extname(filename);
    return filename.slice(0, filename.length - ext.length);
}

/**
 * Ensures that a target directory exists, creating all parent directories if necessary.
 * 
 * @param {string} dirPath - Directory path to verify or create.
 * @returns {string} The resolved absolute path of the created/verified directory.
 * @throws {ValidationError} If directory path is invalid.
 */
function ensureDirectoryExists(dirPath) {
    if (!dirPath || typeof dirPath !== 'string') {
        throw new ValidationError('Directory path must be a non-empty string.', { dirPath });
    }
    const resolvedPath = path.resolve(dirPath);
    if (!fs.existsSync(resolvedPath)) {
        fs.mkdirSync(resolvedPath, { recursive: true });
        logger.debug('Created directory structure', { path: resolvedPath });
    }
    return resolvedPath;
}

/**
 * Computes the normalized storage path for a document's vector database index.
 * 
 * @param {string} docPath - Path to the document.
 * @param {string} [baseDir='./vector/vector-db'] - Base directory for vector stores.
 * @returns {string} Resolved directory path for the document's vector index.
 * @throws {ValidationError} If document path is invalid.
 * 
 * @example
 * getVectorStorePath('files/book.pdf'); // './vector/vector-db/book'
 */
function getVectorStorePath(docPath, baseDir = './vector/vector-db') {
    const baseName = getSafeBasename(docPath);
    if (!baseName) {
        throw new ValidationError(`Cannot compute vector store path from invalid document path: "${docPath}"`, { docPath });
    }
    return path.join(baseDir, baseName);
}

/**
 * Scans a directory recursively or shallowly to find all supported document files.
 * 
 * @param {string} dirPath - Path to the directory to scan.
 * @param {Object} [options={}] - Scan options.
 * @param {boolean} [options.recursive=false] - Whether to scan subdirectories recursively.
 * @param {string[]} [options.allowedExtensions=SUPPORTED_EXTENSIONS] - Filter for extensions.
 * @returns {string[]} List of matching file paths.
 */
function listSupportedDocuments(dirPath, options = {}) {
    const { recursive = false, allowedExtensions = SUPPORTED_EXTENSIONS } = options;
    if (!dirPath || !fs.existsSync(dirPath)) {
        return [];
    }

    const results = [];
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });

    for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);
        if (entry.isDirectory()) {
            if (recursive) {
                results.push(...listSupportedDocuments(fullPath, options));
            }
        } else if (entry.isFile() && isSupportedFileType(entry.name, allowedExtensions)) {
            results.push(fullPath);
        }
    }

    return results;
}

/**
 * Multi-Document RAG Query Executor.
 * Queries across multiple documents using RAG, collecting context and answers.
 * 
 * @param {Object} params - Query parameters.
 * @param {string[]} params.docPaths - Array of document paths to query against.
 * @param {string} params.query - Question to ask across the documents.
 * @param {Function} [params.ragFunction] - Single document RAG function (injected from rag.js).
 * @param {Object} [params.options] - Additional options forwarded to ragFunction.
 * @returns {Promise<Array<{docPath: string, answer: string, success: boolean, error?: string}>>}
 * @throws {ValidationError} If docPaths or query is invalid.
 */
async function queryMultipleDocuments({ docPaths, query, ragFunction, options = {} }) {
    if (!Array.isArray(docPaths) || docPaths.length === 0) {
        throw new ValidationError('docPaths must be a non-empty array of file paths.', { docPaths });
    }
    if (!query || typeof query !== 'string') {
        throw new ValidationError('query must be a non-empty string.', { query });
    }

    const fn = ragFunction || require('./rag').documentRag;
    const results = [];

    for (const docPath of docPaths) {
        try {
            const resp = await fn(docPath, query, options);
            results.push({
                docPath,
                answer: resp && resp.answer ? resp.answer : String(resp),
                success: true,
                rawResponse: resp
            });
        } catch (err) {
            logger.warn('Failed querying document in multi-document batch', {
                docPath,
                errorCode: err.code || 'UNKNOWN_ERROR',
                error: err.message
            });
            results.push({
                docPath,
                answer: null,
                success: false,
                error: err.message,
                errorCode: err.code || 'UNKNOWN_ERROR'
            });
        }
    }

    return results;
}

/**
 * Batch-indexes multiple documents into their respective vector databases.
 * 
 * @param {string[]} docPaths - Array of document paths to index.
 * @param {Object} [options={}] - Options passed to indexer.
 * @param {Function} [options.indexerFunction] - Injected vector store creator.
 * @returns {Promise<Array<{docPath: string, indexed: boolean, vectorPath: string, error?: string}>>}
 * @throws {ValidationError} If docPaths is not an array.
 */
async function batchIndexDocuments(docPaths, options = {}) {
    if (!Array.isArray(docPaths)) {
        throw new ValidationError('docPaths must be an array.', { docPaths });
    }

    const { indexerFunction } = options;
    const indexer = indexerFunction || (async (p) => {
        const rag = require('./rag');
        return rag.loadAndIndexDocument(p, options);
    });

    const results = [];
    for (const docPath of docPaths) {
        const vectorPath = getVectorStorePath(docPath, options.baseDir);
        try {
            await indexer(docPath);
            results.push({ docPath, indexed: true, vectorPath });
        } catch (err) {
            logger.warn('Failed indexing document in batch', {
                docPath,
                errorCode: err.code || 'UNKNOWN_ERROR',
                error: err.message
            });
            results.push({ docPath, indexed: false, vectorPath, error: err.message, errorCode: err.code || 'UNKNOWN_ERROR' });
        }
    }
    return results;
}

// ==========================================
// CLI Diagnostic Demo Runner
// Preserves and upgrades original scratchpad checks
// ==========================================
if (require.main === module) {
    logger.info('=== MoreRAG Diagnostic & Utility Inspection ===');

    const sampleDocx = 'files/how_to_create_pdf_Gresham_Norman_CV_Clean.docx';
    const samplePdf = 'files/Gresham_Norman_CV_Clean.pdf';
    const vectorDir = './vector/vector-db';

    logger.info(`Inspecting sample DOCX: ${sampleDocx}`, {
        extension: getFileExtension(sampleDocx),
        isSupported: isSupportedFileType(sampleDocx),
        basename: getSafeBasename(sampleDocx),
        vectorPath: getVectorStorePath(sampleDocx)
    });

    logger.info(`Inspecting sample PDF: ${samplePdf}`, {
        extension: getFileExtension(samplePdf),
        isSupported: isSupportedFileType(samplePdf),
        basename: getSafeBasename(samplePdf),
        vectorPath: getVectorStorePath(samplePdf)
    });

    const dirExists = fs.existsSync(vectorDir);
    logger.info(`Vector DB Base Directory ("${vectorDir}"): ${dirExists ? 'EXISTS' : 'NOT FOUND'}`);

    if (fs.existsSync('files')) {
        const found = listSupportedDocuments('files');
        logger.info(`Available documents in "files/": ${found.length} detected`, { documents: found });
    }
    logger.info('=== Inspection Complete ===');
}

module.exports = {
    SUPPORTED_EXTENSIONS,
    getFileExtension,
    isSupportedFileType,
    getSafeBasename,
    ensureDirectoryExists,
    getVectorStorePath,
    listSupportedDocuments,
    queryMultipleDocuments,
    batchIndexDocuments
};
