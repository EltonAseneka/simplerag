/**
 * @fileoverview morerag.js - Document Path Utilities and Multi-Document RAG Operations.
 * 
 * Provides robust file system utilities, document format detection, directory ingestion,
 * and multi-document retrieval-augmented generation (RAG) capabilities.
 * 
 * @module morerag
 */

const path = require('path');
const fs = require('fs');

/**
 * Supported document file extensions for ingestion and vector indexing.
 * @type {ReadonlyArray<string>}
 */
const SUPPORTED_EXTENSIONS = Object.freeze(['pdf', 'docx', 'txt', 'md']);

/**
 * Extracts the file extension in normalized, lowercase form without the leading dot.
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
    const ext = path.extname(filePath.trim());
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
 * Handles both POSIX and Windows separators.
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
    const normalized = path.normalize(filePath.trim());
    const ext = path.extname(normalized);
    return path.basename(normalized, ext);
}

/**
 * Ensures that a target directory exists, creating all parent directories if necessary.
 * 
 * @param {string} dirPath - Directory path to verify or create.
 * @returns {string} The resolved absolute path of the created/verified directory.
 * @throws {Error} If directory creation fails.
 */
function ensureDirectoryExists(dirPath) {
    if (!dirPath || typeof dirPath !== 'string') {
        throw new TypeError('Directory path must be a non-empty string.');
    }
    const resolvedPath = path.resolve(dirPath);
    if (!fs.existsSync(resolvedPath)) {
        fs.mkdirSync(resolvedPath, { recursive: true });
    }
    return resolvedPath;
}

/**
 * Computes the normalized storage path for a document's vector database index.
 * 
 * @param {string} docPath - Path to the document.
 * @param {string} [baseDir='./vector/vector-db'] - Base directory for vector stores.
 * @returns {string} Resolved directory path for the document's vector index.
 * 
 * @example
 * getVectorStorePath('files/book.pdf'); // './vector/vector-db/book'
 */
function getVectorStorePath(docPath, baseDir = './vector/vector-db') {
    const baseName = getSafeBasename(docPath);
    if (!baseName) {
        throw new Error(`Cannot compute vector store path from invalid document path: "${docPath}"`);
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
 */
async function queryMultipleDocuments({ docPaths, query, ragFunction, options = {} }) {
    if (!Array.isArray(docPaths) || docPaths.length === 0) {
        throw new Error('docPaths must be a non-empty array of file paths.');
    }
    if (!query || typeof query !== 'string') {
        throw new Error('query must be a non-empty string.');
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
            results.push({
                docPath,
                answer: null,
                success: false,
                error: err.message
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
 */
async function batchIndexDocuments(docPaths, options = {}) {
    if (!Array.isArray(docPaths)) {
        throw new TypeError('docPaths must be an array.');
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
            results.push({ docPath, indexed: false, vectorPath, error: err.message });
        }
    }
    return results;
}

// ==========================================
// CLI Diagnostic Demo Runner
// Preserves and upgrades original scratchpad checks
// ==========================================
if (require.main === module) {
    console.log('=== MoreRAG Diagnostic & Utility Inspection ===\n');

    const sampleDocx = 'files/how_to_create_pdf_Gresham_Norman_CV_Clean.docx';
    const samplePdf = 'files/Gresham_Norman_CV_Clean.pdf';
    const vectorDir = './vector/vector-db';

    console.log(`Document 1: "${sampleDocx}"`);
    console.log(` - Detected Extension: "${getFileExtension(sampleDocx)}"`);
    console.log(` - Is Supported Document: ${isSupportedFileType(sampleDocx)}`);
    console.log(` - Safe Basename: "${getSafeBasename(sampleDocx)}"`);
    console.log(` - Vector DB Destination: "${getVectorStorePath(sampleDocx)}"`);

    console.log(`\nDocument 2: "${samplePdf}"`);
    console.log(` - Detected Extension: "${getFileExtension(samplePdf)}"`);
    console.log(` - Is Supported Document: ${isSupportedFileType(samplePdf)}`);
    console.log(` - Safe Basename: "${getSafeBasename(samplePdf)}"`);
    console.log(` - Vector DB Destination: "${getVectorStorePath(samplePdf)}"`);

    const dirExists = fs.existsSync(vectorDir);
    console.log(`\nVector DB Base Directory ("${vectorDir}"): ${dirExists ? 'EXISTS' : 'NOT FOUND'}`);

    if (fs.existsSync('files')) {
        const found = listSupportedDocuments('files');
        console.log(`\nAvailable documents in "files/":`);
        found.forEach(f => console.log(` - ${f}`));
    }
    console.log('\n=== Inspection Complete ===');
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
