/**
 * @fileoverview Unit tests for morerag.js document utilities and multi-document RAG operations.
 */

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const fs = require('fs');

const {
    SUPPORTED_EXTENSIONS,
    getFileExtension,
    isSupportedFileType,
    getSafeBasename,
    ensureDirectoryExists,
    getVectorStorePath,
    listSupportedDocuments,
    queryMultipleDocuments,
    batchIndexDocuments
} = require('../morerag');

describe('morerag.js - File & Path Utilities', () => {
    describe('SUPPORTED_EXTENSIONS', () => {
        it('should include core document formats', () => {
            assert.ok(SUPPORTED_EXTENSIONS.includes('pdf'));
            assert.ok(SUPPORTED_EXTENSIONS.includes('docx'));
            assert.ok(SUPPORTED_EXTENSIONS.includes('txt'));
            assert.ok(SUPPORTED_EXTENSIONS.includes('md'));
        });

        it('should be frozen and immutable', () => {
            assert.ok(Object.isFrozen(SUPPORTED_EXTENSIONS));
        });
    });

    describe('getFileExtension()', () => {
        it('should return lowercase extension without leading dot', () => {
            assert.equal(getFileExtension('document.pdf'), 'pdf');
            assert.equal(getFileExtension('notes.DOCX'), 'docx');
            assert.equal(getFileExtension('path/to/file.TXT'), 'txt');
            assert.equal(getFileExtension('README.MD'), 'md');
        });

        it('should handle filenames with multiple dots', () => {
            assert.equal(getFileExtension('archive.tar.gz'), 'gz');
            assert.equal(getFileExtension('data.backup.2026.pdf'), 'pdf');
        });

        it('should return empty string for files without extension or invalid input', () => {
            assert.equal(getFileExtension('Dockerfile'), '');
            assert.equal(getFileExtension(''), '');
            assert.equal(getFileExtension(null), '');
            assert.equal(getFileExtension(undefined), '');
            assert.equal(getFileExtension(123), '');
        });

        it('should trim surrounding whitespace', () => {
            assert.equal(getFileExtension('  sample.pdf  '), 'pdf');
        });
    });

    describe('isSupportedFileType()', () => {
        it('should return true for supported extensions', () => {
            assert.equal(isSupportedFileType('book.pdf'), true);
            assert.equal(isSupportedFileType('resume.docx'), true);
            assert.equal(isSupportedFileType('notes.txt'), true);
            assert.equal(isSupportedFileType('guide.md'), true);
        });

        it('should return false for unsupported extensions', () => {
            assert.equal(isSupportedFileType('binary.exe'), false);
            assert.equal(isSupportedFileType('image.png'), false);
            assert.equal(isSupportedFileType('script.js'), false);
            assert.equal(isSupportedFileType('noextension'), false);
        });

        it('should respect custom allowed extensions list', () => {
            assert.equal(isSupportedFileType('data.csv', ['csv', 'json']), true);
            assert.equal(isSupportedFileType('doc.pdf', ['csv', 'json']), false);
        });
    });

    describe('getSafeBasename()', () => {
        it('should extract clean filename without extension', () => {
            assert.equal(getSafeBasename('files/sample.pdf'), 'sample');
            assert.equal(getSafeBasename('files/how_to_create_pdf_CV.docx'), 'how_to_create_pdf_CV');
        });

        it('should handle Windows backslashes and POSIX slashes correctly', () => {
            assert.equal(getSafeBasename('C:\\docs\\folder\\report.pdf'), 'report');
            assert.equal(getSafeBasename('/home/user/docs/report.pdf'), 'report');
        });

        it('should preserve intermediate dots in base filename', () => {
            assert.equal(getSafeBasename('release.v1.0.final.docx'), 'release.v1.0.final');
        });

        it('should return empty string for invalid inputs', () => {
            assert.equal(getSafeBasename(''), '');
            assert.equal(getSafeBasename(null), '');
            assert.equal(getSafeBasename(undefined), '');
        });
    });

    describe('ensureDirectoryExists()', () => {
        const testDir = path.join(__dirname, 'temp_test_dir', 'nested_sub');

        after(() => {
            const rootTemp = path.join(__dirname, 'temp_test_dir');
            if (fs.existsSync(rootTemp)) {
                fs.rmSync(rootTemp, { recursive: true, force: true });
            }
        });

        it('should create nested directories if they do not exist', () => {
            assert.equal(fs.existsSync(testDir), false);
            const created = ensureDirectoryExists(testDir);
            assert.ok(fs.existsSync(testDir));
            assert.equal(created, path.resolve(testDir));
        });

        it('should not fail if directory already exists', () => {
            assert.doesNotThrow(() => ensureDirectoryExists(testDir));
        });

        it('should throw TypeError on invalid directory path', () => {
            assert.throws(() => ensureDirectoryExists(''), TypeError);
            assert.throws(() => ensureDirectoryExists(null), TypeError);
        });
    });

    describe('getVectorStorePath()', () => {
        it('should compute standard destination path for vector index', () => {
            const dest = getVectorStorePath('files/book.pdf', './vector/db');
            assert.equal(dest, path.join('./vector/db', 'book'));
        });

        it('should throw an error for invalid document paths', () => {
            assert.throws(() => getVectorStorePath(''), Error);
        });
    });

    describe('listSupportedDocuments()', () => {
        const testScanDir = path.join(__dirname, 'temp_scan_dir');

        before(() => {
            ensureDirectoryExists(testScanDir);
            fs.writeFileSync(path.join(testScanDir, 'doc1.pdf'), 'fake pdf');
            fs.writeFileSync(path.join(testScanDir, 'doc2.docx'), 'fake docx');
            fs.writeFileSync(path.join(testScanDir, 'notes.txt'), 'fake txt');
            fs.writeFileSync(path.join(testScanDir, 'ignore.exe'), 'binary');
        });

        after(() => {
            if (fs.existsSync(testScanDir)) {
                fs.rmSync(testScanDir, { recursive: true, force: true });
            }
        });

        it('should find all supported document files in a directory', () => {
            const files = listSupportedDocuments(testScanDir);
            assert.equal(files.length, 3);
            const baseNames = files.map(f => path.basename(f)).sort();
            assert.deepEqual(baseNames, ['doc1.pdf', 'doc2.docx', 'notes.txt']);
        });

        it('should return empty array for non-existent directory', () => {
            const files = listSupportedDocuments(path.join(__dirname, 'nonexistent_directory_12345'));
            assert.deepEqual(files, []);
        });
    });
});

describe('morerag.js - Multi-Document RAG Operations', () => {
    describe('queryMultipleDocuments()', () => {
        it('should validate docPaths and query inputs', async () => {
            await assert.rejects(
                () => queryMultipleDocuments({ docPaths: [], query: 'test' }),
                /docPaths must be a non-empty array/
            );
            await assert.rejects(
                () => queryMultipleDocuments({ docPaths: ['file.pdf'], query: '' }),
                /query must be a non-empty string/
            );
        });

        it('should execute query across multiple documents using injected RAG function', async () => {
            const mockRag = async (docPath, query) => ({
                answer: `Answer for ${query} from ${docPath}`
            });

            const results = await queryMultipleDocuments({
                docPaths: ['docA.pdf', 'docB.docx'],
                query: 'What is the summary?',
                ragFunction: mockRag
            });

            assert.equal(results.length, 2);
            assert.equal(results[0].success, true);
            assert.equal(results[0].docPath, 'docA.pdf');
            assert.equal(results[0].answer, 'Answer for What is the summary? from docA.pdf');
            assert.equal(results[1].success, true);
            assert.equal(results[1].docPath, 'docB.docx');
        });

        it('should isolate errors per document without terminating entire batch', async () => {
            const mockRag = async (docPath) => {
                if (docPath === 'failing.pdf') {
                    throw new Error('Failed to parse document');
                }
                return { answer: 'Success answer' };
            };

            const results = await queryMultipleDocuments({
                docPaths: ['success.pdf', 'failing.pdf'],
                query: 'Test query',
                ragFunction: mockRag
            });

            assert.equal(results.length, 2);
            assert.equal(results[0].success, true);
            assert.equal(results[0].answer, 'Success answer');
            assert.equal(results[1].success, false);
            assert.equal(results[1].error, 'Failed to parse document');
        });
    });

    describe('batchIndexDocuments()', () => {
        it('should validate docPaths is an array', async () => {
            await assert.rejects(() => batchIndexDocuments(null), TypeError);
        });

        it('should process batch indexing using injected indexer', async () => {
            const mockIndexer = async (docPath) => {
                if (docPath === 'corrupted.pdf') throw new Error('Corrupted PDF');
                return { indexed: true };
            };

            const results = await batchIndexDocuments(
                ['valid1.pdf', 'corrupted.pdf', 'valid2.pdf'],
                { indexerFunction: mockIndexer, baseDir: './vector/test-db' }
            );

            assert.equal(results.length, 3);
            assert.equal(results[0].indexed, true);
            assert.equal(results[1].indexed, false);
            assert.equal(results[1].error, 'Corrupted PDF');
            assert.equal(results[2].indexed, true);
        });
    });
});
