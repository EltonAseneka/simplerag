/**
 * @fileoverview Unit and Integration tests for rag.js RAG pipeline.
 */

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const fs = require('fs');

const { Document } = require('@langchain/core/documents');
const { PDFLoader } = require('@langchain/community/document_loaders/fs/pdf');
const { DocxLoader } = require('@langchain/community/document_loaders/fs/docx');
const { RunnableLambda } = require('@langchain/core/runnables');

const {
    DEFAULT_CONFIG,
    getLoaderForDocument,
    loadAndSplitDocument,
    getEmbeddings,
    getLlm,
    isVectorStoreCached,
    getOrCreateVectorStore,
    createRagChain,
    documentRag,
    loadAndIndexDocument
} = require('../rag');

// Lightweight mock embeddings for fast, deterministic, offline testing
class MockEmbeddings {
    async embedDocuments(documents) {
        return documents.map(() => [0.1, 0.2, 0.3, 0.4]);
    }

    async embedQuery() {
        return [0.1, 0.2, 0.3, 0.4];
    }
}

// Helper function to create a mock LangChain Runnable LLM
function createMockChatModel(reply = 'Mocked technical response based on context.') {
    return RunnableLambda.from(async (promptVal) => {
        return reply;
    });
}

describe('rag.js - Configuration and Defaults', () => {
    it('DEFAULT_CONFIG should be frozen and contain pipeline constants', () => {
        assert.ok(Object.isFrozen(DEFAULT_CONFIG));
        assert.equal(DEFAULT_CONFIG.chunkSize, 1000);
        assert.equal(DEFAULT_CONFIG.chunkOverlap, 200);
        assert.equal(DEFAULT_CONFIG.k, 3);
        assert.ok(DEFAULT_CONFIG.defaultPrompt.includes('{context}'));
        assert.ok(DEFAULT_CONFIG.defaultPrompt.includes('{input}'));
    });
});

describe('rag.js - Document Loader Resolution', () => {
    const tempDir = path.join(__dirname, 'temp_loader_test');
    const pdfPath = path.join(tempDir, 'test.pdf');
    const docxPath = path.join(tempDir, 'test.docx');
    const txtPath = path.join(tempDir, 'test.txt');
    const mdPath = path.join(tempDir, 'test.md');
    const invalidPath = path.join(tempDir, 'test.xyz');

    before(() => {
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }
        fs.writeFileSync(pdfPath, 'dummy pdf binary content');
        fs.writeFileSync(docxPath, 'dummy docx content');
        fs.writeFileSync(txtPath, 'Hello world text document');
        fs.writeFileSync(mdPath, '# Markdown Title\nContent paragraph');
        fs.writeFileSync(invalidPath, 'some unknown format');
    });

    after(() => {
        if (fs.existsSync(tempDir)) {
            fs.rmSync(tempDir, { recursive: true, force: true });
        }
    });

    it('should throw TypeError for invalid docPath arguments', () => {
        assert.throws(() => getLoaderForDocument(''), TypeError);
        assert.throws(() => getLoaderForDocument(null), TypeError);
        assert.throws(() => getLoaderForDocument(12345), TypeError);
    });

    it('should throw Error for nonexistent files', () => {
        assert.throws(
            () => getLoaderForDocument(path.join(tempDir, 'nonexistent.pdf')),
            /Document file not found/
        );
    });

    it('should return PDFLoader for .pdf files', () => {
        const loader = getLoaderForDocument(pdfPath);
        assert.ok(loader instanceof PDFLoader);
    });

    it('should return DocxLoader for .docx files', () => {
        const loader = getLoaderForDocument(docxPath);
        assert.ok(loader instanceof DocxLoader);
    });

    it('should return SimpleTextLoader for .txt and .md files and load content', async () => {
        const txtLoader = getLoaderForDocument(txtPath);
        const docs = await txtLoader.load();
        assert.equal(docs.length, 1);
        assert.equal(docs[0].pageContent, 'Hello world text document');
        assert.equal(docs[0].metadata.source, path.resolve(txtPath));

        const mdLoader = getLoaderForDocument(mdPath);
        const mdDocs = await mdLoader.load();
        assert.equal(mdDocs.length, 1);
        assert.ok(mdDocs[0].pageContent.includes('# Markdown Title'));
    });

    it('should throw informative error for unsupported document types', () => {
        assert.throws(
            () => getLoaderForDocument(invalidPath),
            /Unsupported document format: "\.xyz"/
        );
    });
});

describe('rag.js - Text Splitting and Chunking', () => {
    it('should split loaded documents into chunks respecting size and overlap', async () => {
        // Create a 2500 character document
        const longText = 'A'.repeat(1200) + ' ' + 'B'.repeat(1200);
        const mockLoader = {
            load: async () => [new Document({ pageContent: longText, metadata: { source: 'memory' } })]
        };

        const chunks = await loadAndSplitDocument('memory.txt', {
            loader: mockLoader,
            chunkSize: 500,
            chunkOverlap: 50
        });

        assert.ok(chunks.length > 2);
        assert.ok(chunks[0].pageContent.length <= 500);
    });

    it('should throw error when document yields empty content', async () => {
        const emptyLoader = {
            load: async () => []
        };

        await assert.rejects(
            () => loadAndSplitDocument('empty.txt', { loader: emptyLoader }),
            /No content could be extracted/
        );
    });
});

describe('rag.js - Embeddings and LLM Configuration', () => {
    it('getEmbeddings() should return custom injected embeddings', () => {
        const mockEmb = new MockEmbeddings();
        const resolved = getEmbeddings({ embeddings: mockEmb });
        assert.equal(resolved, mockEmb);
    });

    it('getEmbeddings() should reject unsupported providers', () => {
        assert.throws(
            () => getEmbeddings({ provider: 'unsupported-provider' }),
            /Unsupported embeddings provider/
        );
    });

    it('getLlm() should return injected custom LLM instance', () => {
        const mockModel = createMockChatModel();
        const resolved = getLlm({ llm: mockModel });
        assert.equal(resolved, mockModel);
    });
});

describe('rag.js - Vector Store Caching and Storage', () => {
    const cacheTestDir = path.join(__dirname, 'temp_cache_test');

    before(() => {
        if (!fs.existsSync(cacheTestDir)) {
            fs.mkdirSync(cacheTestDir, { recursive: true });
        }
    });

    after(() => {
        if (fs.existsSync(cacheTestDir)) {
            fs.rmSync(cacheTestDir, { recursive: true, force: true });
        }
    });

    it('isVectorStoreCached() should correctly detect valid or missing indexes', () => {
        assert.equal(isVectorStoreCached(cacheTestDir), false);

        fs.writeFileSync(path.join(cacheTestDir, 'docstore.json'), '{}');
        assert.equal(isVectorStoreCached(cacheTestDir), false);

        fs.writeFileSync(path.join(cacheTestDir, 'args.json'), '{}');
        assert.equal(isVectorStoreCached(cacheTestDir), true);
    });

    it('getOrCreateVectorStore() should create in-memory store when specified', async () => {
        const docs = [new Document({ pageContent: 'Memory store document chunk', metadata: {} })];
        const store = await getOrCreateVectorStore(docs, new MockEmbeddings(), { inMemory: true });
        assert.ok(store);
        const searchResults = await store.similaritySearch('Memory', 1);
        assert.equal(searchResults.length, 1);
        assert.equal(searchResults[0].pageContent, 'Memory store document chunk');
    });

    it('getOrCreateVectorStore() should require indexPath for persistent mode', async () => {
        const docs = [new Document({ pageContent: 'chunk', metadata: {} })];
        await assert.rejects(
            () => getOrCreateVectorStore(docs, new MockEmbeddings(), { inMemory: false }),
            /indexPath is required/
        );
    });
});

describe('rag.js - End-to-End Pipeline & documentRag()', () => {
    const e2eDir = path.join(__dirname, 'temp_e2e_test');
    const docFile = path.join(e2eDir, 'sample_tech_guide.txt');

    before(() => {
        if (!fs.existsSync(e2eDir)) {
            fs.mkdirSync(e2eDir, { recursive: true });
        }
        const content = `
            Database Architecture Overview:
            B-Trees keep keys sorted and perform updates in-place via page writes.
            LSM Trees buffer writes in an in-memory memtable and append them to disk SSTables sequentially.
            LSM trees offer higher write throughput, while B-Trees provide predictable read performance.
        `;
        fs.writeFileSync(docFile, content.trim());
    });

    after(() => {
        if (fs.existsSync(e2eDir)) {
            fs.rmSync(e2eDir, { recursive: true, force: true });
        }
    });

    it('documentRag() should validate docPath input', async () => {
        await assert.rejects(() => documentRag(''), TypeError);
        await assert.rejects(() => documentRag(null), TypeError);
        await assert.rejects(
            () => documentRag('nonexistent_e2e_doc.txt'),
            /Target document not found/
        );
    });

    it('documentRag() should run end-to-end with mock LLM and mock embeddings', async () => {
        const expectedAnswer = 'LSM trees offer higher write throughput, whereas B-Trees provide predictable reads.';
        const mockModel = createMockChatModel(expectedAnswer);
        const mockEmb = new MockEmbeddings();

        const result = await documentRag(docFile, 'Compare B-Trees and LSM Trees.', {
            llm: mockModel,
            embeddings: mockEmb,
            inMemory: true,
            k: 2
        });

        assert.ok(result);
        assert.equal(typeof result.answer, 'string');
        assert.equal(result.answer, expectedAnswer);
        assert.ok(Array.isArray(result.context));
        assert.ok(result.context.length > 0);
        assert.equal(result.sourceDoc, path.resolve(docFile));
        assert.equal(typeof result.executionTimeMs, 'number');
        assert.ok(result.executionTimeMs >= 0);
    });
});
