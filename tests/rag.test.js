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

const {
    ValidationError,
    DocumentNotFoundError,
    UnsupportedFormatError,
    EmptyDocumentError,
    EmbeddingProviderError,
    LLMProviderError,
    VectorStoreError
} = require('../lib/errors');

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
    return RunnableLambda.from(async (_promptVal) => {
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

    it('should throw ValidationError for invalid docPath arguments', () => {
        assert.throws(() => getLoaderForDocument(''), ValidationError);
        assert.throws(() => getLoaderForDocument(null), ValidationError);
        assert.throws(() => getLoaderForDocument(12345), ValidationError);
    });

    it('should throw DocumentNotFoundError for nonexistent files', () => {
        const missingPath = path.join(tempDir, 'nonexistent.pdf');
        assert.throws(
            () => getLoaderForDocument(missingPath),
            (err) => {
                assert.ok(err instanceof DocumentNotFoundError);
                assert.equal(err.code, 'DOCUMENT_NOT_FOUND');
                return true;
            }
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

    it('should throw UnsupportedFormatError for unsupported document types', () => {
        assert.throws(
            () => getLoaderForDocument(invalidPath),
            (err) => {
                assert.ok(err instanceof UnsupportedFormatError);
                assert.equal(err.code, 'UNSUPPORTED_FORMAT');
                return true;
            }
        );
    });
});

describe('rag.js - Text Splitting and Chunking', () => {
    it('should split loaded documents into chunks respecting size and overlap', async () => {
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

    it('should throw EmptyDocumentError when document yields empty content', async () => {
        const emptyLoader = {
            load: async () => []
        };

        await assert.rejects(
            () => loadAndSplitDocument('empty.txt', { loader: emptyLoader }),
            (err) => {
                assert.ok(err instanceof EmptyDocumentError);
                assert.equal(err.code, 'EMPTY_DOCUMENT');
                return true;
            }
        );
    });
});

describe('rag.js - Embeddings and LLM Configuration', () => {
    it('getEmbeddings() should return custom injected embeddings', () => {
        const mockEmb = new MockEmbeddings();
        const resolved = getEmbeddings({ embeddings: mockEmb });
        assert.equal(resolved, mockEmb);
    });

    it('getEmbeddings() should throw EmbeddingProviderError when OPENAI_API_KEY is missing', () => {
        assert.throws(
            () => getEmbeddings({ provider: 'openai' }),
            (err) => {
                assert.ok(err instanceof EmbeddingProviderError);
                assert.equal(err.code, 'EMBEDDING_PROVIDER_ERROR');
                assert.equal(err.details.provider, 'openai');
                return true;
            }
        );
    });

    it('getEmbeddings() should throw EmbeddingProviderError when Google keys are missing', () => {
        assert.throws(
            () => getEmbeddings({ provider: 'google' }),
            (err) => {
                assert.ok(err instanceof EmbeddingProviderError);
                assert.equal(err.code, 'EMBEDDING_PROVIDER_ERROR');
                assert.equal(err.details.provider, 'google');
                return true;
            }
        );
    });

    it('getEmbeddings() should reject unsupported providers', () => {
        assert.throws(
            () => getEmbeddings({ provider: 'unsupported-provider' }),
            (err) => {
                assert.ok(err instanceof EmbeddingProviderError);
                return true;
            }
        );
    });

    it('getLlm() should return injected custom LLM instance', () => {
        const mockModel = createMockChatModel();
        const resolved = getLlm({ llm: mockModel });
        assert.equal(resolved, mockModel);
    });

    it('getLlm() should throw LLMProviderError when API keys are missing', () => {
        assert.throws(
            () => getLlm(),
            (err) => {
                assert.ok(err instanceof LLMProviderError);
                assert.equal(err.code, 'LLM_PROVIDER_ERROR');
                return true;
            }
        );
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
            (err) => {
                assert.ok(err instanceof VectorStoreError);
                assert.equal(err.code, 'VECTOR_STORE_ERROR');
                return true;
            }
        );
    });
});

describe('rag.js - createRagChain() and loadAndIndexDocument()', () => {
    it('createRagChain() should construct executable retrieval chain', async () => {
        const mockModel = createMockChatModel('Direct chain answer');
        const docs = [new Document({ pageContent: 'Chain test chunk', metadata: {} })];
        const vectorStore = await getOrCreateVectorStore(docs, new MockEmbeddings(), { inMemory: true });

        const chain = await createRagChain({
            llm: mockModel,
            vectorStore,
            k: 1
        });

        assert.ok(chain);
        const res = await chain.invoke({ input: 'Query test' });
        assert.equal(res.answer, 'Direct chain answer');
    });

    it('loadAndIndexDocument() should index file chunks to disk destination', async () => {
        const tempIndexDir = path.join(__dirname, 'temp_index_unit_test');
        const tempDoc = path.join(tempIndexDir, 'sample.txt');

        if (!fs.existsSync(tempIndexDir)) {
            fs.mkdirSync(tempIndexDir, { recursive: true });
        }
        fs.writeFileSync(tempDoc, 'Content for indexing test verification.');

        const res = await loadAndIndexDocument(tempDoc, {
            embeddings: new MockEmbeddings(),
            vectorBaseDir: tempIndexDir
        });

        assert.ok(res.indexPath);
        assert.equal(res.chunkCount, 1);
        assert.ok(fs.existsSync(path.join(res.indexPath, 'docstore.json')));

        if (fs.existsSync(tempIndexDir)) {
            fs.rmSync(tempIndexDir, { recursive: true, force: true });
        }
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
        await assert.rejects(() => documentRag(''), ValidationError);
        await assert.rejects(() => documentRag(null), ValidationError);
        await assert.rejects(
            () => documentRag('nonexistent_e2e_doc.txt'),
            DocumentNotFoundError
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
