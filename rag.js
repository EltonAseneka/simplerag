/**
 * @fileoverview rag.js - Production-ready Retrieval-Augmented Generation (RAG) Engine.
 * 
 * Provides end-to-end document loading, text chunking, vector indexing (HNSWLib/Memory),
 * semantic retrieval, and LLM-powered question answering with source attribution.
 * 
 * @module rag
 */

const fs = require('fs');
const path = require('path');
require('dotenv').config();

const { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { StringOutputParser } = require('@langchain/core/output_parsers');
const { ChatPromptTemplate } = require('@langchain/core/prompts');
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { PDFLoader } = require('@langchain/community/document_loaders/fs/pdf');
const { DocxLoader } = require('@langchain/community/document_loaders/fs/docx');
const { Document } = require('@langchain/core/documents');
const { HNSWLib } = require('@langchain/community/vectorstores/hnswlib');
const { MemoryVectorStore } = require('@langchain/classic/vectorstores/memory');
const { createStuffDocumentsChain } = require('@langchain/classic/chains/combine_documents');
const { createRetrievalChain } = require('@langchain/classic/chains/retrieval');

const {
    getFileExtension,
    isSupportedFileType,
    getSafeBasename,
    ensureDirectoryExists,
    getVectorStorePath
} = require('./morerag');

/**
 * Default configuration values for the RAG pipeline.
 */
const DEFAULT_CONFIG = Object.freeze({
    modelName: 'gemini-1.5-flash',
    temperature: 0.3,
    maxTokens: 1000,
    chunkSize: 1000,
    chunkOverlap: 200,
    k: 3,
    defaultVectorBaseDir: path.join(__dirname, 'vector', 'vector-db'),
    defaultPrompt: `You are an expert technical assistant. Use the following retrieved context to answer the question.
If the answer cannot be determined from the context, state clearly that you do not know.

Context:
{context}

Question:
{input}

Answer:`
});

/**
 * Lightweight text loader for plain text and markdown documents.
 */
class SimpleTextLoader {
    /**
     * @param {string} filePath - Path to the text or markdown file.
     */
    constructor(filePath) {
        this.filePath = filePath;
    }

    /**
     * Reads and parses file contents into a LangChain Document.
     * @returns {Promise<Array<Document>>}
     */
    async load() {
        const text = fs.readFileSync(this.filePath, 'utf-8');
        return [
            new Document({
                pageContent: text,
                metadata: { source: this.filePath }
            })
        ];
    }
}

/**
 * Resolves appropriate LangChain document loader based on file extension.
 * 
 * @param {string} filePath - Path to document file.
 * @returns {PDFLoader|DocxLoader|SimpleTextLoader} Initialized document loader instance.
 * @throws {TypeError} If filePath is invalid.
 * @throws {Error} If file extension is unsupported or file does not exist.
 */
function getLoaderForDocument(filePath) {
    if (!filePath || typeof filePath !== 'string') {
        throw new TypeError('Document path must be a non-empty string.');
    }

    const resolvedPath = path.resolve(filePath);
    if (!fs.existsSync(resolvedPath)) {
        throw new Error(`Document file not found at path: "${resolvedPath}"`);
    }

    const ext = getFileExtension(resolvedPath);
    switch (ext) {
        case 'pdf':
            return new PDFLoader(resolvedPath, { splitPages: true });
        case 'docx':
            return new DocxLoader(resolvedPath);
        case 'txt':
        case 'md':
            return new SimpleTextLoader(resolvedPath);
        default:
            throw new Error(
                `Unsupported document format: ".${ext}". Supported formats are: .pdf, .docx, .txt, .md`
            );
    }
}

/**
 * Loads a document and splits it into semantic chunks.
 * 
 * @param {string} docPath - Path to target document.
 * @param {Object} [options={}] - Chunking options.
 * @param {number} [options.chunkSize=1000] - Size of each chunk in characters.
 * @param {number} [options.chunkOverlap=200] - Character overlap between consecutive chunks.
 * @param {Object} [options.loader] - Optional custom loader instance for dependency injection.
 * @returns {Promise<Array<import('@langchain/core/documents').Document>>} Array of split Document chunks.
 */
async function loadAndSplitDocument(docPath, options = {}) {
    const chunkSize = options.chunkSize || DEFAULT_CONFIG.chunkSize;
    const chunkOverlap = options.chunkOverlap !== undefined ? options.chunkOverlap : DEFAULT_CONFIG.chunkOverlap;

    const loader = options.loader || getLoaderForDocument(docPath);
    const rawDocs = await loader.load();

    if (!rawDocs || rawDocs.length === 0) {
        throw new Error(`No content could be extracted from document: "${docPath}"`);
    }

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap
    });

    return await splitter.splitDocuments(rawDocs);
}

/**
 * Initializes embedding model based on available environment credentials or user override.
 * 
 * @param {Object} [options={}] - Embedding configuration options.
 * @param {Object} [options.embeddings] - Custom embeddings instance (e.g. for testing).
 * @param {'openai'|'google'} [options.provider] - Explicit embeddings provider choice.
 * @returns {OpenAIEmbeddings|GoogleGenerativeAIEmbeddings} Configured embeddings instance.
 */
function getEmbeddings(options = {}) {
    if (options.embeddings) {
        return options.embeddings;
    }

    const provider = options.provider || (process.env.OPENAI_API_KEY ? 'openai' : 'google');

    if (provider === 'openai') {
        if (!process.env.OPENAI_API_KEY) {
            throw new Error('OPENAI_API_KEY environment variable is required when using OpenAI embeddings.');
        }
        return new OpenAIEmbeddings({
            modelName: options.modelName || 'text-embedding-3-small'
        });
    }

    if (provider === 'google') {
        if (!process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
            throw new Error('GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for Google GenAI embeddings.');
        }
        return new GoogleGenerativeAIEmbeddings({
            model: options.modelName || 'text-embedding-004'
        });
    }

    throw new Error(`Unsupported embeddings provider: "${provider}". Use 'openai' or 'google'.`);
}

/**
 * Initializes Chat LLM instance with configurable parameters.
 * 
 * @param {Object} [options={}] - LLM configuration options.
 * @param {Object} [options.llm] - Injected LLM instance for testing.
 * @param {string} [options.modelName] - Model identifier (e.g., 'gemini-1.5-flash').
 * @param {number} [options.temperature] - Sampling temperature (0.0 - 1.0).
 * @param {number} [options.maxTokens] - Maximum tokens for generation.
 * @returns {ChatGoogleGenerativeAI} Configured Chat LLM instance.
 */
function getLlm(options = {}) {
    if (options.llm) {
        return options.llm;
    }

    if (!process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
        throw new Error('GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for ChatGoogleGenerativeAI.');
    }

    return new ChatGoogleGenerativeAI({
        model: options.modelName || DEFAULT_CONFIG.modelName,
        temperature: options.temperature !== undefined ? options.temperature : DEFAULT_CONFIG.temperature,
        maxTokens: options.maxTokens || DEFAULT_CONFIG.maxTokens,
        verbose: options.verbose || false
    });
}

/**
 * Checks whether a valid HNSWLib vector store index exists on disk.
 * 
 * @param {string} indexPath - Directory path where vector index is stored.
 * @returns {boolean} True if the index files exist, false otherwise.
 */
function isVectorStoreCached(indexPath) {
    if (!indexPath || !fs.existsSync(indexPath)) {
        return false;
    }
    const docstorePath = path.join(indexPath, 'docstore.json');
    const argsPath = path.join(indexPath, 'args.json');
    return fs.existsSync(docstorePath) && fs.existsSync(argsPath);
}

/**
 * Retrieves existing vector store from disk or creates and persists a new one.
 * Falls back to in-memory store if disk persistence is disabled.
 * 
 * @param {Array<import('@langchain/core/documents').Document>} splitDocs - Chunks to index.
 * @param {Object} embeddings - Embeddings model instance.
 * @param {Object} [options={}] - Vector store options.
 * @param {string} [options.indexPath] - Target disk directory for index.
 * @param {boolean} [options.inMemory=false] - Force in-memory vector store without disk persistence.
 * @returns {Promise<HNSWLib|MemoryVectorStore>} Vector store instance.
 */
async function getOrCreateVectorStore(splitDocs, embeddings, options = {}) {
    if (options.inMemory) {
        return await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
    }

    const indexPath = options.indexPath;
    if (!indexPath) {
        throw new Error('indexPath is required for persistent vector stores.');
    }

    if (isVectorStoreCached(indexPath)) {
        return await HNSWLib.load(indexPath, embeddings);
    }

    ensureDirectoryExists(indexPath);
    const vectorStore = await HNSWLib.fromDocuments(splitDocs, embeddings);
    await vectorStore.save(indexPath);
    return vectorStore;
}

/**
 * Constructs the runnable LangChain RAG pipeline.
 * 
 * @param {Object} params - Chain parameters.
 * @param {Object} params.llm - Chat model instance.
 * @param {Object} params.vectorStore - Vector store instance.
 * @param {number} [params.k=3] - Number of context chunks to retrieve.
 * @param {string} [params.promptTemplate] - Custom prompt template string.
 * @returns {Promise<Object>} LangChain retrieval chain instance.
 */
async function createRagChain({ llm, vectorStore, k = DEFAULT_CONFIG.k, promptTemplate = DEFAULT_CONFIG.defaultPrompt }) {
    const prompt = ChatPromptTemplate.fromTemplate(promptTemplate);
    const combineDocsChain = await createStuffDocumentsChain({
        llm,
        prompt
    });

    const retriever = vectorStore.asRetriever({ k });

    return await createRetrievalChain({
        combineDocsChain,
        retriever
    });
}

/**
 * Executes an end-to-end RAG query against a document.
 * 
 * @param {string} docPath - Path to the document file (.pdf, .docx, .txt, .md).
 * @param {string} [query] - Question to ask against document context.
 * @param {Object} [options={}] - Pipeline configuration options.
 * @param {number} [options.k=3] - Number of retrieved chunks.
 * @param {Object} [options.llm] - Custom LLM instance.
 * @param {Object} [options.embeddings] - Custom embeddings instance.
 * @param {string} [options.vectorBaseDir] - Base directory for vector index storage.
 * @param {boolean} [options.inMemory=false] - Use in-memory vector store instead of disk.
 * @param {boolean} [options.verbose=false] - Enable debug logging and timing.
 * @returns {Promise<{answer: string, context: Array<Object>, executionTimeMs: number, sourceDoc: string}>}
 * 
 * @example
 * const result = await documentRag('files/manual.pdf', 'How do I reset password?');
 * console.log(result.answer);
 */
async function documentRag(docPath, query = 'Compare B-Trees and LSM Trees according to the book.', options = {}) {
    if (!docPath || typeof docPath !== 'string') {
        throw new TypeError('docPath must be a non-empty string.');
    }

    const startTime = Date.now();
    const resolvedDocPath = path.resolve(docPath);

    if (!fs.existsSync(resolvedDocPath)) {
        throw new Error(`Target document not found at: "${resolvedDocPath}"`);
    }

    const baseDir = options.vectorBaseDir || DEFAULT_CONFIG.defaultVectorBaseDir;
    const baseName = getSafeBasename(resolvedDocPath);
    const indexPath = path.join(baseDir, baseName);

    const embeddings = getEmbeddings(options);
    const llm = getLlm(options);

    let vectorStore;
    if (!options.inMemory && isVectorStoreCached(indexPath)) {
        if (options.verbose) console.log(`[RAG] Loading cached vector index from: ${indexPath}`);
        vectorStore = await HNSWLib.load(indexPath, embeddings);
    } else {
        if (options.verbose) console.log(`[RAG] Processing and chunking document: ${resolvedDocPath}`);
        const splitDocs = await loadAndSplitDocument(resolvedDocPath, options);

        if (options.verbose) console.log(`[RAG] Indexing ${splitDocs.length} chunks...`);
        vectorStore = await getOrCreateVectorStore(splitDocs, embeddings, {
            indexPath,
            inMemory: options.inMemory
        });
    }

    const chain = await createRagChain({
        llm,
        vectorStore,
        k: options.k || DEFAULT_CONFIG.k,
        promptTemplate: options.promptTemplate || DEFAULT_CONFIG.defaultPrompt
    });

    const response = await chain.invoke({
        input: query
    });

    const executionTimeMs = Date.now() - startTime;

    return {
        answer: response.answer,
        context: response.context || [],
        executionTimeMs,
        sourceDoc: resolvedDocPath
    };
}

/**
 * Loads and indexes a document into the vector database without executing a question query.
 * Useful for pre-indexing and batch processing.
 * 
 * @param {string} docPath - Path to target document.
 * @param {Object} [options={}] - Ingestion options.
 * @returns {Promise<{indexPath: string, chunkCount: number}>}
 */
async function loadAndIndexDocument(docPath, options = {}) {
    const resolvedDocPath = path.resolve(docPath);
    const baseDir = options.vectorBaseDir || DEFAULT_CONFIG.defaultVectorBaseDir;
    const baseName = getSafeBasename(resolvedDocPath);
    const indexPath = path.join(baseDir, baseName);

    const embeddings = getEmbeddings(options);
    const splitDocs = await loadAndSplitDocument(resolvedDocPath, options);
    await getOrCreateVectorStore(splitDocs, embeddings, { indexPath });

    return {
        indexPath,
        chunkCount: splitDocs.length
    };
}

// ==========================================
// CLI Execution Guard
// Only executes when run directly via `node rag.js`
// ==========================================
if (require.main === module) {
    (async () => {
        try {
            const targetDoc = process.argv[2] || path.join(__dirname, 'files', 'Designing_Data-Intensive_Applications_TH.pdf');
            const targetQuery = process.argv[3] || 'Compare B-Trees and LSM Trees according to the book.';

            console.log('=== Document RAG Execution ===');
            console.log(`Document : ${targetDoc}`);
            console.log(`Query    : ${targetQuery}\n`);

            if (!fs.existsSync(targetDoc)) {
                console.error(`Error: File not found at path "${targetDoc}".`);
                console.log('Usage: node rag.js <path-to-document> [question]');
                process.exit(1);
            }

            if (!process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
                console.warn('Warning: GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.');
            }
            if (!process.env.OPENAI_API_KEY) {
                console.warn('Warning: OPENAI_API_KEY not found in environment (required if using OpenAI embeddings).');
            }

            const result = await documentRag(targetDoc, targetQuery, { verbose: true });
            console.log('\n--- Answer ---');
            console.log(result.answer);
            console.log(`\nCompleted in ${result.executionTimeMs}ms (retrieved ${result.context.length} context chunks).`);
        } catch (err) {
            console.error('\nExecution failed:', err.message);
            process.exit(1);
        }
    })();
}

module.exports = {
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
};
