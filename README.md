# SimpleRAG: Production-Ready Document Retrieval-Augmented Generation

A modular, robust, and well-tested Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Google Gemini**, **OpenAI Embeddings**, and **HNSWLib** on Node.js.

---

## Architecture Overview


---

## Features

- **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.txt`, and `.md` documents with automatic format detection.
- **Smart Vector Caching**: Automatically computes cross-platform vector store paths. Reuses existing HNSWLib vector indexes on disk to eliminate re-indexing overhead on repeat queries.
- **Provider Agnostic**: Configurable Chat LLMs (Google Gemini, OpenAI) and Embeddings models with automatic fallback.
- **Multi-Document Support (`morerag.js`)**: Batch-indexing across entire folders and cross-document querying with error isolation.
- **Zero Side-Effects on Import**: Safe to require in other projects, scripts, or test runners without triggering unprompted LLM calls or disk writes.
- **Production Quality**: Built with strict parameter validation, cross-platform path handling (POSIX & Windows), JSDoc types, and a comprehensive test suite.


## Installation & Setup

### 1. Prerequisites
- **Node.js**: v18.0.0 or later (tested on Node v20 and v24)
- **npm**: v9 or later

### 2. Install Dependencies
```bash
npm install --legacy-peer-deps
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Usage

### 1. Command-Line Interface (CLI)

#### Run Single Document RAG
Query any document directly from the command line:
```bash
node rag.js "files/Designing_Data-Intensive_Applications_TH.pdf" "Compare B-Trees and LSM Trees according to the book."
```
If no arguments are provided, it runs a default query on the sample PDF.

#### Run Diagnostics & Path Inspection
Inspect document extensions, vector index destinations, and detect available documents in `files/`:
```bash
node morerag.js
# Or via npm script:
npm run diagnostics
```

---

### 2. Programmatic API

#### Single Document Query (`rag.js`)
```javascript
const { documentRag } = require('./rag');

async function main() {
    const response = await documentRag('files/document.pdf', 'What are the key findings?', {
        k: 4,                  // retrieve top 4 context chunks
        verbose: true          // enable execution logging
    });

    console.log('Answer:', response.answer);
    console.log(`Execution time: ${response.executionTimeMs}ms`);
    console.log(`Retrieved chunks: ${response.context.length}`);
}

main();
```

#### Pre-Indexing Documents
```javascript
const { loadAndIndexDocument } = require('./rag');

async function indexFile() {
    const result = await loadAndIndexDocument('files/handbook.pdf');
    console.log(`Saved ${result.chunkCount} chunks to: ${result.indexPath}`);
}
```

#### Querying Across Multiple Documents (`morerag.js`)
```javascript
const { queryMultipleDocuments } = require('./morerag');

async function queryAll() {
    const results = await queryMultipleDocuments({
        docPaths: [
            'files/manual_v1.pdf',
            'files/manual_v2.docx'
        ],
        query: 'What is the recommended installation procedure?'
    });

    for (const res of results) {
        if (res.success) {
            console.log(`[${res.docPath}]: ${res.answer}`);
        } else {
            console.error(`[${res.docPath}] Failed: ${res.error}`);
        }
    }
}
```

#### File Utilities (`morerag.js`)
```javascript
const {
    getFileExtension,
    isSupportedFileType,
    getSafeBasename,
    getVectorStorePath,
    listSupportedDocuments
} = require('./morerag');

getFileExtension('report.PDF');             // 'pdf'
isSupportedFileType('notes.docx');          // true
getSafeBasename('files/data.backup.txt');   // 'data.backup'
getVectorStorePath('files/book.pdf');       // 'vector/vector-db/book'
listSupportedDocuments('./files');          // ['files/doc1.pdf', ...]
```

---

## Running Tests

The test suite uses Node.js's native test runner (`node:test` and `node:assert/strict`), ensuring zero test dependency overhead and sub-second execution.

Run all tests:
```bash
npm test
