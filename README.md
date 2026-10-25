# SimpleRAG: Production-Ready Document Retrieval-Augmented Generation

A modular, robust, and well-tested Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Google Gemini**, **OpenAI Embeddings**, and **HNSWLib** on Node.js.

[![CI](https://github.com/owner/simplerag/actions/workflows/ci.yml/badge.svg)](https://github.com/owner/simplerag/actions/workflows/ci.yml)
[![Node.js Version](https://img.shields.io/badge/node-%3E%3D20.0.0-brightgreen.svg)](https://nodejs.org/)
[![License: ISC](https://img.shields.io/badge/License-ISC-blue.svg)](https://opensource.org/licenses/ISC)
[![Coverage](https://img.shields.io/badge/coverage-89.8%25-brightgreen.svg)](https://github.com/owner/simplerag)

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Target Document (.pdf,    |
                      |   .docx, .txt, .md)         |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | getLoaderForDocument()      |
                      | PDFLoader / Docx / Text     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | loadAndSplitDocument()      |
                      | RecursiveCharacterSplitter  |
                      +--------------+--------------+
                                     |
               +---------------------+---------------------+
               |                                           |
     (Cached on Disk?)                           (First Time Indexing)
               |                                           |
               v                                           v
      +-----------------+                         +-----------------+
      | HNSWLib.load()  |                         | HNSWLib.fromDocs|
      | from ./vector/  |                         | & save to disk  |
      +--------+--------+                         +--------+--------+
               |                                           |
               +---------------------+---------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   HNSWLib Vector Retriever  |
                      +--------------+--------------+
                                     | (Top-k context chunks)
                                     v
                      +-----------------------------+
                      | createStuffDocumentsChain   |
                      | ChatGoogleGenerativeAI (LLM)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       Synthesized Answer    |
                      |     with Source Context     |
                      +-----------------------------+
```

---

## Features

- **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.txt`, and `.md` documents with automatic format detection.
- **Smart Vector Caching**: Automatically computes cross-platform vector store paths. Reuses existing HNSWLib vector indexes on disk to eliminate re-indexing overhead on repeat queries.
- **Provider Agnostic**: Configurable Chat LLMs (Google Gemini, OpenAI) and Embeddings models with automatic fallback.
- **Multi-Document Support (`morerag.js`)**: Batch-indexing across entire folders and cross-document querying with error isolation.
- **Typed Error Hierarchy (`lib/errors.js`)**: Domain-specific error classes with machine-readable error codes (`DOCUMENT_NOT_FOUND`, `UNSUPPORTED_FORMAT`, `EMBEDDING_PROVIDER_ERROR`, `LLM_PROVIDER_ERROR`, `VECTOR_STORE_ERROR`, `VALIDATION_ERROR`).
- **Structured Logging (`lib/logger.js`)**: Structured logger supporting `DEBUG`, `INFO`, `WARN`, `ERROR` levels and JSON/human formats.
- **Zero Side-Effects on Import**: Safe to require in other projects, scripts, or test runners without triggering unprompted LLM calls or disk writes.
- **100% Offline Test Isolation (`tests/setup.js`)**: Tests run completely offline with mocked embeddings and LLMs, zero API key requirements, and zero token costs.
- **High Test Coverage (89.8% Lines)**: Enforced via `c8` threshold checks gating CI.
- **CI/CD Ready**: Pre-configured GitHub Actions workflow (`.github/workflows/ci.yml`) and Dependabot updates (`.github/dependabot.yml`).

---

## Project Structure

```
simplerag/
├── .github/
│   ├── workflows/
│   │   └── ci.yml               # Automated CI pipeline (Node 20, lint, tests, coverage)
│   └── dependabot.yml           # Automated weekly dependency updates
├── files/                       # Sample documents (.pdf, .docx, etc.)
├── lib/
│   ├── errors.js                # Custom typed error hierarchy
│   └── logger.js                # Structured logger (JSON & text formats)
├── vector/                      # Persisted HNSWLib vector indices
│   └── vector-db/
├── tests/                       # Comprehensive test suites
│   ├── setup.js                 # Offline deterministic test environment setup
│   ├── errors.test.js           # Tests for error hierarchy and logger
│   ├── morerag.test.js          # Unit tests for path & multi-doc utils
│   └── rag.test.js              # Unit and integration tests for RAG pipeline
├── .env.example                 # Environment variables template
├── eslint.config.js             # ESLint recommended configuration
├── morerag.js                   # Document path utilities & multi-doc operations
├── package.json                 # Project dependencies, devDependencies, and npm scripts
├── rag.js                       # Core RAG pipeline & CLI runner
└── README.md                    # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
- **Node.js**: v18.0.0 or later (v20+ recommended)
- **npm**: v9 or later

### 2. Fresh Clone Installation
To ensure reproducible installs matching the lockfile:
```bash
npm ci --legacy-peer-deps
# Or if installing from scratch:
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
            console.error(`[${res.docPath}] Failed (${res.errorCode}): ${res.error}`);
        }
    }
}
```

---

## Quality, Testing & CI/CD

### 1. Run Unit & Integration Tests (Offline)
Tests run using Node.js's native test runner (`node:test` and `node:assert/strict`) via `tests/setup.js`. No network access or API keys are required:
```bash
npm test
```

### 2. Run Test Coverage Reporting
Generate statement, function, and branch coverage reports:
```bash
npm run test:coverage
```

### 3. Enforce Coverage Thresholds
Enforces **>=70% lines and functions, >=65% branches** (fails build if coverage drops):
```bash
npm run check-coverage
```

### 4. Code Quality & Linting
Run ESLint:
```bash
npm run lint
# Auto-fix code style issues:
npm run lint:fix
```

### 5. Automated CI Pipeline
On every push and pull request, `.github/workflows/ci.yml` executes:
1. Fresh dependency installation (`npm ci --legacy-peer-deps`)
2. Lint check (`npm run lint`)
3. Unit & integration test execution (`npm test`)
4. Coverage threshold verification (`npm run check-coverage`)
