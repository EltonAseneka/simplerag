/**
 * @fileoverview Test suite setup script.
 * 
 * Ensures all tests run in a strictly offline, isolated, and deterministic environment
 * by stripping live API keys, setting test environment flags, and silencing extraneous logging.
 */

// Force test environment
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'silent';

// Explicitly delete live provider keys to prevent accidental network calls during test runs
delete process.env.OPENAI_API_KEY;
delete process.env.GOOGLE_API_KEY;
delete process.env.GEMINI_API_KEY;

// Log offline confirmation if debugging tests
if (process.env.DEBUG_TESTS) {
    process.stdout.write('[TEST SETUP] Isolated offline environment initialized. All external API keys stripped.\n');
}
