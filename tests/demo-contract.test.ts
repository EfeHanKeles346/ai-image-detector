import assert from 'node:assert/strict';
import test from 'node:test';
import { E92_SHA, demoFileError, parseDemoAnalysis, OUTCOME_COPY } from '../app/demo-contract.ts';

const response = {schema_version: 2, model_id: 'E92', guard_id: 'e92-stability-v1', artifact_sha256: E92_SHA,
  research_only: true, display_policy: 'e92-preserve-alerts-v1', guard_outcome: 'uncertain', review_required: true, outcome: 'uncertain', reason: 'inconsistent_or_borderline', width: 1024, height: 768};

test('only the exact E92 response and a consistent outcome reach the page', () => {
  assert.deepEqual(parseDemoAnalysis(response), response);
  for (const change of [{model_id: 'E32'}, {artifact_sha256: 'a'.repeat(64)}, {research_only: false},
    {outcome: 'real'}, {guard_outcome: 'ai_signal'}, {review_required: false}, {width: 0}, {width: NaN}, {height: 1000000}]) {
    assert.throws(() => parseDemoAnalysis({...response, ...change}));
  }
  assert.throws(() => parseDemoAnalysis({p_ai: .99, verdict: 'real'}));
});

test('file checks reject oversized, empty and unsupported uploads', () => {
  assert.equal(demoFileError({type: 'image/jpeg', size: 12 * 1024 * 1024}), null);
  assert.match(demoFileError({type: 'image/jpeg', size: 12 * 1024 * 1024 + 1})!, /büyük/);
  assert.match(demoFileError({type: 'image/png', size: 0})!, /boş/);
  assert.match(demoFileError({type: 'image/gif', size: 200})!, /desteklenmiyor/);
});

test('negative copy never certifies authenticity and no percentage is rendered', () => {
  assert.match(OUTCOME_COPY.no_clear_signal.text, /gerçek olduğunu kanıtlamaz/);
  assert.match(OUTCOME_COPY.ai_signal.text, /kesin kanıt değil/);
  assert.doesNotMatch(JSON.stringify(OUTCOME_COPY), /%|olasılık|FPR|recall/);
});

test('an unstable original AI alert stays visible with its review warning', () => {
  const result = parseDemoAnalysis({...response, outcome: 'ai_signal'});
  assert.equal(result.outcome, 'ai_signal');
  assert.equal(result.review_required, true);
});

test('24MP phone responses are admitted while responses above32MP are rejected', () => {
  assert.equal(parseDemoAnalysis({...response, width: 5712, height: 4284}).width, 5712);
  assert.throws(() => parseDemoAnalysis({...response, width: 8001, height: 4000}));
});
