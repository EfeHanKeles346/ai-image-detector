import assert from 'node:assert/strict';
import test from 'node:test';
import { E92_SHA, demoFileError, demoResultCopy, parseDemoAnalysis, OUTCOME_COPY, AI_CUT, scorePercent, uncertaintyExplanation } from '../app/demo-contract.ts';

const response = {schema_version: 3, model_id: 'E92', guard_id: 'e92-stability-v1', artifact_sha256: E92_SHA,
  research_only: true, display_policy: 'e92-preserve-alerts-v1', guard_outcome: 'uncertain', review_required: true, outcome: 'uncertain', reason: 'inconsistent_or_borderline', width: 1024, height: 768,
  model_score: {kind: 'raw_e92_score', calibrated: false, original: .02, social_q75: .03, ai_cut: AI_CUT}};

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

test('negative copy never certifies authenticity or claims a probability', () => {
  assert.match(OUTCOME_COPY.no_clear_signal.text, /gerçek olduğunu kanıtlamaz/);
  assert.match(OUTCOME_COPY.ai_signal.text, /kesin kanıt değil/);
  assert.doesNotMatch(JSON.stringify(OUTCOME_COPY), /%|olasılık|FPR|recall/);
});

test('an unstable original AI alert stays visible with its review warning', () => {
  const result = parseDemoAnalysis({...response, outcome: 'ai_signal', model_score: {...response.model_score, original: AI_CUT}});
  assert.equal(result.outcome, 'ai_signal');
  assert.equal(result.review_required, true);
});

test('24MP phone responses are admitted while responses above32MP are rejected', () => {
  assert.equal(parseDemoAnalysis({...response, width: 5712, height: 4284}).width, 5712);
  assert.throws(() => parseDemoAnalysis({...response, width: 8001, height: 4000}));
});


test('raw score presentation rejects uncalibrated claims, missing scores and decision conflicts', () => {
  for (const model_score of [null, undefined, {...response.model_score, calibrated: true},
    {...response.model_score, ai_cut: .5}, {...response.model_score, original: .9},
    {...response.model_score, original: NaN}, {...response.model_score, original: Infinity},
    {...response.model_score, original: -1}, {...response.model_score, original: 1.01},
    {...response.model_score, original: '0.02'}]) {
    assert.throws(() => parseDemoAnalysis({...response, model_score}));
  }
  assert.throws(() => parseDemoAnalysis({...response, schema_version: 2}));
  const small = {...response, reason: 'image_too_small', guard_outcome: 'not_run', model_score: null};
  assert.equal(parseDemoAnalysis(small).model_score, null);
  assert.throws(() => parseDemoAnalysis({...small, model_score: response.model_score}));
  assert.equal(scorePercent(.125), '%12,50');
  assert.equal(scorePercent(AI_CUT), '%7,94');
  assert.equal(scorePercent(0), '%0,00');
  assert.equal(scorePercent(1), '%100,00');
  for (const score of [NaN, Infinity, -1, 1.01]) assert.throws(() => scorePercent(score));
});


test('uncertainty explains transformed crossing, reference veto and borderline separately', () => {
  const result = parseDemoAnalysis(response);
  assert.match(uncertaintyExplanation(result)!, /yeterince düşük değil/);
  assert.match(uncertaintyExplanation({...result, model_score: {...result.model_score!, original: 0, social_q75: 0}})!, /önceki model/);
  assert.match(uncertaintyExplanation({...result, model_score: {...result.model_score!, social_q75: AI_CUT}})!, /kopyası verdi/);
  assert.equal(uncertaintyExplanation({...result, model_score: null}), null);
});

test('the primary result explains the cause without treating low scores as proof', () => {
  const low = parseDemoAnalysis({...response, model_score: {...response.model_score, original: .0001, social_q75: .0039}});
  const before = structuredClone(low);
  assert.equal(demoResultCopy(low).title, 'Sonuç belirsiz');
  assert.match(demoResultCopy(low).text, /önceki model uyarı verdi/);
  assert.match(demoResultCopy(low).next, /tek başına gerçek fotoğraf anlamına gelmez/);
  assert.deepEqual(low, before);
  assert.match(demoResultCopy(parseDemoAnalysis(response)).text, /yeterince düşük değil/);
  const crossing = parseDemoAnalysis({...response, model_score: {...response.model_score, social_q75: AI_CUT}});
  assert.match(demoResultCopy(crossing).text, /kopyası verdi/);
  const negative = parseDemoAnalysis({...low, outcome: 'no_clear_signal', reason: 'limited_negative_evidence', guard_outcome: 'no_clear_signal', review_required: false});
  assert.equal(demoResultCopy(negative), OUTCOME_COPY.no_clear_signal);
  const alert = parseDemoAnalysis({...response, outcome: 'ai_signal', model_score: {...response.model_score, original: AI_CUT}});
  assert.equal(demoResultCopy(alert), OUTCOME_COPY.ai_signal);
  const small = parseDemoAnalysis({...response, reason: 'image_too_small', guard_outcome: 'not_run', model_score: null});
  assert.match(demoResultCopy(small).text, /puanı hesaplanmadı/);
  assert.match(demoResultCopy(small).next, /224 piksel/);
});
