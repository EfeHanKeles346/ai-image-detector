import { AnalysisResponseError } from './analysis-contract.ts';

export const E92_SHA = '3a68c50d7cabd17d74c90bdcaf3b74aaacbc6c07e0bf28e332b1b91f99c9ef35';
export const AI_CUT = 0.07940196245908739;
const REAL_CUT = 0.011505939625203613;
export type ModelScore = {
  kind: 'raw_e92_score';
  calibrated: false;
  original: number;
  social_q75: number;
  ai_cut: typeof AI_CUT;
};

export function scorePercent(score: number): string {
  if (!Number.isFinite(score) || score < 0 || score > 1) throw new AnalysisResponseError();
  return new Intl.NumberFormat('tr-TR', {style: 'percent', minimumFractionDigits: 2,
    maximumFractionDigits: 2}).format(score);
}

export type DemoAnalysis = {
  schema_version: 3;
  model_id: 'E92';
  guard_id: 'e92-stability-v1';
  artifact_sha256: string;
  research_only: true;
  display_policy: 'e92-preserve-alerts-v1';
  guard_outcome: 'ai_signal' | 'no_clear_signal' | 'uncertain' | 'not_run';
  review_required: boolean;
  outcome: 'ai_signal' | 'no_clear_signal' | 'uncertain';
  reason: 'stable_signal' | 'limited_negative_evidence' | 'inconsistent_or_borderline' | 'image_too_small';
  width: number;
  height: number;
  model_score: ModelScore | null;
};

export function uncertaintyExplanation(analysis: DemoAnalysis): string | null {
  const s = analysis.model_score;
  if (!s || analysis.outcome !== 'uncertain') return null;
  if (s.social_q75 >= AI_CUT) return 'Asıl fotoğraf AI uyarısı vermedi; sıkıştırılmış kopyası verdi. Bu yüzden kesin bir sonuç söyleyemiyoruz.';
  if (Math.max(s.original, s.social_q75) < REAL_CUT) return 'Bu modelin iki puanı da düşük, ancak karşılaştırdığımız önceki model uyarı verdi. Bu görüş ayrılığı nedeniyle karar veremiyoruz.';
  return 'AI uyarı sınırı aşılmadı, ancak puanlar belirgin iz yok demek için yeterince düşük değil. Bu yüzden sonuç belirsiz.';
}

export function parseDemoAnalysis(value: unknown): DemoAnalysis {
  if (!value || typeof value !== 'object') throw new AnalysisResponseError();
  const r = value as Record<string, unknown>;
  const reasonMatches = (r.outcome === 'ai_signal' && ['stable_signal', 'inconsistent_or_borderline'].includes(String(r.reason))) ||
    (r.outcome === 'no_clear_signal' && r.reason === 'limited_negative_evidence') ||
    (r.outcome === 'uncertain' && ['inconsistent_or_borderline', 'image_too_small'].includes(String(r.reason)));
  const s = r.model_score as Record<string, unknown> | null | undefined;
  const bounded = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n) && n >= 0 && n <= 1;
  const scoreMatches = r.reason === 'image_too_small' ? s === null :
    s != null && s.kind === 'raw_e92_score' && s.calibrated === false && s.ai_cut === AI_CUT &&
    bounded(s.original) && bounded(s.social_q75) &&
    (r.outcome === 'ai_signal') === (s.original >= AI_CUT) &&
    (r.guard_outcome === 'ai_signal') === (Math.min(s.original, s.social_q75) >= AI_CUT) &&
    (r.guard_outcome !== 'no_clear_signal' || Math.max(s.original, s.social_q75) < REAL_CUT);
  if (r.schema_version !== 3 || !scoreMatches || r.model_id !== 'E92' || r.guard_id !== 'e92-stability-v1' ||
      r.artifact_sha256 !== E92_SHA || r.research_only !== true || !reasonMatches ||
      r.display_policy !== 'e92-preserve-alerts-v1' ||
      !['ai_signal', 'no_clear_signal', 'uncertain', 'not_run'].includes(String(r.guard_outcome)) ||
      r.review_required !== ['uncertain', 'not_run'].includes(String(r.guard_outcome)) ||
      (r.reason === 'stable_signal' && r.guard_outcome !== 'ai_signal') ||
      (r.reason === 'limited_negative_evidence' && r.guard_outcome !== 'no_clear_signal') ||
      (r.reason === 'inconsistent_or_borderline' && r.guard_outcome !== 'uncertain') ||
      (r.reason === 'image_too_small' && r.guard_outcome !== 'not_run') ||
      !Number.isInteger(r.width) || !Number.isInteger(r.height) ||
      Number(r.width) <= 0 || Number(r.height) <= 0 || Number(r.width) * Number(r.height) > 32_000_000) {
    throw new AnalysisResponseError('Bu yanıt güncel modelle doğrulanamadı. Lütfen yeniden deneyin.');
  }
  return r as DemoAnalysis;
}

export function demoFileError(file: { type: string; size: number }): string | null {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    return 'Bu dosya desteklenmiyor. JPG, PNG veya WEBP seçin.';
  }
  if (!file.size) return 'Bu dosya boş. Başka bir fotoğraf seçin.';
  if (file.size > 12 * 1024 * 1024) return 'Dosya 12 MB sınırından büyük. Daha küçük bir dosya seçin.';
  return null;
}

export const OUTCOME_COPY = {
  ai_signal: {
    title: 'Yapay zekâ izleri bulundu',
    text: 'Model, bu görselde yapay zekâ ile üretilmiş görüntülere benzeyen izler buldu. Bu bir işaret; kesin kanıt değil.',
    next: 'Sonucu değerlendirirken görselin kaynağına ve ilk paylaşıldığı yere de bakın.',
  },
  no_clear_signal: {
    title: 'Belirgin bir iz bulunamadı',
    text: 'Model bu görselde belirgin bir yapay zekâ izi bulamadı. Bu, fotoğrafın gerçek olduğunu kanıtlamaz; bazı üretimler gözden kaçabilir.',
    next: 'Önemli bir karar için yalnız bu sonuca güvenmeyin. Görselin kaynağını da kontrol edin.',
  },
  uncertain: {
    title: 'Karar veremedim',
    text: 'Kontroller yeterince tutarlı bir sonuç vermedi. Yanıltıcı bir cevap vermemek için bu görsel hakkında karar vermiyoruz.',
    next: 'Varsa ekran görüntüsü yerine fotoğrafın asıl dosyasını deneyin.',
  },
} as const;
