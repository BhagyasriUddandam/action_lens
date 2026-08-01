import { FailureCard } from './FailureCard'
import { COLORS } from '../lib/theme'

const BUCKET_META = {
  vlm_correct_baseline_wrong: {
    title: 'VLM caught what the baseline missed',
    color: COLORS.good,
  },
  both_wrong: {
    title: 'Genuinely hard clips',
    color: COLORS.inkMuted,
  },
  baseline_correct_vlm_wrong: {
    title: 'Baseline held its ground',
    color: COLORS.baseline,
  },
}
const BUCKET_ORDER = ['vlm_correct_baseline_wrong', 'both_wrong', 'baseline_correct_vlm_wrong']

export function FailureGallery({ cases, bucketCounts, total }) {
  const byBucket = BUCKET_ORDER.map((key) => ({
    key,
    meta: BUCKET_META[key],
    count: bucketCounts[key] ?? 0,
    examples: cases.filter((c) => c.bucket === key),
  })).filter((b) => b.examples.length > 0)

  return (
    <div>
      <p className="mb-8 text-center text-sm text-[var(--color-ink-secondary)]">
        {total} of 128 eval clips had at least one disagreement between the two approaches.
        A few examples from each pattern:
      </p>
      <div className="space-y-12">
        {byBucket.map(({ key, meta, count, examples }) => (
          <div key={key}>
            <h4 className="mb-4 text-center text-base font-semibold" style={{ color: meta.color }}>
              {meta.title} <span className="font-normal text-[var(--color-ink-muted)]">({count} clips)</span>
            </h4>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              {examples.map((c) => (
                <FailureCard
                  key={c.clip_id}
                  clipId={c.clip_id}
                  label={c.label}
                  source={c.source}
                  baselinePred={c.baseline_pred}
                  vlmPred={c.vlm_pred}
                  evidence={c.vlm_evidence}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
