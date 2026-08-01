import { COLORS } from '../lib/theme'

// Correct predictions render as plain text -- they're the expected state and
// don't need celebrating. Only a wrong prediction gets a tinted pill, so the
// eye lands on the actual point of interest in each card.
function PredictionChip({ approach, prediction, correct }) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-14 shrink-0 font-medium text-[var(--color-ink-muted)]">{approach}</span>
      {correct ? (
        <span className="font-semibold text-[var(--color-ink)] capitalize">{prediction}</span>
      ) : (
        <span
          className="rounded-full px-2.5 py-0.5 font-semibold capitalize"
          style={{ backgroundColor: `${COLORS.critical}14`, color: COLORS.critical }}
        >
          {prediction}
        </span>
      )}
    </div>
  )
}

// One clip: its first and last frame (the direction-of-motion evidence the
// VLM's prompt is built on), the true label, both predictions, and the VLM's
// own stated reasoning. Flat, borderless card fill -- matches C10's minimal
// feature-card treatment (tinted background, no border, no shadow).
export function FailureCard({ clipId, label, source, baselinePred, vlmPred, evidence }) {
  return (
    <div className="rounded-xl bg-[var(--color-card)] p-4">
      <div className="mb-3 flex gap-1 overflow-hidden rounded-lg">
        <img src={`/clips/${clipId}_00.jpg`} alt="First frame" className="h-28 w-1/2 object-cover" loading="lazy" />
        <img src={`/clips/${clipId}_15.jpg`} alt="Last frame" className="h-28 w-1/2 object-cover" loading="lazy" />
      </div>
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-semibold text-[var(--color-ink)] capitalize">{label}</span>
        <span className="text-xs text-[var(--color-ink-muted)] uppercase">{source}</span>
      </div>
      <div className="space-y-1.5">
        <PredictionChip approach="Baseline" prediction={baselinePred} correct={baselinePred === label} />
        <PredictionChip approach="VLM" prediction={vlmPred} correct={vlmPred === label} />
      </div>
      {evidence && (
        <p className="mt-3 border-t border-black/10 pt-3 text-xs text-[var(--color-ink-secondary)] italic">
          &ldquo;{evidence}&rdquo;
        </p>
      )}
    </div>
  )
}
