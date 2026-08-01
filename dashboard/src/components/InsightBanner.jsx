import { useInView } from '../hooks/useInView'
import { COLORS } from '../lib/theme'

function BeforeAfterRow({ label, before, after }) {
  const gap = Math.round((after - before) * 100)
  return (
    <div className="flex items-center justify-between gap-4 py-3 first:pt-0 last:pb-0">
      <span className="w-24 shrink-0 font-medium text-[var(--color-ink)]">{label}</span>
      <div className="flex flex-1 items-center gap-3">
        <span className="tabular-nums" style={{ color: COLORS.baseline }}>
          {(before * 100).toFixed(1)}%
        </span>
        <span className="text-[var(--color-ink-muted)]">&rarr;</span>
        <span className="text-lg font-semibold tabular-nums" style={{ color: COLORS.vlm }}>
          {(after * 100).toFixed(1)}%
        </span>
      </div>
      <span
        className="shrink-0 rounded-full px-2.5 py-0.5 text-sm font-semibold"
        style={{ backgroundColor: `${COLORS.good}1a`, color: COLORS.good }}
      >
        +{gap}pts
      </span>
    </div>
  )
}

// The headline finding: the frozen baseline collapses sitting/standing into
// near-random guesses (they're near time-reverses of each other), and the
// VLM separates them cleanly by reasoning about direction of motion. Every
// number here is read from results.json, not hardcoded -- a rerun with a
// different prompt or dataset changes this banner automatically.
export function InsightBanner({ sitStand }) {
  const [ref, inView] = useInView(0.4)

  return (
    <div
      ref={ref}
      className={`mx-auto max-w-2xl rounded-2xl border border-[var(--color-vlm)]/20 bg-[var(--color-vlm)]/5 p-8 transition-all duration-700 ${
        inView ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
      }`}
    >
      <p className="mb-1 text-sm font-semibold tracking-wide text-[var(--color-vlm)] uppercase">
        The headline finding
      </p>
      <h3 className="mb-4 text-xl font-semibold text-[var(--color-ink)]">
        The <span style={{ color: COLORS.vlm }}>VLM</span> separates sitting from standing &mdash; the
        baseline can&rsquo;t
      </h3>
      <p className="mb-5 text-sm text-[var(--color-ink-secondary)]">
        Sitting down and standing up are near time-reverses of each other. A frozen
        vision backbone has no way to encode which direction the motion runs in, so it
        guesses close to a coin flip. Claude reads the frames in order and reasons about
        direction &mdash; and separates both classes cleanly.
      </p>
      <div className="divide-y divide-[var(--color-vlm)]/15">
        <BeforeAfterRow label="Sitting" before={sitStand.baseline_sitting_acc} after={sitStand.vlm_sitting_acc} />
        <BeforeAfterRow label="Standing" before={sitStand.baseline_standing_acc} after={sitStand.vlm_standing_acc} />
      </div>
    </div>
  )
}
