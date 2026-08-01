import { COLORS } from '../lib/theme'
import { StatTile } from './StatTile'

// Deliberately does NOT show a fabricated "$0" baseline cost bar next to the
// VLM's per-clip price -- baseline runs on owned GPU compute, so there's no
// honest per-call dollar figure to compare against a metered API. See the
// caption instead of a misleading side-by-side number.
export function LatencyCostPanel({ baseline, vlm }) {
  return (
    <div>
      <div className="grid grid-cols-1 gap-10 sm:grid-cols-3">
        <StatTile
          label="Baseline latency"
          value={baseline.latency_ms_per_clip}
          decimals={0}
          suffix=" ms"
          color={COLORS.baseline}
          caption="per clip, GPU"
        />
        <StatTile
          label="VLM latency"
          value={vlm.latency_ms_per_clip}
          decimals={0}
          suffix=" ms"
          color={COLORS.vlm}
          caption="per clip, API call"
        />
        <StatTile
          label="VLM cost"
          value={vlm.cost_per_1k_clips_usd}
          decimals={2}
          prefix="$"
          color={COLORS.vlm}
          caption="per 1,000 clips"
        />
      </div>
      <p className="mx-auto mt-8 max-w-lg text-center text-sm text-[var(--color-ink-muted)]">
        {baseline.cost_note}
      </p>
    </div>
  )
}
