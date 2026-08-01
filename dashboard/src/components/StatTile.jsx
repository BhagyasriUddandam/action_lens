import { useInView } from '../hooks/useInView'
import { useCountUp } from '../hooks/useCountUp'

// A single headline number. The one job every stat in the dashboard shares --
// Hero, LatencyCostPanel, and the failure gallery headings all render through
// this component so a number always looks and animates the same way.
export function StatTile({ label, value, decimals = 0, prefix = '', suffix = '', color, caption }) {
  const [ref, inView] = useInView(0.5)
  const display = useCountUp(value, inView, { decimals })

  return (
    <div ref={ref} className="text-center">
      <div className="text-5xl font-bold tabular-nums" style={{ color }}>
        {prefix}
        {display}
        {suffix}
      </div>
      <p className="mt-2 text-sm font-medium text-[var(--color-ink-secondary)]">{label}</p>
      {caption && <p className="mt-1 text-xs text-[var(--color-ink-muted)]">{caption}</p>}
    </div>
  )
}
