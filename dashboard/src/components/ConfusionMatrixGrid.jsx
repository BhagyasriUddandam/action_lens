import { Fragment } from 'react'
import { sequentialColor, COLORS } from '../lib/theme'

const abbr = (label) => label.slice(0, 4).charAt(0).toUpperCase() + label.slice(1, 4)

// One confusion matrix, rendered as a plain div grid (rows = true label,
// columns = predicted). Cell color comes from the shared sequential-blue
// ramp, scaled to this matrix's own max so baseline and VLM are each
// readable on their own terms.
export function ConfusionMatrixGrid({ title, labels, matrix, accentColor }) {
  const vmax = Math.max(...matrix.flat(), 1)

  return (
    <div>
      <h4 className="mb-3 text-center text-sm font-semibold" style={{ color: accentColor }}>
        {title}
      </h4>
      <div className="inline-grid" style={{ gridTemplateColumns: `2.5rem repeat(${labels.length}, 2.75rem)` }}>
        <div />
        {labels.map((l) => (
          <div key={l} className="pb-1 text-center text-[11px] font-medium text-[var(--color-ink-muted)]">
            {abbr(l)}
          </div>
        ))}
        {matrix.map((row, i) => (
          <Fragment key={labels[i]}>
            <div className="flex items-center justify-end pr-2 text-[11px] font-medium text-[var(--color-ink-muted)]">
              {abbr(labels[i])}
            </div>
            {row.map((count, j) => {
              const bg = sequentialColor(count, vmax)
              const textDark = count < vmax * 0.6
              return (
                <div
                  key={`${labels[i]}-${labels[j]}`}
                  className="m-[1px] flex h-11 w-11 items-center justify-center rounded text-sm font-semibold tabular-nums"
                  style={{ backgroundColor: bg, color: textDark ? COLORS.ink : '#fff' }}
                >
                  {count}
                </div>
              )
            })}
          </Fragment>
        ))}
      </div>
    </div>
  )
}
