import { useLayoutEffect, useRef } from 'react'
import gsap from 'gsap'
import { COLORS } from '../lib/theme'
import { useInView } from '../hooks/useInView'

const MAX_HEIGHT = 200 // px, represents 100%

function Bar({ pct, color, barRef }) {
  return (
    <div className="flex h-[200px] w-9 items-end">
      <div
        ref={barRef}
        className="w-full rounded-t"
        style={{ backgroundColor: color, height: 0 }}
        data-target={(pct * MAX_HEIGHT).toFixed(1)}
      />
    </div>
  )
}

function Legend() {
  return (
    <div className="mb-8 flex justify-center gap-6 text-sm text-[var(--color-ink-secondary)]">
      <span className="flex items-center gap-2">
        <span className="h-3 w-3 rounded" style={{ backgroundColor: COLORS.baseline }} />
        Baseline
      </span>
      <span className="flex items-center gap-2">
        <span className="h-3 w-3 rounded" style={{ backgroundColor: COLORS.vlm }} />
        VLM
      </span>
    </div>
  )
}

// Custom-built grouped bar chart (no charting library) so the mark specs and
// colors match the dataviz skill's rules exactly: thin bars, rounded top
// ends, a legend for the two series, direct value labels, no dual axis.
export function AccuracyChart({ classes, baselinePerClass, vlmPerClass }) {
  const [containerRef, inView] = useInView(0.3)
  const barRefs = useRef([])
  barRefs.current = []

  const addBarRef = (el) => {
    if (el) barRefs.current.push(el)
  }

  useLayoutEffect(() => {
    if (!inView) return
    const ctx = gsap.context(() => {
      gsap.to(barRefs.current, {
        height: (i, el) => `${el.dataset.target}px`,
        duration: 0.9,
        ease: 'power2.out',
        stagger: 0.06,
      })
    })
    return () => ctx.revert()
  }, [inView])

  return (
    <div ref={containerRef}>
      <Legend />
      <div className="flex justify-center gap-10">
        {classes.map((cls) => {
          const b = baselinePerClass[cls].accuracy
          const v = vlmPerClass[cls].accuracy
          return (
            <div key={cls} className="flex flex-col items-center gap-2">
              <div className="flex items-end gap-1.5 text-xs font-semibold tabular-nums text-[var(--color-ink-secondary)]">
                <span style={{ color: COLORS.baseline }}>{Math.round(b * 100)}%</span>
                <span className="text-[var(--color-ink-muted)]">/</span>
                <span style={{ color: COLORS.vlm }}>{Math.round(v * 100)}%</span>
              </div>
              <div className="flex items-end gap-1.5">
                <Bar pct={b} color={COLORS.baseline} barRef={addBarRef} />
                <Bar pct={v} color={COLORS.vlm} barRef={addBarRef} />
              </div>
              <span className="text-sm font-medium text-[var(--color-ink)] capitalize">{cls}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
