import { useLayoutEffect, useRef } from 'react'
import gsap from 'gsap'
import { COLORS } from '../lib/theme'
import { StatTile } from './StatTile'

export function Hero({ baselineAccuracy, vlmAccuracy }) {
  const rootRef = useRef(null)

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      gsap
        .timeline({ defaults: { ease: 'power3.out' } })
        .from('[data-hero-eyebrow]', { opacity: 0, y: 12, duration: 0.5 })
        .from('[data-hero-title]', { opacity: 0, y: 16, duration: 0.6 }, '-=0.3')
        .from('[data-hero-sub]', { opacity: 0, y: 12, duration: 0.5 }, '-=0.3')
        .from('[data-hero-stats]', { opacity: 0, y: 20, duration: 0.6 }, '-=0.2')
    }, rootRef)
    return () => ctx.revert()
  }, [])

  return (
    <section ref={rootRef} className="mx-auto max-w-4xl px-6 pt-32 pb-20 text-center">
      <p
        data-hero-eyebrow
        className="mb-4 flex items-center justify-center gap-3 text-sm font-semibold tracking-wide text-[var(--color-ink-muted)] uppercase"
      >
        <span className="h-px w-6 bg-[var(--color-ink-muted)]/40" />
        ActionLens
      </p>
      <h1 data-hero-title className="text-4xl font-semibold text-[var(--color-ink)] sm:text-5xl">
        Which video-AI approach should you use?
      </h1>
      <p data-hero-sub className="mx-auto mt-5 max-w-xl text-lg text-[var(--color-ink-secondary)]">
        A pretrained action-recognition model and a vision-language model, benchmarked
        head to head on the same 128 clips.
      </p>

      <div data-hero-stats className="mx-auto mt-16 grid max-w-md grid-cols-2 gap-8">
        <StatTile label="Baseline accuracy" value={baselineAccuracy} decimals={1} suffix="%" color={COLORS.baseline} />
        <StatTile label="VLM accuracy" value={vlmAccuracy} decimals={1} suffix="%" color={COLORS.vlm} />
      </div>
    </section>
  )
}
