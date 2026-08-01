export function SectionHeading({ eyebrow, title, subtitle }) {
  return (
    <div className="mb-14 text-center">
      {eyebrow && (
        <p className="mb-3 flex items-center justify-center gap-3 text-sm font-semibold tracking-wide text-[var(--color-ink-muted)] uppercase">
          <span className="h-px w-6 bg-[var(--color-ink-muted)]/40" />
          {eyebrow}
        </p>
      )}
      <h2 className="text-3xl font-semibold text-[var(--color-ink)] sm:text-4xl">{title}</h2>
      {subtitle && (
        <p className="mx-auto mt-3 max-w-2xl text-[var(--color-ink-secondary)]">{subtitle}</p>
      )}
    </div>
  )
}
