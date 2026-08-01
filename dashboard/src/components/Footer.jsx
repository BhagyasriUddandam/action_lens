export function Footer({ dataset }) {
  return (
    <footer className="px-6 py-10 text-center text-xs text-[var(--color-ink-muted)]">
      <p>
        {dataset.n_eval} eval clips across {dataset.classes.length} classes &middot; falling drawn from
        URFD ({dataset.falling_sources.urfd}) and HMDB51 ({dataset.falling_sources.hmdb51}) so no class maps
        1:1 to a single source
      </p>
      <p className="mt-1">ActionLens &mdash; a pretrained-baseline vs. VLM benchmark</p>
    </footer>
  )
}
