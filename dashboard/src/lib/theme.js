// Single source of truth for chart colors in JS contexts (canvas-free divs,
// GSAP tweens). Mirrors the CSS custom properties in index.css.
//
// Palette matches C10 Labs' visual identity (c10labs.com, observed 2026-08-01):
// one accent color (crimson) on a near-white/near-black base, used sparingly.
// Rather than two saturated series hues, this follows their own convention for
// "compare two numbers, one is the standout" -- seen in their stat row, where
// every number is near-black except the one they want you to notice, which is
// crimson. Baseline (the old way) = ink. VLM (the finding) = accent.
export const COLORS = {
  baseline: '#141414', // ink -- the neutral, "standard approach" identity
  vlm: '#e11d48', // crimson accent -- the standout finding
  good: '#0ca30c',
  critical: '#b91c1c', // distinct hue-lean from the crimson accent, so "wrong" never reads as "VLM"
  ink: '#141414',
  inkSecondary: '#52525b',
  inkMuted: '#8a8a8a',

  // grayscale sequential ramp for confusion-matrix magnitude -- kept quiet/
  // restrained on purpose; crimson stays reserved for the accent role, not
  // spent on heatmap fill.
  seq: ['#fafafa', '#e3e3e3', '#c2c2c2', '#8a8a8a', '#4a4a4a', '#141414'],
}

export function sequentialColor(value, max) {
  if (max <= 0) return COLORS.seq[0]
  const t = Math.max(0, Math.min(1, value / max))
  const idx = Math.round(t * (COLORS.seq.length - 1))
  return COLORS.seq[idx]
}
