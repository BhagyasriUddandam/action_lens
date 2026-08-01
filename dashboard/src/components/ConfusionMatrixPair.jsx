import { ConfusionMatrixGrid } from './ConfusionMatrixGrid'
import { COLORS } from '../lib/theme'

export function ConfusionMatrixPair({ baseline, vlm }) {
  return (
    <div className="flex flex-wrap justify-center gap-16">
      <ConfusionMatrixGrid
        title="Baseline"
        labels={baseline.labels}
        matrix={baseline.matrix}
        accentColor={COLORS.baseline}
      />
      <ConfusionMatrixGrid title="VLM" labels={vlm.labels} matrix={vlm.matrix} accentColor={COLORS.vlm} />
    </div>
  )
}
