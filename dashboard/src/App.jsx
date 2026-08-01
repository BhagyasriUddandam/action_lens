import { useResults } from './hooks/useResults'
import { Hero } from './components/Hero'
import { InsightBanner } from './components/InsightBanner'
import { SectionHeading } from './components/SectionHeading'
import { AccuracyChart } from './components/AccuracyChart'
import { ConfusionMatrixPair } from './components/ConfusionMatrixPair'
import { LatencyCostPanel } from './components/LatencyCostPanel'
import { FailureGallery } from './components/FailureGallery'
import { Footer } from './components/Footer'

function Centered({ children }) {
  return <div className="flex min-h-screen items-center justify-center text-[var(--color-ink-secondary)]">{children}</div>
}

export default function App() {
  const { data, error } = useResults()

  if (error) {
    return (
      <Centered>
        <div className="text-center">
          <p className="font-semibold text-[color:var(--color-critical)]">Couldn&rsquo;t load results.json</p>
          <p className="mt-1 text-sm">{error}</p>
          <p className="mt-3 text-sm">
            Run <code className="rounded bg-black/5 px-1.5 py-0.5">bun run sync</code> after{' '}
            <code className="rounded bg-black/5 px-1.5 py-0.5">python src/evaluate.py</code>.
          </p>
        </div>
      </Centered>
    )
  }

  if (!data) return <Centered>Loading&hellip;</Centered>

  const { dataset, approaches, comparison, failure_cases } = data

  return (
    <main className="mx-auto max-w-5xl">
      <Hero
        baselineAccuracy={approaches.baseline.overall_accuracy * 100}
        vlmAccuracy={approaches.vlm.overall_accuracy * 100}
      />

      <section className="px-6 pb-24">
        <InsightBanner sitStand={comparison.sit_stand} />
      </section>

      <div className="mx-6 border-t border-black/[0.06]" />

      <section className="px-6 py-28">
        <SectionHeading eyebrow="Per class" title="Where each approach wins and loses" />
        <AccuracyChart
          classes={dataset.classes}
          baselinePerClass={approaches.baseline.per_class}
          vlmPerClass={approaches.vlm.per_class}
        />
      </section>

      <div className="mx-6 border-t border-black/[0.06]" />

      <section className="px-6 py-28">
        <SectionHeading
          eyebrow="Confusion matrices"
          title="What each approach confuses"
          subtitle="Rows are the true label, columns the prediction. Off-diagonal mass is error."
        />
        <ConfusionMatrixPair baseline={approaches.baseline.confusion_matrix} vlm={approaches.vlm.confusion_matrix} />
      </section>

      <div className="mx-6 border-t border-black/[0.06]" />

      <section className="px-6 py-28">
        <SectionHeading eyebrow="Tradeoff" title="Speed and cost, not just accuracy" />
        <LatencyCostPanel baseline={approaches.baseline} vlm={approaches.vlm} />
      </section>

      <div className="mx-6 border-t border-black/[0.06]" />

      <section className="px-6 py-28">
        <SectionHeading eyebrow="Failure cases" title="Where the two approaches disagree" />
        <FailureGallery
          cases={failure_cases}
          bucketCounts={comparison.failure_bucket_counts}
          total={comparison.n_failures_total}
        />
      </section>

      <div className="mx-6 border-t border-black/[0.06]" />

      <Footer dataset={dataset} />
    </main>
  )
}
