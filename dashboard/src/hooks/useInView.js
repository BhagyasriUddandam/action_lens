import { useEffect, useRef, useState } from 'react'

// Fires once when the element first scrolls into view. Backs every
// scroll-triggered reveal in the dashboard so animations play as the CEO
// scrolls through the story, not all at once on load.
export function useInView(threshold = 0.3) {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true)
          observer.disconnect()
        }
      },
      { threshold }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [threshold])

  return [ref, inView]
}
