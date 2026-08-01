import { useEffect, useRef, useState } from 'react'
import gsap from 'gsap'

// Animates a number from 0 to `target` once `start` becomes true. Used for
// every headline stat so numbers count up as they enter view, rather than
// appearing as static text.
export function useCountUp(target, start, { decimals = 0, duration = 1.2 } = {}) {
  const [value, setValue] = useState(0)
  const obj = useRef({ v: 0 })

  useEffect(() => {
    if (!start) return
    const tween = gsap.to(obj.current, {
      v: target,
      duration,
      ease: 'power2.out',
      onUpdate: () => setValue(obj.current.v),
    })
    return () => tween.kill()
  }, [start, target, duration])

  return value.toFixed(decimals)
}
