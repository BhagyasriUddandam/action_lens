import { useEffect, useState } from 'react'

// The dashboard's only data source. Loaded once at startup; no other state
// management is needed for a static, read-only view.
export function useResults() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/results.json')
      .then((res) => {
        if (!res.ok) throw new Error(`results.json responded ${res.status}`)
        return res.json()
      })
      .then(setData)
      .catch((err) => setError(err.message))
  }, [])

  return { data, error }
}
