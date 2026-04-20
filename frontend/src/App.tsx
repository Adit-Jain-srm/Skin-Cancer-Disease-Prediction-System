import { useCallback, useEffect, useState } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { Header } from './components/Header'
import { UploadCard } from './components/UploadCard'
import { ResultsCard } from './components/ResultsCard'
import { HistorySection } from './components/HistorySection'
import { StatsSection } from './components/StatsSection'
import { labelForClass } from './lib/classNames'
import type { DeviceConfig, HistoryEntry, PredictResponse, SessionStats } from './lib/types'

const initialStats: SessionStats = {
  totalAnalyses: 0,
  totalConfidence: 0,
  totalTime: 0,
}

function App() {
  const reduce = useReducedMotion()
  const [deviceConfig, setDeviceConfig] = useState<DeviceConfig | null>(null)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [historyVersion, setHistoryVersion] = useState(0)
  const [activeHistoryIndex, setActiveHistoryIndex] = useState<number | null>(null)
  const [resultAnimKey, setResultAnimKey] = useState(0)
  const [sessionStats, setSessionStats] = useState<SessionStats>(initialStats)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const r = await fetch('/api/config')
        if (!r.ok) throw new Error('config')
        const data = (await r.json()) as DeviceConfig
        if (!cancelled) setDeviceConfig(data)
      } catch {
        if (!cancelled) setDeviceConfig({ gpu_available: false, device: 'Offline', api_url: '/api' })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!success) return
    const t = window.setTimeout(() => setSuccess(''), 3000)
    return () => window.clearTimeout(t)
  }, [success])

  const onImageLoad = useCallback((dataUrl: string) => {
    setCurrentImage(dataUrl)
    setError('')
    setSuccess('')
    setResult(null)
    setActiveHistoryIndex(null)
  }, [])

  const onClear = useCallback(() => {
    setCurrentImage(null)
    setResult(null)
    setError('')
    setSuccess('')
    setActiveHistoryIndex(null)
  }, [])

  const onAnalyze = useCallback(async () => {
    if (!currentImage) return
    setLoading(true)
    setError('')
    setSuccess('')
    setResult(null)

    try {
      const t0 = performance.now()
      const imgRes = await fetch(currentImage)
      const blob = await imgRes.blob()
      const formData = new FormData()
      formData.append('image', blob, 'upload.jpg')

      const apiResponse = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      })

      if (!apiResponse.ok) {
        throw new Error(`API error: ${apiResponse.status}`)
      }

      const data = (await apiResponse.json()) as PredictResponse
      const elapsed = Math.round(performance.now() - t0)

      if (!data.success && 'error' in data) {
        throw new Error(String((data as { error?: string }).error ?? 'Unknown error'))
      }

      setResult(data)
      setResultAnimKey((k) => k + 1)

      const p = data.prediction.prediction
      const entry: HistoryEntry = {
        image: currentImage,
        classCode: p.class,
        className: labelForClass(p.class),
        confidence: p.confidence,
      }
      setHistory((h) => {
        const next = [entry, ...h].slice(0, 10)
        return next
      })
      setHistoryVersion((v) => v + 1)
      setActiveHistoryIndex(0)

      setSessionStats((s) => ({
        totalAnalyses: s.totalAnalyses + 1,
        totalConfidence: s.totalConfidence + p.confidence,
        totalTime: s.totalTime + elapsed,
      }))

      setSuccess(`Analysis complete (${elapsed}ms)`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [currentImage])

  const onHistorySelect = useCallback((index: number) => {
    const item = history[index]
    if (!item) return
    setCurrentImage(item.image)
    setActiveHistoryIndex(index)
  }, [history])

  return (
    <>
      <Header config={deviceConfig} />
      <div className="container">
        <motion.h1
          className="page-title"
          initial={reduce ? false : { opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduce ? 0 : 0.55, ease: [0.25, 0.1, 0.25, 1] }}
        >
          Skin lesion analysis
        </motion.h1>

        <div className="main-grid">
          <UploadCard
            currentImage={currentImage}
            loading={loading}
            error={error}
            success={success}
            onImageLoad={onImageLoad}
            onClear={onClear}
            onAnalyze={onAnalyze}
          />
          <ResultsCard result={result} loading={loading} animationKey={resultAnimKey} />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
          <HistorySection
            history={history}
            activeIndex={activeHistoryIndex}
            onSelect={onHistorySelect}
            listKey={historyVersion}
          />
          <StatsSection stats={sessionStats} />
        </div>
      </div>

      <footer className="site-footer">
        <p style={{ margin: '0 0 0.5rem' }}>
          <strong>Disclaimer:</strong> This AI tool is for educational purposes only. Always consult a
          dermatologist for medical advice.
        </p>
        <p style={{ margin: 0 }}>Skin Cancer Detection System · Powered by deep learning</p>
      </footer>
    </>
  )
}

export default App
