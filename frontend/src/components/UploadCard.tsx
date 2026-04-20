import { useCallback, useRef, type ChangeEvent, type DragEvent } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { GlassCard } from './ui/GlassCard'
import { Button } from './ui/Button'
import { Spinner } from './ui/Spinner'

type UploadCardProps = {
  currentImage: string | null
  loading: boolean
  error: string
  success: string
  onImageLoad: (dataUrl: string) => void
  onClear: () => void
  onAnalyze: () => void
}

export function UploadCard({
  currentImage,
  loading,
  error,
  success,
  onImageLoad,
  onClear,
  onAnalyze,
}: UploadCardProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const reduce = useReducedMotion()

  const openPicker = () => inputRef.current?.click()

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file || !file.type.startsWith('image/')) {
        return
      }
      const reader = new FileReader()
      reader.onload = (e) => {
        const url = e.target?.result
        if (typeof url === 'string') onImageLoad(url)
      }
      reader.readAsDataURL(file)
    },
    [onImageLoad],
  )

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    handleFile(file)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    handleFile(file)
  }

  return (
    <GlassCard
      className="inner-card-padding"
      initial={reduce ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduce ? 0 : 0.5, delay: reduce ? 0 : 0.08 }}
    >
      <h2>Upload image</h2>
      <div className="info-strip" style={{ marginBottom: '1.25rem' }}>
        <strong>Tip:</strong> Use a clear photo of the lesion. This tool is for education only —
        always see a dermatologist for medical advice.
      </div>

      <motion.div
        className="upload-zone"
        onClick={openPicker}
        onDragOver={(e) => {
          e.preventDefault()
          e.currentTarget.classList.add('drag-active')
        }}
        onDragLeave={(e) => e.currentTarget.classList.remove('drag-active')}
        onDrop={(e) => {
          e.currentTarget.classList.remove('drag-active')
          onDrop(e)
        }}
        whileHover={reduce ? undefined : { scale: 1.01 }}
        whileTap={reduce ? undefined : { scale: 0.99 }}
        style={{ opacity: currentImage ? 0.65 : 1 }}
      >
        <div style={{ fontSize: '2.25rem', marginBottom: '0.5rem' }} aria-hidden>
          📁
        </div>
        <h3 style={{ fontFamily: 'var(--font-display)', color: 'var(--accent-strong)', margin: '0 0 0.35rem' }}>
          Click or drag to upload
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', margin: 0 }}>
          JPG, PNG, or BMP — up to 50MB
        </p>
      </motion.div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={onChange}
      />

      {currentImage ? (
        <img src={currentImage} alt="Selected lesion preview" className="image-preview" />
      ) : null}

      <div
        className={`alert alert-error ${error ? 'visible' : ''}`.trim()}
        role="alert"
      >
        {error}
      </div>
      <div
        className={`alert alert-success ${success ? 'visible' : ''}`.trim()}
        role="status"
      >
        {success}
      </div>

      {loading ? (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.75rem',
            margin: '1.25rem 0',
            color: 'var(--accent-strong)',
          }}
        >
          <Spinner size={26} />
          <span>Analyzing…</span>
        </div>
      ) : null}

      {currentImage ? (
        <>
          <Button
            variant="primary"
            loading={loading}
            onClick={onAnalyze}
            style={{ marginBottom: '0.75rem' }}
          >
            {loading ? 'Analyzing…' : 'Analyze image'}
          </Button>
          <Button variant="secondary" onClick={onClear} disabled={loading}>
            Clear
          </Button>
        </>
      ) : null}
    </GlassCard>
  )
}
