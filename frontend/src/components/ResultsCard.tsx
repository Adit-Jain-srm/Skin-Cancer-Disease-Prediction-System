import { useMemo } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import { GlassCard } from './ui/GlassCard'
import { StaggerItem, StaggerList } from './ui/StaggerList'
import { Skeleton } from './ui/Skeleton'
import { labelForClass } from '../lib/classNames'
import type { PredictResponse } from '../lib/types'

type ResultsCardProps = {
  result: PredictResponse | null
  loading: boolean
  animationKey: number
}

export function ResultsCard({ result, loading, animationKey }: ResultsCardProps) {
  const reduce = useReducedMotion()

  const sortedClasses = useMemo(() => {
    if (!result?.prediction?.probabilities) return []
    return Object.entries(result.prediction.probabilities).sort((a, b) => b[1] - a[1])
  }, [result])

  const pred = result?.prediction?.prediction

  return (
    <GlassCard
      className="inner-card-padding"
      initial={reduce ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduce ? 0 : 0.5, delay: reduce ? 0 : 0.14 }}
    >
      <h2>Analysis results</h2>

      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <Skeleton style={{ height: '4.5rem', width: '100%' }} />
          <Skeleton style={{ height: '12px', width: '40%' }} />
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} style={{ height: '2.75rem', width: '100%' }} />
          ))}
        </div>
      ) : null}

      {!loading && result && pred ? (
        <motion.div
          key={animationKey}
          initial={reduce ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: reduce ? 0 : 0.35 }}
        >
          <div
            style={{
              marginBottom: '1.5rem',
              paddingBottom: '1.5rem',
              borderBottom: '1px solid var(--glass-border)',
            }}
          >
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem', alignItems: 'flex-start' }}>
              <div>
                <div
                  className="mono"
                  style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}
                >
                  Top prediction
                </div>
                <div style={{ fontWeight: 600, fontSize: '1.2rem' }}>
                  {labelForClass(pred.class)}
                </div>
              </div>
              <div style={{ flex: '1 1 200px', minWidth: 0 }}>
                <div
                  className="mono"
                  style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}
                >
                  Confidence
                </div>
                <div className="confidence-bar" style={{ marginBottom: '0.5rem' }}>
                  <motion.div
                    className="confidence-fill"
                    initial={reduce ? false : { scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: reduce ? 0 : 0.55, ease: [0.25, 0.1, 0.25, 1] }}
                    style={{
                      width: `${pred.confidence * 100}%`,
                      transformOrigin: 'left center',
                    }}
                  />
                </div>
                <div
                  className="mono"
                  style={{ fontSize: '1.65rem', fontWeight: 700, color: 'var(--accent-strong)' }}
                >
                  {(pred.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          <div
            className="mono"
            style={{
              fontSize: '0.78rem',
              color: 'var(--text-muted)',
              marginBottom: '0.85rem',
              fontWeight: 600,
            }}
          >
            All classifications
          </div>
          <StaggerList key={`classes-${animationKey}`} className="class-rows">
            {sortedClasses.map(([code, confidence]) => (
              <StaggerItem key={code}>
                <div>
                  <div className="class-row-header">
                    <span>{labelForClass(code)}</span>
                    <span className="mono" style={{ color: 'var(--accent-strong)' }}>
                      {(confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="class-bar-track">
                    <motion.div
                      className="class-bar-fill"
                      initial={reduce ? false : { scaleX: 0 }}
                      animate={{ scaleX: 1 }}
                      transition={{
                        duration: reduce ? 0 : 0.5,
                        ease: [0.25, 0.1, 0.25, 1],
                      }}
                      style={{
                        width: `${confidence * 100}%`,
                        transformOrigin: 'left center',
                      }}
                    />
                  </div>
                </div>
              </StaggerItem>
            ))}
          </StaggerList>
        </motion.div>
      ) : null}

      {!loading && !result ? (
        <div className="placeholder-center">
          <p style={{ margin: 0 }}>Upload an image to see results</p>
        </div>
      ) : null}
    </GlassCard>
  )
}
