import { motion, useReducedMotion } from 'framer-motion'
import { GlassCard } from './ui/GlassCard'
import { StaggerItem, StaggerList } from './ui/StaggerList'
import type { HistoryEntry } from '../lib/types'

type HistorySectionProps = {
  history: HistoryEntry[]
  activeIndex: number | null
  onSelect: (index: number) => void
  listKey: number
}

export function HistorySection({
  history,
  activeIndex,
  onSelect,
  listKey,
}: HistorySectionProps) {
  const reduce = useReducedMotion()

  return (
    <GlassCard
      className="inner-card-padding"
      initial={reduce ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduce ? 0 : 0.5, delay: reduce ? 0 : 0.2 }}
    >
      <h2>Analysis history</h2>
      {history.length === 0 ? (
        <div className="placeholder-center" style={{ minHeight: '120px' }}>
          <p style={{ margin: 0 }}>No analyses yet</p>
        </div>
      ) : (
        <StaggerList key={listKey} className="history-grid">
          {history.map((item, index) => (
            <StaggerItem key={`${item.image.slice(-40)}-${index}`}>
              <motion.button
                type="button"
                className={`history-cell ${activeIndex === index ? 'active' : ''}`.trim()}
                onClick={() => onSelect(index)}
                whileHover={reduce ? undefined : { scale: 1.02 }}
                whileTap={reduce ? undefined : { scale: 0.98 }}
                style={{
                  padding: 0,
                  margin: 0,
                  width: '100%',
                  display: 'block',
                  cursor: 'pointer',
                  font: 'inherit',
                  color: 'inherit',
                  background: 'transparent',
                }}
              >
                <img
                  src={item.image}
                  alt={`${item.className} thumbnail`}
                  className="history-thumb"
                />
                <div className="history-meta">
                  <div
                    style={{
                      fontWeight: 600,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {item.className}
                  </div>
                  <div className="mono" style={{ color: 'var(--text-muted)' }}>
                    {(item.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </motion.button>
            </StaggerItem>
          ))}
        </StaggerList>
      )}
    </GlassCard>
  )
}
