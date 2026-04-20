import { useReducedMotion } from 'framer-motion'
import { GlassCard } from './ui/GlassCard'
import { StaggerItem, StaggerList } from './ui/StaggerList'
import type { SessionStats } from '../lib/types'

type StatsSectionProps = {
  stats: SessionStats
}

export function StatsSection({ stats }: StatsSectionProps) {
  const reduce = useReducedMotion()
  const avgConf =
    stats.totalAnalyses > 0
      ? Math.round((stats.totalConfidence / stats.totalAnalyses) * 100)
      : 0
  const avgTime =
    stats.totalAnalyses > 0 ? Math.round(stats.totalTime / stats.totalAnalyses) : 0

  const tiles = [
    { label: 'Images analyzed', value: String(stats.totalAnalyses) },
    { label: 'Avg confidence', value: `${avgConf}%` },
    { label: 'Avg processing', value: `${avgTime}ms` },
  ]

  return (
    <GlassCard
      className="inner-card-padding"
      initial={reduce ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduce ? 0 : 0.5, delay: reduce ? 0 : 0.26 }}
    >
      <h2>Session statistics</h2>
      <StaggerList className="stats-grid" stagger={0.08}>
        {tiles.map((t) => (
          <StaggerItem key={t.label}>
            <div className="stat-tile">
              <div className="stat-value">{t.value}</div>
              <div className="stat-label">{t.label}</div>
            </div>
          </StaggerItem>
        ))}
      </StaggerList>
    </GlassCard>
  )
}
