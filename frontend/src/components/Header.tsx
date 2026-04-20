import { motion, useReducedMotion } from 'framer-motion'
import type { DeviceConfig } from '../lib/types'

type HeaderProps = {
  config: DeviceConfig | null
}

export function Header({ config }: HeaderProps) {
  const reduce = useReducedMotion()

  return (
    <header className="glass-header" style={{ position: 'sticky', top: 0, zIndex: 1000 }}>
      <div className="header-inner">
        <motion.div
          className="logo-row"
          initial={reduce ? false : { opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduce ? 0 : 0.45, ease: [0.25, 0.1, 0.25, 1] }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            aria-hidden
          >
            <circle cx="12" cy="12" r="10" />
            <path d="M12 6v6l4 2" />
          </svg>
          Derma AI
        </motion.div>
        <div
          className={`device-badge ${config?.gpu_available ? 'gpu' : 'cpu'}`}
          id="deviceBadge"
        >
          <span
            className={`dot ${config ? 'dot-pulse' : ''}`}
            aria-hidden
          />
          <span id="deviceText">{config ? config.device : 'Checking device…'}</span>
        </div>
      </div>
    </header>
  )
}
