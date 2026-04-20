import { motion, useReducedMotion } from 'framer-motion'

type SpinnerProps = {
  size?: number
  className?: string
  label?: string
}

export function Spinner({ size = 28, className = '', label = 'Loading' }: SpinnerProps) {
  const reduce = useReducedMotion()

  if (reduce) {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        className={className}
        role="img"
        aria-label={label}
      >
        <circle
          cx="16"
          cy="16"
          r="12"
          fill="none"
          stroke="rgba(124, 158, 255, 0.35)"
          strokeWidth="3"
        />
        <circle
          cx="16"
          cy="16"
          r="12"
          fill="none"
          stroke="var(--accent-strong)"
          strokeWidth="3"
          strokeDasharray="56 20"
          strokeLinecap="round"
          transform="rotate(-90 16 16)"
        />
      </svg>
    )
  }

  return (
    <motion.svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      className={className}
      role="img"
      aria-label={label}
      animate={{ rotate: 360 }}
      transition={{ repeat: Infinity, duration: 0.85, ease: 'linear' }}
      style={{ transformOrigin: 'center' }}
    >
      <circle
        cx="16"
        cy="16"
        r="12"
        fill="none"
        stroke="rgba(124, 158, 255, 0.2)"
        strokeWidth="3"
      />
      <circle
        cx="16"
        cy="16"
        r="12"
        fill="none"
        stroke="url(#spinnerGrad)"
        strokeWidth="3"
        strokeLinecap="round"
        strokeDasharray="48 28"
        transform="rotate(-90 16 16)"
      />
      <defs>
        <linearGradient id="spinnerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="var(--accent)" />
          <stop offset="100%" stopColor="var(--accent-strong)" />
        </linearGradient>
      </defs>
    </motion.svg>
  )
}
