import type { ReactNode } from 'react'
import type { HTMLMotionProps } from 'framer-motion'
import { motion } from 'framer-motion'

type GlassCardProps = HTMLMotionProps<'div'> & {
  children: ReactNode
}

export function GlassCard({ children, className = '', ...rest }: GlassCardProps) {
  return (
    <motion.div className={`glass-card ${className}`.trim()} {...rest}>
      {children}
    </motion.div>
  )
}
