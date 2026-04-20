import type { HTMLMotionProps } from 'framer-motion'
import { useReducedMotion } from 'framer-motion'
import { motion } from 'framer-motion'

type Variant = 'primary' | 'secondary' | 'danger'

type ButtonProps = HTMLMotionProps<'button'> & {
  variant?: Variant
  loading?: boolean
}

const variantClass: Record<Variant, string> = {
  primary: 'btn-primary',
  secondary: 'btn-secondary',
  danger: 'btn-danger',
}

export function Button({
  variant = 'primary',
  loading = false,
  children,
  disabled,
  className = '',
  ...props
}: ButtonProps) {
  const reduce = useReducedMotion()
  const isDisabled = disabled || loading

  return (
    <motion.button
      type="button"
      className={`btn ${variantClass[variant]} ${className}`.trim()}
      disabled={isDisabled}
      whileHover={reduce || isDisabled ? undefined : { scale: 1.02 }}
      whileTap={reduce || isDisabled ? undefined : { scale: 0.98 }}
      transition={{ type: 'spring', stiffness: 520, damping: 28 }}
      {...props}
    >
      {children}
    </motion.button>
  )
}
