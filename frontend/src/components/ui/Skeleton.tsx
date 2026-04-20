type SkeletonProps = {
  className?: string
  style?: React.CSSProperties
}

export function Skeleton({ className = '', style }: SkeletonProps) {
  return (
    <div
      className={`skeleton skeleton-animated ${className}`.trim()}
      style={style}
      aria-hidden
    />
  )
}
