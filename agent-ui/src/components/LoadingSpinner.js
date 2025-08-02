'use client'

import { Loader2 } from './Icons'

export default function LoadingSpinner({ size = 'md', className = '' }) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  }

  return (
    <Loader2 
      className={`animate-spin text-primary-500 ${sizeClasses[size]} ${className}`}
    />
  )
}
