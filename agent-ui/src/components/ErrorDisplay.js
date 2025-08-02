'use client'

import { motion } from 'framer-motion'
import { 
  AlertCircle, 
  Github, 
  Search, 
  RefreshCw, 
  ExternalLink,
  HelpCircle,
  Zap
} from './Icons'

export default function ErrorDisplay({ error, username, onRetry, onClear }) {
  // Determine error type and customize message
  const getErrorInfo = (errorMessage) => {
    const message = errorMessage.toLowerCase()
    
    if (message.includes('not found') || message.includes('404')) {
      return {
        type: 'not_found',
        title: '🔍 GitHub Profile Not Found',
        subtitle: `We couldn't find a GitHub user named "${username}"`,
        description: 'Double-check the username spelling or try a different one.',
        icon: Search,
        color: 'blue',
        suggestions: [
          'Verify the username spelling',
          'Check if the profile is public',
          'Try searching on GitHub.com first'
        ]
      }
    }
    
    if (message.includes('rate limit') || message.includes('403')) {
      return {
        type: 'rate_limit',
        title: '⏰ Rate Limit Reached',
        subtitle: 'GitHub API rate limit exceeded',
        description: 'Too many requests in a short time. Please wait a few minutes.',
        icon: RefreshCw,
        color: 'orange',
        suggestions: [
          'Wait 5-10 minutes before trying again',
          'GitHub allows 60 requests per hour for unauthenticated users',
          'Try again later for best results'
        ]
      }
    }
    
    if (message.includes('network') || message.includes('connection')) {
      return {
        type: 'network',
        title: '🌐 Connection Issue',
        subtitle: 'Unable to connect to GitHub',
        description: 'Please check your internet connection and try again.',
        icon: Zap,
        color: 'red',
        suggestions: [
          'Check your internet connection',
          'Try refreshing the page',
          'GitHub might be temporarily unavailable'
        ]
      }
    }
    
    // Generic error
    return {
      type: 'generic',
      title: '⚠️ Something Went Wrong',
      subtitle: 'An unexpected error occurred',
      description: 'Please try again or contact support if the issue persists.',
      icon: AlertCircle,
      color: 'red',
      suggestions: [
        'Try again with a different username',
        'Refresh the page and retry',
        'Check if GitHub is accessible'
      ]
    }
  }

  const errorInfo = getErrorInfo(error)
  const IconComponent = errorInfo.icon

  const colorClasses = {
    blue: {
      bg: 'from-blue-50 to-cyan-50',
      border: 'border-blue-200',
      icon: 'text-blue-600',
      title: 'text-blue-800',
      subtitle: 'text-blue-700',
      button: 'bg-blue-600 hover:bg-blue-700',
      accent: 'bg-blue-100 text-blue-800'
    },
    orange: {
      bg: 'from-orange-50 to-yellow-50',
      border: 'border-orange-200',
      icon: 'text-orange-600',
      title: 'text-orange-800',
      subtitle: 'text-orange-700',
      button: 'bg-orange-600 hover:bg-orange-700',
      accent: 'bg-orange-100 text-orange-800'
    },
    red: {
      bg: 'from-red-50 to-pink-50',
      border: 'border-red-200',
      icon: 'text-red-600',
      title: 'text-red-800',
      subtitle: 'text-red-700',
      button: 'bg-red-600 hover:bg-red-700',
      accent: 'bg-red-100 text-red-800'
    }
  }

  const colors = colorClasses[errorInfo.color]

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mb-8"
    >
      <div className={`glass-card rounded-2xl overflow-hidden border ${colors.border} bg-gradient-to-br ${colors.bg}`}>
        <div className="p-8">
          {/* Header */}
          <div className="flex items-start mb-6">
            <div className={`p-4 rounded-2xl mr-6 bg-white/70 border ${colors.border}`}>
              <IconComponent className={`w-8 h-8 ${colors.icon}`} />
            </div>
            <div className="flex-1">
              <h3 className={`text-2xl font-bold ${colors.title} mb-2`}>
                {errorInfo.title}
              </h3>
              <p className={`text-lg ${colors.subtitle} mb-2`}>
                {errorInfo.subtitle}
              </p>
              <p className="text-gray-600">
                {errorInfo.description}
              </p>
            </div>
          </div>

          {/* Error Details */}
          <div className="bg-white/50 rounded-xl p-6 mb-6 border border-gray-200">
            <div className="flex items-center mb-3">
              <HelpCircle className="w-5 h-5 text-gray-500 mr-2" />
              <span className="font-semibold text-gray-700">What can you do?</span>
            </div>
            <ul className="space-y-2">
              {errorInfo.suggestions.map((suggestion, index) => (
                <li key={index} className="flex items-start">
                  <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                  <span className="text-gray-600">{suggestion}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4">
            {onRetry && (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={onRetry}
                className={`px-6 py-3 ${colors.button} text-white rounded-xl font-medium flex items-center gap-2 transition-colors`}
              >
                <RefreshCw className="w-5 h-5" />
                Try Again
              </motion.button>
            )}
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={onClear}
              className="px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-xl font-medium flex items-center gap-2 transition-colors"
            >
              <Search className="w-5 h-5" />
              Try Different Username
            </motion.button>

            {errorInfo.type === 'not_found' && (
              <motion.a
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                href={`https://github.com/${username}`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium flex items-center gap-2 transition-colors"
              >
                <Github className="w-5 h-5" />
                Check on GitHub
                <ExternalLink className="w-4 h-4" />
              </motion.a>
            )}
          </div>

          {/* Fun Fact */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className={`mt-6 p-4 ${colors.accent} rounded-xl`}
          >
            <div className="flex items-center">
              <span className="text-2xl mr-3">💡</span>
              <div>
                <span className="font-semibold">Pro Tip: </span>
                {errorInfo.type === 'not_found' && "Popular GitHub usernames to try: octocat, torvalds, gaearon, or your own!"}
                {errorInfo.type === 'rate_limit' && "GitHub's rate limiting helps keep their service fast and reliable for everyone."}
                {errorInfo.type === 'network' && "GitHub serves over 100 million developers worldwide - that's a lot of traffic!"}
                {errorInfo.type === 'generic' && "Every great developer has encountered errors - it's all part of the journey!"}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
