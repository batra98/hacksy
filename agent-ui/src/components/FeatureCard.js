'use client'

export default function FeatureCard({ icon: Icon, title, description, className = '' }) {
  return (
    <div className={`glass-card rounded-xl p-6 hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 ${className}`}>
      <div className="flex items-center mb-4">
        <div className="p-3 bg-gradient-to-r from-primary-500 to-accent-500 rounded-lg">
          <Icon className="w-6 h-6 text-white" />
        </div>
      </div>
      <h3 className="text-lg font-semibold text-gray-800 mb-2">{title}</h3>
      <p className="text-gray-600 text-sm leading-relaxed">{description}</p>
    </div>
  )
}
