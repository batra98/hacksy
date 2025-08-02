'use client'

import { useState } from 'react'
import { 
  Github, 
  ExternalLink, 
  Clock, 
  Target, 
  Code, 
  Lightbulb,
  ChevronDown,
  ChevronUp,
  Star,
  Zap
} from './Icons'
import ProfileVisualization from './ProfileVisualization'

export default function RecommendationDisplay({ recommendations, username, profileData }) {
  const [expandedProjects, setExpandedProjects] = useState({})

  const toggleProject = (index) => {
    setExpandedProjects(prev => ({
      ...prev,
      [index]: !prev[index]
    }))
  }

  // Parse the recommendations text to extract structured data
  const parseRecommendations = (text) => {
    // Extract profile analysis section
    const profileMatch = text.match(/📊 \*\*Profile Analysis Summary\*\*(.*?)🚀/s)
    const profileInfo = profileMatch ? profileMatch[1].trim() : ''

    // Extract projects section - look for numbered projects
    const projectsMatch = text.match(/🚀 \*\*Top \d+-?\d* Hackathon Project Recommendations\*\*(.*?)💡/s)
    const projectsText = projectsMatch ? projectsMatch[1] : text

    // Split by numbered items (1., 2., 3., etc.)
    const projectSections = projectsText.split(/\n\d+\.\s+/).filter(section => section.trim())

    const projects = projectSections.map((section, index) => {
      // Extract title (first line with emoji and title)
      const titleMatch = section.match(/🎯\s*\*\*(.+?)\*\*/)
      const title = titleMatch ? `🎯 ${titleMatch[1]}` : `Project ${index + 1}`

      // Extract description
      const descMatch = section.match(/\*\*Description:\*\*\s*(.+?)(?=\*\*Tech Stack:|$)/s)
      const description = descMatch ? descMatch[1].trim() : ''

      // Extract tech stack
      const techMatch = section.match(/\*\*Tech Stack:\*\*\s*(.+?)(?=\*\*Implementation:|$)/s)
      const techStack = techMatch ? techMatch[1].trim() : ''

      // Extract implementation
      const implMatch = section.match(/\*\*Implementation:\*\*(.*?)(?=\*\*Difficulty:|$)/s)
      const implementation = implMatch ? implMatch[1].trim() : ''

      // Extract difficulty
      const difficultyMatch = section.match(/\*\*Difficulty:\*\*\s*(.+?)(?=\*\*Impact:|$)/s)
      const difficulty = difficultyMatch ? difficultyMatch[1].trim() : 'Intermediate'

      // Extract impact
      const impactMatch = section.match(/\*\*Impact:\*\*\s*(.+?)(?=\*\*Time Estimate:|$)/s)
      const impact = impactMatch ? impactMatch[1].trim() : ''

      // Extract time estimate
      const timeMatch = section.match(/\*\*Time Estimate:\*\*\s*(.+?)(?=\n|$)/s)
      const timeEstimate = timeMatch ? timeMatch[1].trim() : '24-48 hours'

      return {
        title,
        description,
        techStack,
        implementation,
        difficulty,
        timeEstimate,
        impact
      }
    })

    return { profileInfo, projects }
  }

  const { profileInfo, projects } = parseRecommendations(recommendations)

  const getDifficultyColor = (difficulty) => {
    const level = difficulty.toLowerCase()
    if (level.includes('beginner')) return 'bg-green-100 text-green-800'
    if (level.includes('advanced')) return 'bg-red-100 text-red-800'
    return 'bg-blue-100 text-blue-800'
  }

  const getDifficultyIcon = (difficulty) => {
    const level = difficulty.toLowerCase()
    if (level.includes('beginner')) return '🟢'
    if (level.includes('advanced')) return '🔴'
    return '🟡'
  }

  // Parse tech stack into individual tools
  const parseTechStack = (techStack) => {
    if (!techStack) return []
    return techStack.split(/[,+&]/).map(tool => tool.trim()).filter(tool => tool.length > 0)
  }

  // Get color for tech tool
  const getToolColor = (tool) => {
    const toolLower = tool.toLowerCase()
    if (toolLower.includes('python')) return 'bg-blue-100 text-blue-800 border-blue-200'
    if (toolLower.includes('javascript') || toolLower.includes('js')) return 'bg-yellow-100 text-yellow-800 border-yellow-200'
    if (toolLower.includes('react')) return 'bg-cyan-100 text-cyan-800 border-cyan-200'
    if (toolLower.includes('go')) return 'bg-teal-100 text-teal-800 border-teal-200'
    if (toolLower.includes('node')) return 'bg-green-100 text-green-800 border-green-200'
    if (toolLower.includes('flask') || toolLower.includes('fastapi')) return 'bg-red-100 text-red-800 border-red-200'
    if (toolLower.includes('docker')) return 'bg-blue-100 text-blue-800 border-blue-200'
    if (toolLower.includes('api')) return 'bg-purple-100 text-purple-800 border-purple-200'
    return 'bg-gray-100 text-gray-800 border-gray-200'
  }

  return (
    <div className="space-y-8">
      {/* Profile Visualization */}
      {profileData && (
        <ProfileVisualization
          profileData={profileData}
          username={username}
        />
      )}

      {/* Profile Summary */}
      {profileInfo && (
        <div className="glass-card rounded-2xl p-8 border border-gradient-to-r from-blue-200 to-purple-200 bg-gradient-to-br from-blue-50 to-purple-50">
          <div className="flex items-center mb-6">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl mr-4">
              <Github className="w-8 h-8 text-white" />
            </div>
            <div>
              <h3 className="text-2xl font-bold text-gray-800">Profile Analysis</h3>
              <p className="text-gray-600">GitHub insights for {username}</p>
            </div>
          </div>
          <div className="bg-white/70 rounded-xl p-6 backdrop-blur-sm">
            <div className="text-gray-700 whitespace-pre-line leading-relaxed text-lg">
              {profileInfo}
            </div>
          </div>
        </div>
      )}

      {/* Projects Grid */}
      {projects.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center mb-6">
            <Zap className="w-6 h-6 text-yellow-500 mr-3" />
            <h3 className="text-2xl font-bold text-gray-800">Recommended Projects</h3>
          </div>
          
          <div className="grid gap-8">
            {projects.map((project, index) => (
              <div key={index} className="glass-card rounded-2xl overflow-hidden hover:shadow-2xl transition-all duration-300 border border-gray-200 hover:border-blue-300">
                <div className="p-8">
                  {/* Project Header */}
                  <div className="flex items-start justify-between mb-6">
                    <div className="flex-1">
                      <h4 className="text-2xl font-bold text-gray-800 mb-3 flex items-center">
                        {project.title}
                        <span className="ml-3 text-lg">
                          {getDifficultyIcon(project.difficulty)}
                        </span>
                      </h4>
                      <p className="text-gray-600 leading-relaxed text-lg mb-4">
                        {project.description}
                      </p>
                    </div>
                    <button
                      onClick={() => toggleProject(index)}
                      className="ml-6 p-3 hover:bg-blue-50 rounded-xl transition-colors border border-gray-200 hover:border-blue-300"
                    >
                      {expandedProjects[index] ?
                        <ChevronUp className="w-6 h-6 text-blue-600" /> :
                        <ChevronDown className="w-6 h-6 text-blue-600" />
                      }
                    </button>
                  </div>

                  {/* Tech Stack Tools */}
                  {project.techStack && (
                    <div className="mb-6">
                      <div className="flex items-center mb-3">
                        <Code className="w-5 h-5 text-purple-600 mr-2" />
                        <span className="font-semibold text-gray-800">Tech Stack</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {parseTechStack(project.techStack).map((tool, toolIndex) => (
                          <span
                            key={toolIndex}
                            className={`px-3 py-2 rounded-lg text-sm font-medium border ${getToolColor(tool)} transition-all hover:scale-105`}
                          >
                            {tool}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Quick Info */}
                  <div className="flex flex-wrap gap-4 mb-6">
                    <span className={`px-4 py-2 rounded-full text-sm font-medium ${getDifficultyColor(project.difficulty)}`}>
                      {project.difficulty}
                    </span>
                    {project.timeEstimate && (
                      <span className="px-4 py-2 bg-blue-50 text-blue-700 rounded-full text-sm font-medium flex items-center border border-blue-200">
                        <Clock className="w-4 h-4 mr-2" />
                        {project.timeEstimate}
                      </span>
                    )}
                  </div>

                  {/* Expanded Details */}
                  {expandedProjects[index] && (
                    <div className="space-y-6 pt-6 border-t border-gray-200 animate-slide-up">

                      {project.implementation && (
                        <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-xl p-6 border border-orange-200">
                          <div className="flex items-center mb-4">
                            <Lightbulb className="w-5 h-5 text-orange-600 mr-3" />
                            <span className="font-semibold text-gray-800 text-lg">Implementation Guide</span>
                          </div>
                          <div className="text-gray-700 whitespace-pre-line leading-relaxed">
                            {project.implementation}
                          </div>
                        </div>
                      )}

                      {project.impact && (
                        <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200">
                          <div className="flex items-center mb-4">
                            <Target className="w-5 h-5 text-green-600 mr-3" />
                            <span className="font-semibold text-gray-800 text-lg">Impact & Value</span>
                          </div>
                          <p className="text-gray-700 leading-relaxed">
                            {project.impact}
                          </p>
                        </div>
                      )}

                      <div className="flex gap-4 pt-4">
                        <button className="btn-primary flex items-center gap-2 px-6 py-3 text-sm font-medium">
                          <Star className="w-4 h-4" />
                          Save Project
                        </button>
                        <button className="btn-secondary flex items-center gap-2 px-6 py-3 text-sm font-medium">
                          <ExternalLink className="w-4 h-4" />
                          Learn More
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fallback for unstructured content */}
      {projects.length === 0 && (
        <div className="glass-card rounded-xl p-6">
          <div className="prose prose-lg max-w-none">
            <div className="whitespace-pre-wrap text-gray-700 leading-relaxed">
              {recommendations}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
