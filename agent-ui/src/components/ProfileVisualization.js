'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Github, 
  Users, 
  Code, 
  Star,
  Calendar,
  MapPin,
  Building,
  GitBranch,
  Activity,
  TrendingUp
} from './Icons'

export default function ProfileVisualization({ profileData, username }) {
  const [animatedStats, setAnimatedStats] = useState({
    repos: 0,
    followers: 0,
    following: 0
  })

  // Animate numbers on mount
  useEffect(() => {
    const duration = 2000
    const steps = 60
    const interval = duration / steps

    let step = 0
    const timer = setInterval(() => {
      step++
      const progress = step / steps
      const easeOut = 1 - Math.pow(1 - progress, 3)

      setAnimatedStats({
        repos: Math.floor(easeOut * (profileData?.repos || 0)),
        followers: Math.floor(easeOut * (profileData?.followers || 0)),
        following: Math.floor(easeOut * (profileData?.following || 0))
      })

      if (step >= steps) {
        clearInterval(timer)
        setAnimatedStats({
          repos: profileData?.repos || 0,
          followers: profileData?.followers || 0,
          following: profileData?.following || 0
        })
      }
    }, interval)

    return () => clearInterval(timer)
  }, [profileData])

  // Language colors mapping
  const languageColors = {
    'Python': '#3776ab',
    'JavaScript': '#f7df1e',
    'TypeScript': '#3178c6',
    'Go': '#00add8',
    'Java': '#ed8b00',
    'C++': '#00599c',
    'C': '#a8b9cc',
    'Rust': '#000000',
    'Swift': '#fa7343',
    'Kotlin': '#7f52ff',
    'PHP': '#777bb4',
    'Ruby': '#cc342d',
    'C#': '#239120',
    'HTML': '#e34f26',
    'CSS': '#1572b6',
    'Shell': '#89e051',
    'Jupyter Notebook': '#da5b0b'
  }

  const getLanguageColor = (lang) => languageColors[lang] || '#6b7280'

  // Calculate language percentages (mock data for visualization)
  const languages = profileData?.languages || []
  const languageData = languages.map((lang, index) => ({
    name: lang,
    percentage: Math.max(30 - index * 5, 5), // Mock percentages
    color: getLanguageColor(lang)
  }))

  // Experience level calculation
  const getExperienceLevel = (repos) => {
    if (repos > 50) return { level: 'Expert', color: '#dc2626', percentage: 90 }
    if (repos > 20) return { level: 'Advanced', color: '#ea580c', percentage: 75 }
    if (repos > 5) return { level: 'Intermediate', color: '#ca8a04', percentage: 60 }
    return { level: 'Beginner', color: '#16a34a', percentage: 30 }
  }

  const experience = getExperienceLevel(profileData?.repos || 0)

  // Activity score (mock calculation)
  const activityScore = Math.min(((profileData?.repos || 0) * 2 + (profileData?.followers || 0)) / 10, 100)

  return (
    <div className="space-y-8">
      {/* Profile Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="glass-card rounded-2xl p-8 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 border border-blue-200"
      >
        <div className="flex items-center mb-6">
          <div className="p-4 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl mr-6 shadow-lg">
            <Github className="w-10 h-10 text-white" />
          </div>
          <div>
            <h2 className="text-3xl font-bold text-gray-800 mb-2">
              {profileData?.name || username}
            </h2>
            <p className="text-gray-600 text-lg">@{username}</p>
            {profileData?.bio && (
              <p className="text-gray-700 mt-2 max-w-2xl">{profileData.bio}</p>
            )}
          </div>
        </div>

        {/* Profile Details */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {profileData?.company && (
            <div className="flex items-center text-gray-600">
              <Building className="w-5 h-5 mr-2" />
              <span>{profileData.company}</span>
            </div>
          )}
          {profileData?.location && (
            <div className="flex items-center text-gray-600">
              <MapPin className="w-5 h-5 mr-2" />
              <span>{profileData.location}</span>
            </div>
          )}
          {profileData?.created_at && (
            <div className="flex items-center text-gray-600">
              <Calendar className="w-5 h-5 mr-2" />
              <span>Joined {new Date(profileData.created_at).getFullYear()}</span>
            </div>
          )}
        </div>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Repositories */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="glass-card rounded-xl p-6 text-center bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200"
        >
          <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl w-fit mx-auto mb-4">
            <Code className="w-8 h-8 text-white" />
          </div>
          <div className="text-3xl font-bold text-gray-800 mb-2">
            {animatedStats.repos.toLocaleString()}
          </div>
          <div className="text-gray-600">Public Repositories</div>
        </motion.div>

        {/* Followers */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="glass-card rounded-xl p-6 text-center bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200"
        >
          <div className="p-3 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl w-fit mx-auto mb-4">
            <Users className="w-8 h-8 text-white" />
          </div>
          <div className="text-3xl font-bold text-gray-800 mb-2">
            {animatedStats.followers.toLocaleString()}
          </div>
          <div className="text-gray-600">Followers</div>
        </motion.div>

        {/* Following */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="glass-card rounded-xl p-6 text-center bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200"
        >
          <div className="p-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl w-fit mx-auto mb-4">
            <GitBranch className="w-8 h-8 text-white" />
          </div>
          <div className="text-3xl font-bold text-gray-800 mb-2">
            {animatedStats.following.toLocaleString()}
          </div>
          <div className="text-gray-600">Following</div>
        </motion.div>
      </div>

      {/* Languages and Experience */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Programming Languages */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="glass-card rounded-xl p-6 bg-gradient-to-br from-orange-50 to-red-50 border border-orange-200"
        >
          <div className="flex items-center mb-6">
            <Code className="w-6 h-6 text-orange-600 mr-3" />
            <h3 className="text-xl font-bold text-gray-800">Programming Languages</h3>
          </div>
          
          <div className="space-y-4">
            {languageData.map((lang, index) => (
              <motion.div 
                key={lang.name}
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ duration: 1, delay: 0.5 + index * 0.1 }}
                className="space-y-2"
              >
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-700">{lang.name}</span>
                  <span className="text-sm text-gray-500">{lang.percentage}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${lang.percentage}%` }}
                    transition={{ duration: 1, delay: 0.5 + index * 0.1 }}
                    className="h-3 rounded-full"
                    style={{ backgroundColor: lang.color }}
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Experience Level & Activity */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="space-y-6"
        >
          {/* Experience Level */}
          <div className="glass-card rounded-xl p-6 bg-gradient-to-br from-yellow-50 to-orange-50 border border-yellow-200">
            <div className="flex items-center mb-4">
              <Star className="w-6 h-6 text-yellow-600 mr-3" />
              <h3 className="text-xl font-bold text-gray-800">Experience Level</h3>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold mb-2" style={{ color: experience.color }}>
                {experience.level}
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${experience.percentage}%` }}
                  transition={{ duration: 1.5, delay: 0.8 }}
                  className="h-4 rounded-full"
                  style={{ backgroundColor: experience.color }}
                />
              </div>
              <div className="text-sm text-gray-600">
                Based on {profileData?.repos || 0} public repositories
              </div>
            </div>
          </div>

          {/* Activity Score */}
          <div className="glass-card rounded-xl p-6 bg-gradient-to-br from-teal-50 to-cyan-50 border border-teal-200">
            <div className="flex items-center mb-4">
              <Activity className="w-6 h-6 text-teal-600 mr-3" />
              <h3 className="text-xl font-bold text-gray-800">Activity Score</h3>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-teal-600 mb-2">
                {Math.round(activityScore)}
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${activityScore}%` }}
                  transition={{ duration: 1.5, delay: 1 }}
                  className="h-4 rounded-full bg-gradient-to-r from-teal-500 to-cyan-500"
                />
              </div>
              <div className="text-sm text-gray-600">
                Community engagement & activity
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Recent Repositories */}
      {profileData?.recent_repos && profileData.recent_repos.length > 0 && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="glass-card rounded-xl p-6 bg-gradient-to-br from-indigo-50 to-purple-50 border border-indigo-200"
        >
          <div className="flex items-center mb-6">
            <TrendingUp className="w-6 h-6 text-indigo-600 mr-3" />
            <h3 className="text-xl font-bold text-gray-800">Recent Projects</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {profileData.recent_repos.slice(0, 6).map((repo, index) => (
              <motion.div 
                key={repo}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: 0.7 + index * 0.1 }}
                className="bg-white/70 rounded-lg p-4 border border-indigo-200 hover:border-indigo-300 transition-colors"
              >
                <div className="flex items-center">
                  <GitBranch className="w-4 h-4 text-indigo-600 mr-2" />
                  <span className="font-medium text-gray-800 truncate">{repo}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}
