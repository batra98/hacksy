'use client';

import { useState } from 'react';
import {
  Trophy,
  Github,
  Sparkles,
  Rocket,
  Users,
  Code,
  Brain,
  Search,
  Star,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  ExternalLink
} from '../components/Icons';
import LoadingSpinner from '../components/LoadingSpinner';
import FeatureCard from '../components/FeatureCard';
import RecommendationDisplay from '../components/RecommendationDisplay';
import ErrorDisplay from '../components/ErrorDisplay';

export default function Home() {
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState('');
  const [profileData, setProfileData] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim()) return;

    setLoading(true);
    setError('');
    setResult('');
    setProfileData(null);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim() }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Analysis failed');
      }

      setResult(data.recommendations);
      setProfileData(data.profile);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Analysis",
      description: "Advanced AI analyzes your GitHub profile to understand your coding patterns and interests"
    },
    {
      icon: Rocket,
      title: "Smart Recommendations",
      description: "Get personalized hackathon project ideas that match your skills and experience"
    },
    {
      icon: Users,
      title: "Community Insights",
      description: "Discover trending technologies and popular project categories in the developer community"
    },
    {
      icon: Code,
      title: "Skill Matching",
      description: "Find projects that leverage your existing skills while introducing new challenges"
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-white to-accent-50"></div>
        <div className="absolute top-0 left-0 w-full h-full">
          <div className="absolute top-20 left-10 w-72 h-72 bg-primary-200 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse-slow"></div>
          <div className="absolute top-40 right-10 w-72 h-72 bg-accent-200 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse-slow" style={{animationDelay: '2s'}}></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
          {/* Header */}
          <div className="text-center mb-16">
            <div className="flex items-center justify-center mb-6">
              <div className="p-4 bg-gradient-to-r from-primary-500 to-accent-500 rounded-2xl shadow-lg">
                <Trophy className="w-12 h-12 text-white" />
              </div>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              <span className="gradient-text">Hacksy</span>
              <br />
              <span className="text-gray-800">AI Agents Recommender</span>
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Discover personalized AI agent and hackathon project recommendations powered by advanced AI analysis of your GitHub profile
            </p>
          </div>

          {/* Search Form */}
          <div className="max-w-2xl mx-auto mb-16">
            <div className="glass-card rounded-2xl p-8 shadow-2xl">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <Github className="h-5 w-5 text-gray-400" />
                  </div>
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="Enter GitHub username (e.g., torvalds)"
                    className="input-field pl-12 text-lg"
                    disabled={loading}
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading || !username.trim()}
                  className="btn-primary w-full text-lg py-4 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
                >
                  {loading ? (
                    <>
                      <LoadingSpinner size="sm" />
                      <span>Analyzing Profile<span className="loading-dots"></span></span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-5 h-5" />
                      Get AI Recommendations
                      <ArrowRight className="w-5 h-5" />
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      {!result && !error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">Why Choose Hacksy?</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our AI-powered platform provides intelligent insights to help you discover the perfect projects and opportunities
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <FeatureCard
                key={index}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                className="animate-fade-in"
                style={{animationDelay: `${index * 0.1}s`}}
              />
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <ErrorDisplay
          error={error}
          username={username}
          onRetry={() => handleSubmit({ preventDefault: () => {} })}
          onClear={() => {
            setError('');
            setUsername('');
            setResult('');
            setProfileData(null);
          }}
        />
      )}

      {/* Results */}
      {result && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
          <div className="animate-slide-up">
            {/* Header */}
            <div className="glass-card rounded-2xl p-8 shadow-2xl mb-8">
              <div className="flex items-center mb-4">
                <div className="p-3 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl mr-4">
                  <CheckCircle className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-gray-800">
                    AI Recommendations for {username}
                  </h2>
                  <p className="text-gray-600 mt-1">
                    Personalized insights based on your GitHub profile analysis
                  </p>
                </div>
              </div>
            </div>

            {/* Recommendations Display */}
            <RecommendationDisplay
              recommendations={result}
              username={username}
              profileData={profileData}
            />

            {/* Action Buttons */}
            <div className="mt-12 flex flex-wrap gap-4 justify-center">
              <button
                onClick={() => {
                  setResult('');
                  setUsername('');
                  setProfileData(null);
                }}
                className="btn-secondary flex items-center gap-2"
              >
                <Search className="w-4 h-4" />
                New Search
              </button>
              <a
                href={`https://github.com/${username}`}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary flex items-center gap-2"
              >
                <Github className="w-4 h-4" />
                View GitHub Profile
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex items-center justify-center mb-4">
              <Trophy className="w-8 h-8 text-accent-400 mr-2" />
              <span className="text-2xl font-bold">Hacksy</span>
            </div>
            <p className="text-gray-400 mb-4">
              Empowering developers with AI-driven insights for hackathons and projects
            </p>
            <div className="flex items-center justify-center space-x-6 text-sm text-gray-500">
              <span>Built with ❤️ for the developer community</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
