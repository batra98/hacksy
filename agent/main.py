#!/usr/bin/env python3
"""
AI Agents Backend Service for Hackathon Recommender
Handles agent orchestration and MCP tool integration
"""

import os
import logging
import aiohttp
import asyncio
import json
import random
from typing import Dict, Any, List, Optional

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
class AnalysisRequest(BaseModel):
    username: str
    agent: str = "hackathon_recommender"

class AnalysisResponse(BaseModel):
    success: bool
    agent: str
    recommendations: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"

class AgentConfig:
    def __init__(self, config_path: str = "agents.yaml"):
        self.config_path = config_path
        self.agents = {}
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load agent configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
                self.agents = data.get('agents', {})
                self.config = data.get('config', {})
            logger.info(f"Loaded {len(self.agents)} agents from config")
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")
            # Use default config if file not found
            self.agents = {
                "hackathon_recommender": {
                    "name": "Hackathon Project Recommender",
                    "description": "Analyzes GitHub profiles to recommend personalized hackathon projects"
                }
            }
            self.config = {}

class GitHubAnalyzer:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        self.base_url = "https://api.github.com"
        
    async def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Fetch real GitHub user profile data"""
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
            
        try:
            async with aiohttp.ClientSession() as session:
                # Get user basic info
                async with session.get(f"{self.base_url}/users/{username}", headers=headers) as response:
                    if response.status == 404:
                        raise Exception(f"GitHub user '{username}' not found. Please check the username and try again.")
                    elif response.status == 403:
                        raise Exception("GitHub API rate limit exceeded. Please try again in a few minutes.")
                    elif response.status != 200:
                        raise Exception(f"GitHub API error: Unable to fetch profile (Status: {response.status})")
                    user_data = await response.json()
                
                # Get user repositories
                async with session.get(f"{self.base_url}/users/{username}/repos?per_page=100", headers=headers) as response:
                    if response.status == 200:
                        repos_data = await response.json()
                    else:
                        repos_data = []
                
                # Analyze repositories for languages
                languages = {}
                for repo in repos_data[:20]:  # Analyze top 20 repos
                    if repo.get('language'):
                        lang = repo['language']
                        languages[lang] = languages.get(lang, 0) + 1
                
                # Get top languages
                top_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]
                top_languages = [lang[0] for lang in top_languages]
                
                return {
                    "username": username,
                    "name": user_data.get('name', username),
                    "bio": user_data.get('bio', ''),
                    "repos": user_data.get('public_repos', 0),
                    "followers": user_data.get('followers', 0),
                    "following": user_data.get('following', 0),
                    "languages": top_languages,
                    "company": user_data.get('company', ''),
                    "location": user_data.get('location', ''),
                    "created_at": user_data.get('created_at', ''),
                    "repository_count": len(repos_data),
                    "recent_repos": [repo['name'] for repo in repos_data[:5]]
                }
                
        except Exception as e:
            logger.error(f"GitHub API error for {username}: {e}")
            # Re-raise the exception to be handled by the calling function
            # This ensures proper error messages reach the frontend
            raise e

class AIAgentClient:
    """Client for communicating with AI models via Gemini API"""

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url
        # Configure Gemini API
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            logger.warning("GEMINI_API_KEY not found, will use fallback responses")
            self.model = None

    async def call_agent(self, agent_config: Dict[str, Any], profile_data: Dict[str, Any]) -> str:
        """Call an AI agent with the given configuration and data"""
        try:
            # Extract agent configuration
            instructions = agent_config.get('instructions', '')
            temperature = agent_config.get('parameters', {}).get('temperature', 0.7)
            max_tokens = agent_config.get('parameters', {}).get('max_tokens', 1500)

            # Create the prompt for the AI agent
            prompt = self._create_agent_prompt(instructions, profile_data)

            # Try Gemini API first, fallback if not available
            if self.model:
                try:
                    return await self._call_gemini(prompt, temperature, max_tokens)
                except Exception as gemini_error:
                    logger.warning(f"Gemini call failed: {gemini_error}, using fallback")
            else:
                logger.info("Gemini API not configured, using fallback")

            return self._generate_fallback_response(profile_data)

        except Exception as e:
            logger.error(f"AI agent call failed: {e}")
            return self._generate_fallback_response(profile_data)

    def _create_agent_prompt(self, instructions: str, profile_data: Dict[str, Any]) -> str:
        """Create a detailed prompt for the AI agent"""
        username = profile_data.get('username', 'Unknown')
        languages = profile_data.get('languages', [])
        repos = profile_data.get('repos', 0)
        followers = profile_data.get('followers', 0)
        bio = profile_data.get('bio', '')
        company = profile_data.get('company', '')
        recent_repos = profile_data.get('recent_repos', [])

        prompt = f"""
{instructions}

GITHUB PROFILE DATA:
Username: {username}
Bio: {bio}
Company: {company}
Public Repositories: {repos}
Followers: {followers}
Primary Languages: {', '.join(languages[:5])}
Recent Repositories: {', '.join(recent_repos[:5])}

TASK:
Based on this GitHub profile analysis, generate 5 creative and personalized hackathon project recommendations.

CRITICAL FORMATTING REQUIREMENTS - FOLLOW EXACTLY:

📊 **Profile Analysis Summary**
[Brief analysis of the user's skills and experience]

🚀 **Top 5 Hackathon Project Recommendations**

1. 🎯 **[Project Title]**
DESC: [Clear 2-3 sentence description of what the project does]
TECH: [Comma-separated list of specific technologies]
IMPL: [Step-by-step implementation approach]
DIFF: [Beginner/Intermediate/Advanced]
IMPACT: [Problem it solves and value]
TIME: [Hours estimate like "24-36 hours"]

2. 🎯 **[Project Title]**
DESC: [Description]
TECH: [Technologies]
IMPL: [Implementation]
DIFF: [Difficulty]
IMPACT: [Impact]
TIME: [Time estimate]

[Continue for projects 3-5 with EXACT same format]

💡 **Hackathon Strategy Tips**
[Brief tips]

ABSOLUTELY CRITICAL: Use the exact DESC:, TECH:, IMPL:, DIFF:, IMPACT:, TIME: format for EVERY project. No exceptions.
"""
        return prompt

    async def _call_gemini(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Gemini API for AI generation"""
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1,
            )

            # Generate response using Gemini
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config
            )

            if response.text:
                return response.text
            else:
                raise Exception("No response text generated")

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise e

    def _generate_fallback_response(self, profile_data: Dict[str, Any]) -> str:
        """Generate a smart fallback response when AI calls fail"""
        username = profile_data.get('username', 'Developer')
        languages = profile_data.get('languages', ['Python'])
        repos = profile_data.get('repos', 0)
        followers = profile_data.get('followers', 0)
        bio = profile_data.get('bio', '')
        recent_repos = profile_data.get('recent_repos', [])

        primary_lang = languages[0] if languages else 'Python'

        # Determine experience level
        if repos > 50:
            experience = "Expert"
            complexity = "Advanced"
        elif repos > 20:
            experience = "Advanced"
            complexity = "Intermediate-Advanced"
        elif repos > 5:
            experience = "Intermediate"
            complexity = "Beginner-Intermediate"
        else:
            experience = "Beginner"
            complexity = "Beginner"

        # Generate varied project suggestions based on profile
        project_templates = {
            "Python": [
                {
                    "title": "🤖 AI-Powered Code Assistant",
                    "description": "Build an intelligent coding companion that helps developers write better code with AI suggestions and automated refactoring.",
                    "tech_stack": "Python, OpenAI API, FastAPI, SQLite, React",
                    "implementation": "1. Set up FastAPI backend with OpenAI integration\n2. Create code analysis endpoints\n3. Build React frontend with code editor\n4. Implement real-time suggestions\n5. Add SQLite for user preferences",
                    "difficulty": "Intermediate",
                    "time": "36-48 hours",
                    "impact": "Boost developer productivity by 40% through intelligent code assistance"
                },
                {
                    "title": "📊 Real-time Analytics Dashboard",
                    "description": "Create a beautiful, interactive dashboard that visualizes live data streams with customizable widgets and alerts.",
                    "tech_stack": "Python, Streamlit, Pandas, WebSocket, PostgreSQL",
                    "implementation": "1. Design data ingestion pipeline\n2. Set up PostgreSQL database\n3. Create Streamlit dashboard components\n4. Implement WebSocket for real-time updates\n5. Add customizable widget system",
                    "difficulty": "Beginner-Intermediate",
                    "time": "24-36 hours",
                    "impact": "Help businesses make data-driven decisions with real-time insights"
                },
                {
                    "title": "🔍 Smart GitHub Repository Analyzer",
                    "description": "Analyze any GitHub repo to provide insights on code quality, security vulnerabilities, and improvement suggestions.",
                    "tech_stack": "Python, GitHub API, AST, ML Models, Flask",
                    "implementation": "1. Integrate GitHub API for repo access\n2. Build AST parser for code analysis\n3. Implement security vulnerability detection\n4. Create ML models for quality scoring\n5. Design Flask web interface",
                    "difficulty": "Intermediate-Advanced",
                    "time": "40-48 hours",
                    "impact": "Help developers maintain better codebases through automated analysis"
                }
            ],
            "JavaScript": [
                {
                    "title": "⚡ Interactive AI Web App",
                    "description": "Create a dynamic web application with AI integration and beautiful user interface for enhanced user experience.",
                    "tech_stack": "Next.js, Node.js, AI APIs, MongoDB, Tailwind CSS",
                    "implementation": "1. Set up Next.js project with Tailwind\n2. Create Node.js API endpoints\n3. Integrate AI services (OpenAI/Gemini)\n4. Design responsive UI components\n5. Connect MongoDB for data persistence",
                    "difficulty": "Intermediate",
                    "time": "32-48 hours",
                    "impact": "Showcase modern web development skills with AI integration"
                },
                {
                    "title": "🎮 Real-time Collaboration Tool",
                    "description": "Build a live coding/collaboration platform where teams can work together in real-time with shared workspaces.",
                    "tech_stack": "React + Socket.io + Express + Redis + CodeMirror",
                    "difficulty": "Advanced",
                    "time": "40-48 hours",
                    "impact": "Enable remote team collaboration"
                }
            ],
            "Go": [
                {
                    "title": "🚀 High-Performance API Gateway",
                    "description": "Build a lightning-fast API gateway with rate limiting, authentication, and monitoring",
                    "tech_stack": "Go + Gin + Redis + PostgreSQL + Docker",
                    "difficulty": "Intermediate-Advanced",
                    "time": "36-48 hours",
                    "impact": "Handle millions of API requests efficiently"
                }
            ]
        }

        # Select projects based on user's languages
        selected_projects = []
        for lang in languages[:3]:  # Check top 3 languages
            if lang in project_templates:
                selected_projects.extend(project_templates[lang])

        # Add experience-based projects
        if repos > 20:
            selected_projects.append({
                "title": "🔍 Advanced Code Intelligence Platform",
                "description": f"With your {repos} repositories, build a sophisticated codebase analyzer that provides deep insights",
                "tech_stack": f"{primary_lang} + ML + Graph Databases + Web UI",
                "difficulty": "Advanced",
                "time": "48+ hours",
                "impact": "Transform how developers understand codebases"
            })

        # Default projects if no matches
        if not selected_projects:
            selected_projects = [
                {
                    "title": "🤖 Universal AI Assistant",
                    "description": "Start your AI journey with a versatile assistant that can help with various tasks",
                    "tech_stack": "Python + OpenAI API + Streamlit + SQLite",
                    "difficulty": "Beginner-Intermediate",
                    "time": "24-36 hours",
                    "impact": "Learn AI integration fundamentals"
                },
                {
                    "title": "📚 Smart Learning Companion",
                    "description": "Build an AI that helps developers learn new technologies through personalized recommendations",
                    "tech_stack": "Next.js + Python + Vector DB + AI APIs",
                    "difficulty": "Intermediate",
                    "time": "32-48 hours",
                    "impact": "Accelerate developer learning"
                }
            ]

        # Select 5 projects for variety
        import random
        final_projects = random.sample(selected_projects, min(5, len(selected_projects)))

        # Format projects with new simplified structure
        project_text = ""
        for i, project in enumerate(final_projects, 1):
            implementation = project.get('implementation', 'Start with core functionality, then add advanced features. Focus on MVP first.')
            project_text += f"""
{i}. 🎯 **{project['title']}**
DESC: {project['description']}
TECH: {project['tech_stack']}
IMPL: {implementation}
DIFF: {project['difficulty']}
IMPACT: {project['impact']}
TIME: {project['time']}

"""

        return f"""📊 **Profile Analysis Summary**

`{username}` demonstrates {experience.lower()} level skills with {repos} repositories and expertise in {', '.join(languages[:3]) if languages else 'multiple technologies'}. {f'Bio: {bio}' if bio else 'Ready for creative hackathon challenges.'}

🚀 **Top 5 Hackathon Project Recommendations**

{project_text}

💡 **Hackathon Strategy for {username}:**
• Leverage your {primary_lang} expertise as your foundation
• Build on patterns from your {repos} existing repositories
• Consider your {followers} followers as potential early users
• Focus on {complexity.lower()} complexity for 24-48 hour timeline

🛠️ **Recommended Tech Stack:**
• **Primary**: {primary_lang} (your strongest language)
• **AI/ML**: OpenAI API, Anthropic Claude, or Hugging Face
• **Backend**: {'FastAPI' if 'Python' in languages else 'Express.js' if 'JavaScript' in languages or 'TypeScript' in languages else 'Go Gin' if 'Go' in languages else 'Your preferred framework'}
• **Database**: PostgreSQL, MongoDB, or Vector databases
• **Deployment**: Docker, Vercel, or cloud platforms

🎯 **Success Tips:**
- Start with your strongest language ({primary_lang})
- Build something you'd actually use
- Focus on one core feature done really well
- Plan for demo-friendly functionality

Ready to build something amazing? Your {experience.lower()} level skills are perfect for these projects! 🚀"""

class AgentService:
    def __init__(self):
        self.config = AgentConfig()
        self.gateway_url = os.getenv('MCPGATEWAY_URL', 'mcp-gateway:8811')
        self.github_analyzer = GitHubAnalyzer()
        self.ai_client = AIAgentClient(self.gateway_url)
    
    async def analyze_github_profile(self, username: str, agent_name: str = "hackathon_recommender") -> AnalysisResponse:
        """Analyze a GitHub profile and generate hackathon recommendations"""
        try:
            # Validate username
            if not username or not username.strip():
                raise Exception("Username cannot be empty. Please enter a valid GitHub username.")

            username = username.strip()

            # Basic username validation
            if len(username) > 39:  # GitHub username max length
                raise Exception("Username is too long. GitHub usernames must be 39 characters or less.")

            if not username.replace('-', '').replace('_', '').isalnum():
                raise Exception("Invalid username format. GitHub usernames can only contain letters, numbers, hyphens, and underscores.")

            agent_config = self.config.agents.get(agent_name, {})
            if not agent_config:
                raise Exception(f"Agent {agent_name} not found in configuration")

            logger.info(f"Starting analysis for {username} with agent {agent_name}")

            # Get real GitHub profile data
            profile = await self.github_analyzer.get_user_profile(username)

            # Generate AI-powered personalized recommendations
            recommendations = await self.ai_client.call_agent(agent_config, profile)
            
            return AnalysisResponse(
                success=True,
                agent=agent_name,
                recommendations=recommendations,
                profile=profile
            )
            
        except Exception as e:
            logger.error(f"Analysis failed for {username}: {e}")
            return AnalysisResponse(
                success=False,
                agent=agent_name,
                error=str(e)
            )
    


# Initialize FastAPI app
app = FastAPI(
    title="AI Agents Hackathon Recommender",
    description="Backend service for analyzing GitHub profiles and recommending hackathon projects",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
agent_service = AgentService()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_profile(request: AnalysisRequest):
    """Analyze a GitHub profile and generate hackathon recommendations"""
    return await agent_service.analyze_github_profile(request.username, request.agent)

@app.get("/agents")
async def list_agents():
    """List available agents"""
    return {"agents": list(agent_service.config.agents.keys())}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Hacksy - AI Agents Hackathon Recommender API", "version": "1.0.0"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7777))  # Render uses PORT env var
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
