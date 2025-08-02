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
    def __init__(self, config_path: str = "/agents.yaml"):
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
                    if response.status != 200:
                        raise Exception(f"GitHub API error: {response.status}")
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
            # Fallback to basic data
            return {
                "username": username,
                "repos": 0,
                "languages": [],
                "followers": 0,
                "following": 0,
                "error": str(e)
            }

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
Based on this GitHub profile analysis, generate 5-7 creative and personalized hackathon project recommendations. Each recommendation should:

1. Be tailored to the developer's skill level and language preferences
2. Be feasible to complete in 24-48 hours during a hackathon
3. Include specific technical implementation details
4. Suggest appropriate tech stacks
5. Be innovative and solve real problems

Format your response as a detailed markdown-style recommendation with:
- Project titles with emojis
- Clear descriptions
- Technical stack suggestions
- Implementation approach
- Difficulty assessment

Make each recommendation unique and creative, not generic templates.
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
        projects = []

        # Language-specific projects
        if "Python" in languages:
            projects.extend([
                "🐍 **AI-Powered Development Assistant**\n   Create an intelligent coding companion with {primary_lang} expertise\n   Stack: Python + OpenAI API + FastAPI + Vector DB",
                "📊 **Smart Analytics Dashboard**\n   Build a real-time data visualization platform\n   Stack: Python + Streamlit + Pandas + PostgreSQL",
                "🤖 **Automated Code Reviewer**\n   Develop an AI system that reviews and suggests code improvements\n   Stack: Python + AST + ML Models + GitHub API"
            ])

        if "JavaScript" in languages or "TypeScript" in languages:
            projects.extend([
                "⚡ **Interactive AI Web App**\n   Create a dynamic web application with AI integration\n   Stack: Next.js + Node.js + AI APIs + MongoDB",
                "🎮 **Real-time Collaboration Tool**\n   Build a live coding/collaboration platform\n   Stack: React + Socket.io + Express + Redis",
                "📱 **Progressive Web Assistant**\n   Develop a PWA that helps developers with daily tasks\n   Stack: TypeScript + Service Workers + IndexedDB"
            ])

        if "Go" in languages:
            projects.extend([
                "🚀 **High-Performance API Gateway**\n   Build a lightning-fast microservice architecture\n   Stack: Go + Docker + Kubernetes + Redis",
                "🔧 **Developer CLI Toolkit**\n   Create powerful command-line tools for developers\n   Stack: Go + Cobra + Viper + SQLite"
            ])

        # Experience-based projects
        if repos > 20:
            projects.append("🔍 **Advanced Code Intelligence Platform**\n   With your {repos} repositories, build a sophisticated codebase analyzer\n   Stack: {primary_lang} + ML + Graph Databases + Web UI")

        # Default projects if no matches
        if not projects:
            projects = [
                "🤖 **Universal AI Assistant**\n   Start your AI journey with a versatile assistant\n   Stack: Python + OpenAI API + Streamlit + SQLite",
                "📚 **Smart Learning Companion**\n   Build an AI that helps developers learn new technologies\n   Stack: Next.js + Python + Vector DB + AI APIs",
                "🛠️ **Developer Productivity Suite**\n   Create tools that enhance developer workflow\n   Stack: Your preferred language + APIs + Database"
            ]

        # Select 3-5 projects randomly for variety
        import random
        selected_projects = random.sample(projects, min(5, len(projects)))

        project_text = ""
        for i, project in enumerate(selected_projects, 1):
            project_text += f"{i}. {project.format(primary_lang=primary_lang, repos=repos)}\n\n"

        return f"""🏆 AI Agents Hackathon Project Recommendations for {username}

📊 **Profile Analysis:**
• GitHub: {repos} repositories, {followers} followers
• Languages: {', '.join(languages[:3]) if languages else 'Multiple technologies'}
• Experience: {experience} level
• Recent work: {', '.join(recent_repos[:3]) if recent_repos else 'Various projects'}
{f'• Bio: {bio[:60]}...' if bio else ''}

🚀 **Personalized Project Ideas ({complexity} Difficulty):**

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
