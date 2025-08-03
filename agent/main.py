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
from typing import Dict, Any, List, Optional, Union

import ssl
import certifi
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import google.generativeai as genai
from vapi import Vapi
import nest_asyncio

# Apply nest_asyncio to allow Vapi's async within FastAPI/Uvicorn
nest_asyncio.apply()

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

class AssistantConfigurationResponse(BaseModel):
    success: bool
    config: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# A model to represent the tool call from Vapi
class VapiToolCallRequest(BaseModel):
    class ToolCall(BaseModel):
        name: str
        parameters: Dict[str, Any]
    tool_call: ToolCall

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
        self.ssl_context = self._create_ssl_context()

        # Log token status for debugging (without exposing the actual token)
        if self.token:
            logger.info(f"GitHub token configured: {self.token[:12]}...")
        else:
            logger.warning("No GitHub token found - API rate limits will be lower")

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Creates an SSL context that trusts default CAs and an optional custom CA."""
        context = ssl.create_default_context(cafile=certifi.where())
        custom_ca_path = os.getenv("CUSTOM_CA_BUNDLE_PATH")
        if custom_ca_path and os.path.exists(custom_ca_path):
            logger.info(f"Loading custom CA from: {custom_ca_path}")
            context.load_verify_locations(cafile=custom_ca_path)
        return context
        
    async def get_user_profile(self, username: str) -> Dict[str, Any]:
        """Fetch real GitHub user profile data"""
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
            
        try:
            async with aiohttp.ClientSession() as session:
                # Get user basic info
                async with session.get(f"{self.base_url}/users/{username}", headers=headers, ssl=self.ssl_context) as response:
                    if response.status == 404:
                        raise Exception(f"GitHub user '{username}' not found. Please check the username and try again.")
                    elif response.status == 403:
                        raise Exception("GitHub API rate limit exceeded. Please try again in a few minutes.")
                    elif response.status != 200:
                        raise Exception(f"GitHub API error: Unable to fetch profile (Status: {response.status})")
                    user_data = await response.json()
                
                # Get user repositories with detailed info
                async with session.get(f"{self.base_url}/users/{username}/repos?per_page=100&sort=updated", headers=headers, ssl=self.ssl_context) as response:
                    if response.status == 200:
                        repos_data = await response.json()
                    else:
                        repos_data = []

                # Get user's recent activity (events)
                async with session.get(f"{self.base_url}/users/{username}/events/public?per_page=30", headers=headers, ssl=self.ssl_context) as response:
                    if response.status == 200:
                        events_data = await response.json()
                    else:
                        events_data = []

                # Analyze repositories in detail
                repo_analysis = await self._analyze_repositories(session, headers, username, repos_data[:20])  # Analyze top 20 repos

                # Analyze activity patterns
                activity_analysis = self._analyze_activity_patterns(events_data, repos_data)

                # Calculate comprehensive language statistics
                languages = {}
                language_bytes = {}
                for repo in repos_data:
                    if repo.get('language'):
                        languages[repo['language']] = languages.get(repo['language'], 0) + 1
                        # Weight by stars and size for better language ranking
                        weight = (repo.get('stargazers_count', 0) + 1) * (repo.get('size', 1) + 1)
                        language_bytes[repo['language']] = language_bytes.get(repo['language'], 0) + weight

                # Get top languages by usage and expertise
                top_languages_by_count = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:8]
                top_languages_by_bytes = sorted(language_bytes.items(), key=lambda x: x[1], reverse=True)[:8]

                # Combine and deduplicate languages
                all_languages = list(dict.fromkeys([lang[0] for lang in top_languages_by_bytes + top_languages_by_count]))[:10]

                return {
                    "username": username,
                    "name": user_data.get('name', username),
                    "bio": user_data.get('bio', ''),
                    "repos": user_data.get('public_repos', 0),
                    "followers": user_data.get('followers', 0),
                    "following": user_data.get('following', 0),
                    "languages": all_languages,
                    "company": user_data.get('company', ''),
                    "location": user_data.get('location', ''),
                    "created_at": user_data.get('created_at', ''),
                    "repository_count": len(repos_data),
                    "recent_repos": [repo['name'] for repo in repos_data[:5]],

                    # Enhanced data for better recommendations
                    "repo_analysis": repo_analysis,
                    "activity_analysis": activity_analysis,
                    "expertise_level": self._calculate_expertise_level(user_data, repos_data, activity_analysis),
                    "preferred_domains": self._extract_project_domains(repos_data),
                    "collaboration_style": self._analyze_collaboration_style(repos_data, events_data),
                    "recent_activity_score": activity_analysis.get('recent_activity_score', 0),
                    "technology_diversity": len(all_languages),
                    "project_complexity_preference": repo_analysis.get('avg_complexity', 'intermediate')
                }
                
        except Exception as e:
            logger.error(f"GitHub API error for {username}: {e}")
            # Re-raise the exception to be handled by the calling function
            # This ensures proper error messages reach the frontend
            raise e

    async def _analyze_repositories(self, session, headers, username, repos_data):
        """Analyze repositories for deeper insights"""
        analysis = {
            "total_stars": 0,
            "total_forks": 0,
            "avg_complexity": "intermediate",
            "popular_topics": [],
            "frameworks_used": [],
            "project_types": [],
            "recent_activity": False
        }

        all_topics = []
        frameworks = []
        project_types = []

        for repo in repos_data:
            # Aggregate stats
            analysis["total_stars"] += repo.get('stargazers_count', 0)
            analysis["total_forks"] += repo.get('forks_count', 0)

            # Collect topics
            if repo.get('topics'):
                all_topics.extend(repo['topics'])

            # Analyze repo characteristics
            repo_name = repo.get('name', '').lower()
            repo_desc = (repo.get('description') or '').lower()

            # Detect frameworks and project types
            if any(fw in repo_name or fw in repo_desc for fw in ['react', 'vue', 'angular']):
                frameworks.append('Frontend Framework')
            if any(fw in repo_name or fw in repo_desc for fw in ['django', 'flask', 'fastapi', 'express']):
                frameworks.append('Backend Framework')
            if any(fw in repo_name or fw in repo_desc for fw in ['ml', 'ai', 'neural', 'tensorflow', 'pytorch']):
                project_types.append('AI/ML')
            if any(fw in repo_name or fw in repo_desc for fw in ['api', 'rest', 'graphql']):
                project_types.append('API Development')
            if any(fw in repo_name or fw in repo_desc for fw in ['mobile', 'android', 'ios', 'flutter']):
                project_types.append('Mobile Development')

            # Check recent activity (updated in last 6 months)
            from datetime import datetime, timedelta
            if repo.get('updated_at'):
                try:
                    updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
                    if updated > datetime.now().replace(tzinfo=updated.tzinfo) - timedelta(days=180):
                        analysis["recent_activity"] = True
                except:
                    pass

        # Determine complexity based on stars, forks, and repo count
        avg_stars = analysis["total_stars"] / max(len(repos_data), 1)
        if avg_stars > 50 or analysis["total_forks"] > 20:
            analysis["avg_complexity"] = "advanced"
        elif avg_stars < 5 and analysis["total_forks"] < 3:
            analysis["avg_complexity"] = "beginner"

        # Get most popular topics
        from collections import Counter
        topic_counts = Counter(all_topics)
        analysis["popular_topics"] = [topic for topic, count in topic_counts.most_common(5)]

        # Get unique frameworks and project types
        analysis["frameworks_used"] = list(set(frameworks))
        analysis["project_types"] = list(set(project_types))

        return analysis

    def _analyze_activity_patterns(self, events_data, repos_data):
        """Analyze user's activity patterns"""
        analysis = {
            "recent_activity_score": 0,
            "activity_type": "moderate",
            "preferred_days": [],
            "commit_frequency": "weekly",
            "collaboration_level": "individual"
        }

        if not events_data:
            return analysis

        # Count recent events (last 30 days)
        from datetime import datetime, timedelta
        recent_events = 0
        push_events = 0
        pr_events = 0

        for event in events_data:
            try:
                event_date = datetime.fromisoformat(event['created_at'].replace('Z', '+00:00'))
                if event_date > datetime.now().replace(tzinfo=event_date.tzinfo) - timedelta(days=30):
                    recent_events += 1

                    if event['type'] == 'PushEvent':
                        push_events += 1
                    elif event['type'] in ['PullRequestEvent', 'IssuesEvent']:
                        pr_events += 1
            except:
                continue

        # Calculate activity score (0-100)
        analysis["recent_activity_score"] = min(recent_events * 3, 100)

        # Determine activity type
        if recent_events > 20:
            analysis["activity_type"] = "very_active"
        elif recent_events > 10:
            analysis["activity_type"] = "active"
        elif recent_events > 3:
            analysis["activity_type"] = "moderate"
        else:
            analysis["activity_type"] = "low"

        # Determine collaboration level
        if pr_events > push_events * 0.3:
            analysis["collaboration_level"] = "collaborative"
        elif pr_events > 0:
            analysis["collaboration_level"] = "mixed"

        return analysis

    def _calculate_expertise_level(self, user_data, repos_data, activity_analysis):
        """Calculate overall expertise level"""
        factors = {
            "account_age": 0,
            "repo_count": 0,
            "followers": 0,
            "activity": 0,
            "complexity": 0
        }

        # Account age (years on GitHub)
        try:
            from datetime import datetime
            created = datetime.fromisoformat(user_data['created_at'].replace('Z', '+00:00'))
            years = (datetime.now().replace(tzinfo=created.tzinfo) - created).days / 365
            factors["account_age"] = min(int(years * 10), 30)  # Max 30 points
        except:
            factors["account_age"] = 10

        # Repository count
        repo_count = user_data.get('public_repos', 0)
        factors["repo_count"] = min(repo_count * 2, 25)  # Max 25 points

        # Followers (social proof)
        followers = user_data.get('followers', 0)
        factors["followers"] = min(followers * 0.5, 20)  # Max 20 points

        # Recent activity
        factors["activity"] = activity_analysis.get('recent_activity_score', 0) * 0.15  # Max 15 points

        # Repository complexity (stars, forks)
        total_stars = sum(repo.get('stargazers_count', 0) for repo in repos_data[:10])
        factors["complexity"] = min(int(total_stars * 0.2), 10)  # Max 10 points

        total_score = sum(factors.values())

        if total_score > 70:
            return "expert"
        elif total_score > 40:
            return "intermediate"
        elif total_score > 15:
            return "beginner"
        else:
            return "newcomer"

    def _extract_project_domains(self, repos_data):
        """Extract preferred project domains from repositories"""
        domains = []

        for repo in repos_data[:15]:  # Check top 15 repos
            name = (repo.get('name') or '').lower()
            desc = (repo.get('description') or '').lower()
            topics = repo.get('topics', [])

            # Web development
            if any(term in name or term in desc for term in ['web', 'website', 'frontend', 'backend', 'fullstack']):
                domains.append('Web Development')

            # Data science / AI
            if any(term in name or term in desc for term in ['data', 'ml', 'ai', 'machine', 'neural', 'analysis']):
                domains.append('Data Science & AI')

            # Mobile development
            if any(term in name or term in desc for term in ['mobile', 'android', 'ios', 'app', 'flutter', 'react-native']):
                domains.append('Mobile Development')

            # DevOps / Infrastructure
            if any(term in name or term in desc for term in ['docker', 'kubernetes', 'ci', 'cd', 'deploy', 'infrastructure']):
                domains.append('DevOps & Infrastructure')

            # Gaming
            if any(term in name or term in desc for term in ['game', 'unity', 'pygame', 'gaming']):
                domains.append('Game Development')

            # Blockchain
            if any(term in name or term in desc for term in ['blockchain', 'crypto', 'web3', 'smart', 'contract']):
                domains.append('Blockchain & Web3')

            # Check topics for additional domains
            for topic in topics:
                topic_lower = topic.lower()
                if topic_lower in ['machine-learning', 'artificial-intelligence', 'deep-learning']:
                    domains.append('Data Science & AI')
                elif topic_lower in ['web-development', 'frontend', 'backend']:
                    domains.append('Web Development')
                elif topic_lower in ['mobile', 'android', 'ios']:
                    domains.append('Mobile Development')

        # Return unique domains, most common first
        from collections import Counter
        domain_counts = Counter(domains)
        return [domain for domain, count in domain_counts.most_common(5)]

    def _analyze_collaboration_style(self, repos_data, events_data):
        """Analyze collaboration preferences"""
        forked_repos = sum(1 for repo in repos_data if repo.get('fork', False))
        original_repos = len(repos_data) - forked_repos

        # Count collaboration events
        collab_events = sum(1 for event in events_data if event.get('type') in ['PullRequestEvent', 'IssuesEvent', 'ForkEvent'])

        if forked_repos > original_repos * 0.5 or collab_events > 5:
            return "collaborative"
        elif forked_repos > 0 or collab_events > 0:
            return "mixed"
        else:
            return "independent"

class AIAgentClient:
    """Client for communicating with AI models via Gemini API"""

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url
        # Configure Gemini API
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if self.gemini_api_key:
            genai.configure(
                api_key=self.gemini_api_key,
                transport='rest'
            )
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

            # Try Gemini API
            if self.model:
                try:
                    return await self._call_gemini(prompt, temperature, max_tokens)
                except Exception as gemini_error:
                    logger.error(f"Gemini call failed: {gemini_error}")
                    # Re-raise with user-friendly message
                    if "API key" in str(gemini_error).lower() or "authentication" in str(gemini_error).lower():
                        raise Exception("AI service authentication failed. Please check the API key configuration.")
                    elif "rate limit" in str(gemini_error).lower() or "quota" in str(gemini_error).lower():
                        raise Exception("AI service rate limit exceeded. Please try again in a few minutes.")
                    elif "network" in str(gemini_error).lower() or "connection" in str(gemini_error).lower():
                        raise Exception("Unable to connect to AI service. Please check your internet connection and try again.")
                    else:
                        raise Exception(f"AI service temporarily unavailable. Please try again later.")
            else:
                raise Exception("AI service is not configured. Please check your Gemini API key configuration.")

        except Exception as e:
            logger.error(f"AI agent call failed: {e}")
            # Re-raise the exception to be handled by the calling function
            raise e

    def _create_agent_prompt(self, instructions: str, profile_data: Dict[str, Any]) -> str:
        """Create a detailed prompt for the AI agent with comprehensive profile analysis"""
        username = profile_data.get('username', 'Unknown')
        languages = profile_data.get('languages', [])
        repos = profile_data.get('repos', 0)
        followers = profile_data.get('followers', 0)
        bio = profile_data.get('bio', '')
        company = profile_data.get('company', '')
        recent_repos = profile_data.get('recent_repos', [])

        # Enhanced data
        repo_analysis = profile_data.get('repo_analysis', {})
        activity_analysis = profile_data.get('activity_analysis', {})
        expertise_level = profile_data.get('expertise_level', 'intermediate')
        preferred_domains = profile_data.get('preferred_domains', [])
        collaboration_style = profile_data.get('collaboration_style', 'mixed')
        recent_activity_score = profile_data.get('recent_activity_score', 0)
        project_complexity_preference = profile_data.get('project_complexity_preference', 'intermediate')

        prompt = f"""
{instructions}

COMPREHENSIVE GITHUB PROFILE ANALYSIS:

👤 BASIC INFO:
Username: {username}
Bio: {bio}
Company: {company}
Public Repositories: {repos}
Followers: {followers}

💻 TECHNICAL PROFILE:
Primary Languages: {', '.join(languages[:8])}
Expertise Level: {expertise_level}
Technology Diversity: {len(languages)} different languages
Recent Repositories: {', '.join(recent_repos[:5])}

🏆 REPOSITORY ANALYSIS:
Total Stars Earned: {repo_analysis.get('total_stars', 0)}
Total Forks: {repo_analysis.get('total_forks', 0)}
Project Complexity Preference: {project_complexity_preference}
Popular Topics: {', '.join(repo_analysis.get('popular_topics', [])[:5])}
Frameworks Used: {', '.join(repo_analysis.get('frameworks_used', []))}
Project Types: {', '.join(repo_analysis.get('project_types', []))}

📊 ACTIVITY PATTERNS:
Recent Activity Score: {recent_activity_score}/100
Activity Type: {activity_analysis.get('activity_type', 'moderate')}
Collaboration Style: {collaboration_style}
Recent Activity: {'Yes' if repo_analysis.get('recent_activity', False) else 'No'}

🎯 PREFERRED DOMAINS:
{', '.join(preferred_domains) if preferred_domains else 'General Development'}

TASK:
Based on this comprehensive GitHub profile analysis, generate 5 highly personalized and creative hackathon project recommendations that:

1. Match the user's expertise level ({expertise_level})
2. Leverage their strongest languages: {', '.join(languages[:3])}
3. Align with their preferred domains: {', '.join(preferred_domains[:3]) if preferred_domains else 'versatile projects'}
4. Consider their collaboration style: {collaboration_style}
5. Match their project complexity preference: {project_complexity_preference}
6. Build upon their existing experience with: {', '.join(repo_analysis.get('frameworks_used', [])[:3])}

Make each recommendation unique and exciting, considering their {recent_activity_score}/100 activity score and {repo_analysis.get('total_stars', 0)} total stars earned.

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

class VapiClient:
    """Client for handling voice interactions with Vapi."""
    def __init__(self):
        self.api_key = os.getenv('VAPI_API_KEY')
        self.public_agents_url = os.getenv('PUBLIC_AGENTS_URL')
        
        if not all([self.api_key, self.public_agents_url]):
            logger.warning("Vapi environment variables (VAPI_API_KEY, PUBLIC_AGENTS_URL) are not fully configured. Voice features will be disabled.")
            self.client = None
        else:
            # We don't need to make API calls from the backend for web-based calls,
            # but we can keep the client initialization for future use.
            self.client = Vapi(self.api_key)
            logger.info("Vapi client configured successfully for generating assistant configurations.")

    def get_assistant_config(self) -> Dict[str, Any]:
        """Constructs the assistant configuration for Vapi calls."""
        if not self.public_agents_url:
             raise Exception("Vapi service is not configured due to missing PUBLIC_AGENTS_URL.")

        # The URL Vapi will call when the tool is invoked by the assistant.
        tool_callback_url = f"{self.public_agents_url}/vapi-tool-call/analyze"

        return {
            "model": {
                "provider": "openai", # Vapi often uses OpenAI's function calling standard
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are Hacksy, a friendly and expert AI assistant that helps developers brainstorm hackathon project ideas over a voice call.

Your process is:
1. Greet the user and ask for their GitHub username.
2. Once you have the username, use the `get_hackathon_recommendations` tool to analyze their profile.
3. When the analysis is complete, you will receive a summary of their skills.
4. Based on this summary, start a conversation. Suggest one project idea at a time. Be engaging and ask for their feedback on the ideas.
5. Keep your responses concise and conversational.
"""
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_hackathon_recommendations",
                            "description": "Analyzes a GitHub username to get a summary of the user's skills and expertise to brainstorm hackathon ideas.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "username": {
                                        "type": "string",
                                        "description": "The GitHub username to analyze, e.g., 'torvalds'."
                                    }
                                },
                                "required": ["username"]
                            },
                            "url": tool_callback_url, # The public endpoint Vapi will POST to
                            "method": "POST"
                        }
                    }
                ]
            },
            "voice": "jennifer-playht",
            "firstMessage": "Hey! I'm Hacksy, your personal hackathon brainstorming partner. To get started, could you please tell me your GitHub username?"
        }


class AgentService:
    def __init__(self):
        self.config = AgentConfig()
        self.gateway_url = os.getenv('MCPGATEWAY_URL', 'mcp-gateway:8811')
        self.github_analyzer = GitHubAnalyzer()
        self.ai_client = AIAgentClient(self.gateway_url)
        self.vapi_client = VapiClient()
    
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

    def get_brainstorming_config(self) -> AssistantConfigurationResponse:
        """Provides the assistant configuration for a web-based Vapi session."""
        try:
            config = self.vapi_client.get_assistant_config()
            return AssistantConfigurationResponse(success=True, config=config)
        except Exception as e:
            logger.error(f"Failed to get brainstorming session config: {e}")
            return AssistantConfigurationResponse(success=False, error=str(e))
    


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

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy")

@app.post("/analyze", response_model=AnalysisResponse, tags=["Core"])
async def analyze_profile(request: AnalysisRequest):
    """Analyze a GitHub profile and generate hackathon recommendations"""
    return await agent_service.analyze_github_profile(request.username, request.agent)

@app.get("/agents", tags=["General"])
async def list_agents():
    """List available agents"""
    return {"agents": list(agent_service.config.agents.keys())}

@app.get("/brainstorm/config", response_model=AssistantConfigurationResponse, tags=["Voice AI"])
async def get_brainstorm_config():
    """
    Provides the full assistant configuration for the frontend to initialize a web-based Vapi call.
    """
    return agent_service.get_brainstorming_config()

@app.post("/vapi-tool-call/analyze", tags=["Voice AI"])
async def handle_vapi_tool_call(request: VapiToolCallRequest):
    """
    Webhook endpoint for Vapi to call when a tool is invoked.
    This should not be called directly by users.
    """
    tool_name = request.tool_call.name
    parameters = request.tool_call.parameters
    
    if tool_name == "get_hackathon_recommendations":
        username = parameters.get("username")
        if not username:
            return {"tool_response": "I'm sorry, I didn't catch the username. Could you please repeat it?"}
        
        logger.info(f"Vapi tool call received for username: {username}")
        
        # Run the existing analysis logic
        analysis_result = await agent_service.analyze_github_profile(username)
        
        if analysis_result.success and analysis_result.profile:
            # If successful, create a concise summary for the AI to use in conversation.
            profile = analysis_result.profile
            summary = (
                f"Okay, I've analyzed the GitHub profile for {profile.get('name', username)}. "
                f"They seem to be an {profile.get('expertise_level', 'intermediate')} developer. "
                f"Their top languages are {', '.join(profile.get('languages', [])[:3])}. "
                f"They have experience in domains like {', '.join(profile.get('preferred_domains', ['general development'])[:2])}. "
                f"Now, let's brainstorm some ideas based on that. How about we start with a project related to {profile.get('preferred_domains', ['web development'])[0]}?"
            )
            return {"tool_response": summary}
        else:
            # If analysis fails, inform the AI so it can tell the user.
            error_message = analysis_result.error or "I couldn't analyze that profile for some reason."
            return {"tool_response": f"I'm sorry, I ran into an issue. {error_message} Could you try a different username?"}
    
    return {"tool_response": "Unknown tool called."}

@app.get("/", tags=["General"])
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
