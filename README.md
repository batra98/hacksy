![AI Agents](https://img.shields.io/badge/AI-Agents-blue) ![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-orange) ![Next.js](https://img.shields.io/badge/Next.js-13-black) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![Docker](https://img.shields.io/badge/Docker-Compose-blue) ![MCP](https://img.shields.io/badge/MCP-Protocol-green)

# Hacksy - AI Agents Hackathon Project Recommender

> **Discover your perfect hackathon project with AI-powered GitHub profile analysis using Google Gemini**

An intelligent multi-agent system that analyzes real GitHub profiles and generates truly personalized hackathon project recommendations using Google's Gemini 1.5 Flash AI, agentic workflows, and real-time data integration.




## ✨ Features

- **🤖 True AI-Powered Recommendations**: Google Gemini 1.5 Flash generates unique, creative project ideas
- **📊 Real GitHub Analysis**: Live integration with GitHub API for authentic profile data
- **🎯 Personalized Suggestions**: Tailored to programming languages, experience level, and repository history
- **🏗️ Multi-Agent Architecture**: Specialized agents with MCP protocol integration
- **🔒 Secure & Scalable**: Docker microservices with environment-based secrets management
- **⚡ Modern Tech Stack**: Next.js UI + Python agents + Gemini AI + Docker orchestration
- **🚀 Production Ready**: Comprehensive error handling, fallbacks, and monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js UI    │───▶│  Python Agents  │───▶│   MCP Gateway   │
│   (Port 3003)   │    │   (Port 7777)   │    │   (Port 8811)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       ▼
         │                       │              ┌─────────────────┐
         │                       │              │  GitHub API     │
         │                       │              │  DuckDuckGo     │
         │                       │              │  Fetch Tools    │
         │                       │              │     │
         │                       │              └─────────────────┘
         │                       ▼
         │              ┌─────────────────┐
         │              │  Gemini 1.5     │
         │              │  Flash API      │
         │              │  (Google AI)    │
         │              └─────────────────┘
         ▼
┌─────────────────┐
│  GitHub User    │
│   Profile       │
│   Analysis      │
└─────────────────┘
```

### 🔄 **How It Works:**
1. **User Input** → GitHub username via web interface
2. **Profile Analysis** → Real GitHub API data (repos, languages, activity)
3. **AI Processing** → Gemini generates personalized project recommendations
4. **Smart Fallbacks** → Intelligent responses when AI is unavailable
5. **Results** → Unique hackathon project ideas with tech stacks & implementation details

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- GitHub Personal Access Token ([Get one here](https://github.com/settings/tokens))
- Google Gemini API Key ([Get one here](https://aistudio.google.com/app/apikey))
- Vapi API Key for web-based voice integration ([Get one here](https://vapi.ai/dashboard))
- A public URL for the agent service (e.g., using ngrok) for Vapi callbacks during local development.

### 1. Clone & Setup

```bash
git clone https://github.com/batra98/hacksy
cd hacksy
```

### 2. Configure API Keys

Create environment files with your API keys:

```bash
# Create .mcp.env for GitHub access
echo "github.personal_access_token=YOUR_GITHUB_TOKEN_HERE" > .mcp.env

# Create .env for Gemini AI
echo "GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE" > .env

# Add Vapi and public URL to .env for voice features
echo "VAPI_API_KEY=YOUR_VAPI_API_KEY_HERE" >> .env
echo "PUBLIC_AGENTS_URL=http://your-ngrok-url.ngrok.io" >> .env
```



### 3. Launch the Application

```bash
# Build and start all services
docker compose up -d --build

# Monitor the logs (optional)
docker compose logs -f agents
```

### 4. Access the Application

- **🌐 Web Interface**: http://localhost:3003
- **🤖 Agents API**: http://localhost:7777
- **❤️ Health Check**: http://localhost:7777/health

## 🎯 Usage

1. **🌐 Open** the web interface at http://localhost:3003
2. **👤 Enter** any GitHub username (e.g., "microsoft", "torvalds", "batra98")
3. **🚀 Click** "Get Recommendations"
4. **✨ Receive** AI-generated, personalized hackathon project suggestions!

### 🎭 **Example Results:**

**For Linus Torvalds (C/Systems):**
- 🐧 Linux Kernel Module Development
- ⚡ Real-time System Performance Monitor
- 🔍 Network Scanner in C

**For Dan Abramov (JavaScript/React):**
- 🎨 CSS Animation Playground
- 🤝 Real-time Collaborative Code Editor
- 📊 Data Visualization Tool

## 🔧 Development

### Tech Stack Details

- **🎨 Frontend**: Next.js 13, React 18, TypeScript
- **🤖 Backend**: Python 3.11, FastAPI, Pydantic
- **🧠 AI**: Google Gemini 1.5 Flash API
- **🔗 Integration**: Model Context Protocol (MCP)
- **📦 Deployment**: Docker Compose, multi-stage builds
- **🔒 Security**: Environment-based secrets, non-root containers

### API Endpoints

#### 🤖 Agents Service (Port 7777)

- `GET /health` - Health check
- `POST /analyze` - Analyze GitHub profile with AI
- `GET /agents` - List available agents
- `GET /` - API information

```bash
# Example: Get AI recommendations for a user
curl -X POST http://localhost:7777/analyze \
  -H "Content-Type: application/json" \
  -d '{"username": "torvalds", "agent": "hackathon_recommender"}'
```

### Local Development

```bash
# Start services in development mode
docker compose up --build

# Watch specific service logs
docker compose logs -f agents     # AI agents service
docker compose logs -f agents-ui  # Next.js frontend
docker compose logs -f mcp-gateway # MCP protocol gateway
```

## 🧠 AI Agent System

### Multi-Agent Architecture

**🎯 Hackathon Recommender Agent**
- Generates creative, personalized project ideas
- Considers skill level, languages, and experience
- Provides implementation guidance and tech stacks

**🔍 GitHub Analyzer Agent**
- Analyzes repository patterns and activity
- Extracts programming language preferences
- Assesses developer experience and interests

### Intelligent Features

- **🎨 Creative Generation**: Each recommendation is unique and tailored
- **📊 Profile Analysis**: Real GitHub data drives personalization
- **🛡️ Smart Fallbacks**: Graceful degradation when AI is unavailable
- **⚡ Fast Response**: Optimized prompts for quick generation
- **🔄 Continuous Learning**: Agent configurations can be updated

## 🌟 Why Hacksy?

- **🚀 Perfect for Hackathons**: Solves a real problem developers face
- **🤖 Cutting-Edge AI**: Demonstrates modern agentic AI architecture
- **🏗️ Production Ready**: Scalable, secure, and well-architected
- **📈 Impressive Demo**: Shows dramatically different results for different users
- **🛠️ Technical Depth**: MCP protocol, microservices, AI integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini** for powerful AI capabilities
- **Model Context Protocol** for secure agent integration
- **Docker** for containerization excellence
- **FastAPI & Next.js** for modern web architecture

---

**Built with ❤️ for the hackathon community by [batra98](https://github.com/batra98)**

*Ready to find your perfect hackathon project? Let Hacksy's AI agents guide you! 🚀*