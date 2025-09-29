#!/bin/bash

# AI Resume Matcher - Easy Deployment Script
# This script sets up and runs the AI Resume Matcher using Docker

set -e

echo "🚀 AI Resume Matcher - Easy Deployment"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file from .env.example"
        echo "📝 Please edit .env file and add your API keys:"
        echo "   - OPENAI_API_KEY=your_openai_key_here"
        echo "   - GROQ_API_KEY=your_groq_key_here (optional)"
        echo ""
        read -p "Press Enter after updating the .env file..."
    else
        echo "❌ No .env.example file found. Please create .env file manually."
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/{resumes,jobs,vectordb,results,temp}
mkdir -p logs

# Create .gitkeep files to preserve directory structure
touch data/resumes/.gitkeep
touch data/jobs/.gitkeep
touch data/vectordb/.gitkeep
touch data/results/.gitkeep
touch data/temp/.gitkeep

echo "🔧 Building Docker image..."
docker-compose build

echo "🚀 Starting AI Resume Matcher..."
docker-compose up -d

echo ""
echo "✅ AI Resume Matcher is now running!"
echo "🌐 Open your browser and go to: http://localhost:8501"
echo ""
echo "📋 Useful commands:"
echo "   Stop the application:     docker-compose down"
echo "   View logs:               docker-compose logs -f"
echo "   Restart the application: docker-compose restart"
echo "   Update the application:  docker-compose build && docker-compose up -d"
echo ""
echo "💡 To stop the application, run: docker-compose down"