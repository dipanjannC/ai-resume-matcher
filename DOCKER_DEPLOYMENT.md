# 🐳 Docker Deployment Guide

## Quick Deployment

### Prerequisites

- Docker installed on your system
- Docker Compose installed
- OpenAI API key

### 1-Command Deployment

```bash
# Clone and deploy
git clone <your-repo-url>
cd ai-resume-matcher
chmod +x deploy.sh
./deploy.sh
```

This will:

- ✅ Check Docker installation
- ✅ Create environment file from template
- ✅ Build Docker image
- ✅ Start the application
- ✅ Open browser to <http://localhost:8501>

## Manual Deployment

### Step 1: Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

Add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here  # optional
```

### Step 2: Build and Run

```bash
# Build the image
docker-compose build

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f
```

### Step 3: Access Application

Open your browser and navigate to: **<http://localhost:8501>**

## Docker Commands

### Basic Operations

```bash
# Start application
docker-compose up -d

# Stop application
docker-compose down

# Restart application
docker-compose restart

# View live logs
docker-compose logs -f

# View application status
docker-compose ps
```

### Development

```bash
# Build without cache
docker-compose build --no-cache

# Run in foreground (see logs)
docker-compose up

# Update and restart
docker-compose build && docker-compose up -d
```

### Maintenance

```bash
# Clean up containers
docker-compose down --volumes

# Remove images
docker rmi ai-resume-matcher_ai-resume-matcher

# Full cleanup
docker system prune -a
```

## Volume Management

The application uses volumes for persistent data:

```yaml
volumes:
  - ./data:/app/data    # Resume and job data
  - ./logs:/app/logs    # Application logs
```

### Backup Data

```bash
# Backup data directory
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Restore from backup
tar -xzf backup-20240929.tar.gz
```

## Troubleshooting

### Common Issues

**Port Already in Use**

```bash
# Check what's using port 8501
lsof -i :8501

# Use different port
docker-compose -f docker-compose.yml run --service-ports -p 8502:8501 ai-resume-matcher
```

**Permission Issues**

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

**API Key Issues**

```bash
# Check environment variables
docker-compose exec ai-resume-matcher printenv | grep API

# Update .env and restart
docker-compose restart
```

### Health Checks

```bash
# Check application health
curl http://localhost:8501/_stcore/health

# Check container status
docker-compose ps

# View detailed logs
docker-compose logs ai-resume-matcher
```

## Production Deployment

### Environment Variables

```env
# Production settings
DEBUG=false
LOG_LEVEL=WARNING
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
```

### Security

- Use secrets management for API keys
- Enable HTTPS with reverse proxy
- Limit file upload sizes
- Configure proper logging

### Scaling

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  ai-resume-matcher:
    build: .
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
    ports:
      - "8501-8503:8501"
```

## Monitoring

### Logs

```bash
# Live logs
docker-compose logs -f --tail=100

# Export logs
docker-compose logs > app-logs-$(date +%Y%m%d).log
```

### Resource Usage

```bash
# Container stats
docker stats

# Detailed resource usage
docker-compose exec ai-resume-matcher top
```

## Advanced Configuration

### Custom Dockerfile

```dockerfile
# Extend the base image
FROM ai-resume-matcher_ai-resume-matcher:latest

# Add custom configurations
COPY custom-config.py /app/
ENV CUSTOM_CONFIG=true

# Custom entrypoint
COPY entrypoint.sh /app/
ENTRYPOINT ["/app/entrypoint.sh"]
```

### Docker Compose Override

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  ai-resume-matcher:
    environment:
      - CUSTOM_SETTING=value
    volumes:
      - ./custom:/app/custom
```

## Best Practices

1. **Environment Management**: Always use .env files
2. **Data Persistence**: Mount data volumes properly
3. **Log Management**: Configure log rotation
4. **Health Monitoring**: Use health checks
5. **Security**: Keep API keys secure
6. **Updates**: Regularly update base images
7. **Backups**: Automate data backups

---

## Support

For issues with Docker deployment:

1. Check the logs: `docker-compose logs -f`
2. Verify environment variables in `.env`
3. Ensure Docker and Docker Compose are up to date
4. Check port availability: `netstat -tulpn | grep 8501`

## Quick Reference

| Command | Description |
|---------|-------------|
| `./deploy.sh` | One-command deployment |
| `docker-compose up -d` | Start application |
| `docker-compose down` | Stop application |
| `docker-compose logs -f` | View live logs |
| `docker-compose restart` | Restart services |
| `docker-compose build` | Rebuild image |
