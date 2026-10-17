# Deployment Guide

## Prerequisites
- Python 3.10+
- pip
- Docker (optional)

## Local Development
\\\ash
git clone https://github.com/a18-n03/dsa-tutor-rag.git
cd dsa-tutor-rag
pip install -r requirements.txt
python main.py
\\\

## Docker Deployment
\\\ash
docker build -t dsa-tutor-rag .
docker run -p 8000:8000 dsa-tutor-rag
\\\

## Production Deployment

### Vercel
1. Connect your GitHub repository
2. Vercel will auto-detect the framework
3. Deploy automatically

### Render
1. Create a new Web Service
2. Connect your GitHub repository
3. Set build command: \pip install -r requirements.txt\
4. Set start command: \uvicorn main:app\

### Railway
1. Connect your GitHub repository
2. Railway will auto-detect the framework
3. Deploy automatically

## Environment Variables
Copy \.env.example\ to \.env\ and configure:
\\\
ENVIRONMENT=production
SECRET_KEY=your-secret-key
API_KEY=your-api-key
\\\

## Monitoring
- Health check endpoint: \/api/health\
- Logs: Check your hosting platform's dashboard
- Metrics: Implement Prometheus or similar