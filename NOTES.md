# Development Notes

## Architecture
- Backend: Python/FastAPI
- Frontend: HTML/CSS/JS
- Database: SQLite/Redis

## Key Decisions
- Using FastAPI for async support
- Serverless deployment on Vercel
- In-memory storage for simplicity

## Performance Considerations
- Keep response times under 100ms
- Use caching where possible
- Optimize database queries

## Security Notes
- Use environment variables for secrets
- Validate all user input
- Rate limiting recommended