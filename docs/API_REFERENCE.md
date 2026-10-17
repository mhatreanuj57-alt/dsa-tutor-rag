# API Reference

## Base URL
\https://api.example.com\

## Authentication
All API requests require an API key in the header:
\\\
Authorization: Bearer YOUR_API_KEY
\\\

## Endpoints

### Health Check
\\\
GET /api/health
\\\
Returns server status.

**Response:**
\\\json
{
  "status": "healthy",
  "version": "1.0.0"
}
\\\

### Main Endpoint
\\\
POST /api/main
\\\
Process data.

**Request Body:**
\\\json
{
  "data": "input data"
}
\\\

**Response:**
\\\json
{
  "result": "processed data",
  "status": "success"
}
\\\

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |