# Docker Setup for Unmute Backend

This guide explains how to run the Unmute backend using Docker.

## Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (usually included with Docker Desktop)
- Gemini API key

## Quick Start

### 1. Set up environment variables

Create a `.env` file in the root directory:

```bash
# For root .env file
cat > .env << 'EOF'
GEMINI_API_KEY=your_actual_api_key_here
API_BASE_URL=http://localhost:8000
EOF
```

Also create `backend/.env` for backend-specific configs:

```bash
# For backend/.env file
cat > backend/.env << 'EOF'
GEMINI_API_KEY=your_actual_api_key_here
EOF
```

**Environment Variables:**
- `GEMINI_API_KEY` - Your Gemini API key (required)
- `API_BASE_URL` - Base URL for API testing scripts (default: http://localhost:8000)

### 2a. Build and run with Docker Compose (Recommended)

```bash
docker-compose up --build
```

The backend will be available at `http://localhost:8000`

### 2b. Build and run with Docker directly (without docker-compose)

**Build the image:**
```bash
docker build -t unmute-backend .
```

**Run the container:**
```bash
docker run -d \
  --name unmute-backend \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key_here \
  -v "$(pwd)/backend:/app/backend" \
  -v "$(pwd)/sgsl_dataset:/app/sgsl_dataset:ro" \
  -v "$(pwd)/sgsl_processed:/app/sgsl_processed:ro" \
  unmute-backend
```

**Or run interactively (see logs):**
```bash
docker run -it \
  --name unmute-backend \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key_here \
  -v "$(pwd)/backend:/app/backend" \
  -v "$(pwd)/sgsl_dataset:/app/sgsl_dataset:ro" \
  -v "$(pwd)/sgsl_processed:/app/sgsl_processed:ro" \
  unmute-backend
```

**Using environment file:**
```bash
docker run -d \
  --name unmute-backend \
  -p 8000:8000 \
  --env-file backend/.env \
  -v "$(pwd)/backend:/app/backend" \
  -v "$(pwd)/sgsl_dataset:/app/sgsl_dataset:ro" \
  -v "$(pwd)/sgsl_processed:/app/sgsl_processed:ro" \
  unmute-backend
```

### 3. Run in detached mode (background)

**With docker-compose:**
```bash
docker-compose up -d
```

**With Docker directly:**
```bash
# Already covered above with -d flag
```

### 4. View logs

```bash
docker-compose logs -f backend
```

### 5. Stop the backend

```bash
docker-compose down
```

## Alternative: Using Docker directly

### Build the image

```bash
docker build -t unmute-backend .
```

### Run the container

```bash
docker run -d \
  --name unmute-backend \
  -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key_here \
  -v $(pwd)/sgsl_dataset:/app/sgsl_dataset:ro \
  -v $(pwd)/sgsl_processed:/app/sgsl_processed:ro \
  unmute-backend
```

### View logs

```bash
docker logs -f unmute-backend
```

### Stop the container

```bash
docker stop unmute-backend
docker rm unmute-backend
```

## API Endpoints

Once running, you can access:

- **Health check**: http://localhost:8000/health
- **API docs**: http://localhost:8000/docs
- **Translation**: http://localhost:8000/api/translate

## Development Mode

For development with hot reload, uncomment the `command` override in `docker-compose.yml`:

```yaml
command: uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Then run:
```bash
docker-compose up
```

Changes to the backend code will automatically reload the server.

## Troubleshooting

### Port already in use

If port 8000 is already in use, change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Use 8001 on host instead
```

### API key not working

Make sure your `.env` file has the correct API key and no extra spaces:
```
GEMINI_API_KEY=AIza...
```

### Container fails to start

Check logs:
```bash
docker-compose logs backend
```

### Cannot access static files

Ensure the `sgsl_dataset` and `sgsl_processed` directories exist and are mounted correctly.

## Production Deployment

For production, consider:

1. **Using a reverse proxy** (nginx, Caddy) with SSL/TLS
2. **Setting resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```
3. **Using a production-ready server** instead of uvicorn default workers
4. **Setting up logging** and monitoring
5. **Using environment-specific .env files**

## Cleaning Up

Remove all containers, images, and volumes:

```bash
docker-compose down -v
docker rmi unmute-backend
```
