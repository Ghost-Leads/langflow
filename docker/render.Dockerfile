FROM python:3.11-slim

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libpq-dev \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean

# Set the working directory
WORKDIR /app

# Copy all files
COPY . .

# Install uv and backend dependencies
RUN pip install uv
RUN uv pip install --system -e ".[postgresql]"

# Build frontend (builds into src/frontend/build)
WORKDIR /app/src/frontend
RUN npm install && npm run build

# Return to app root
WORKDIR /app

# Expose LangFlow's default port
EXPOSE 7860

# Start LangFlow
ENTRYPOINT ["python", "-m", "langflow", "run"]
