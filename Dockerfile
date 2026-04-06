FROM python:3.11-slim

LABEL maintainer="CalendarSchedulingEnv Team"
LABEL description="OpenEnv Calendar Scheduling Environment — HTTP Server"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn httpx openai

# Copy package and server
COPY scheduling_env/ ./scheduling_env/
COPY app.py .
COPY openenv.yaml .

# Verify environment loads
RUN python -c "from scheduling_env import CalendarEnv; print('OK')"

# Expose port used by HF Spaces
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
