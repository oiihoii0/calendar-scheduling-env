FROM python:3.11-slim

LABEL maintainer="CalendarSchedulingEnv Team"
LABEL description="OpenEnv Calendar Scheduling Environment — HTTP Server"

WORKDIR /app

# Install only what the server needs to run
RUN pip install --no-cache-dir \
    gymnasium==0.29.1 \
    numpy==1.26.4 \
    python-dateutil==2.9.0 \
    typing-extensions==4.15.0 \
    fastapi==0.135.2 \
    uvicorn==0.42.0 \
    httpx==0.28.1 \
    openai==2.30.0 \
    pydantic==2.12.5

# Copy package and server
COPY scheduling_env/ ./scheduling_env/
COPY app.py .
COPY openenv.yaml .
COPY server/ ./server/

# Verify environment loads
RUN python -c "from scheduling_env import CalendarEnv; print('CalendarEnv OK')"
RUN python -c "from scheduling_env.grader import grade_schedule; print('Grade schedule OK')"
RUN python -c "from app import app; print('FastAPI app OK')"

# Expose HF Spaces port
EXPOSE 7860

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
