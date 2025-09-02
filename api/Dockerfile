# file: api/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ /app/src/
COPY models/ /app/models/
COPY api/main.py /app/main.py

# Expose the port FastAPI runs on
##
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]