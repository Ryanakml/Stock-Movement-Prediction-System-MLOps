# file: api/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

COPY./src /app/src
COPY./models /app/models
COPY./api/main.py /app/main.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
