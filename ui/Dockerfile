# file: ui/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY ui/requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

COPY ui/app.py.
COPY data/processed/AAPL_final_dataset.csv

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
