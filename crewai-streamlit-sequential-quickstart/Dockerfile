# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN poetry env use python3.10 && poetry install --only main

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py"]
