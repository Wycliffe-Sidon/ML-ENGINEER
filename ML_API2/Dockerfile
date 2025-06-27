# Use the base Python image
FROM python:3.13.3

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Define the command to run your application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8001"]