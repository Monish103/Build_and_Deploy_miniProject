FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for pdf processing
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose Dash app port
EXPOSE 8050

# Run the web app
CMD ["python", "app.py"]
