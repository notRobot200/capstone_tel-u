# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory in the container
COPY . /app/

# Download NLTK data
RUN python -m nltk.downloader stopwords

# Download spacy model
RUN python -m spacy download en_core_web_lg

# Expose the port Flask is running on
EXPOSE 8080

# Define the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
