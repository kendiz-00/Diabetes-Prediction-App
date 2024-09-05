# Use the official Python image from the slim-buster distribution
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Run the application using Gunicorn, binding to all interfaces on port 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "run:app"]
