# Use an official TensorFlow Docker image as the base image
FROM tensorflow/tensorflow:latest

# Install any necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-opencv

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the Python requirements file
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project files and folders into the container
COPY . .

# Command to run your Python application
CMD ["python", "app.py"]
