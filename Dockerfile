# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy your Flask app code
COPY flask_app/ /app/

# Copy model artifacts
# COPY saved_models/model.pkl /app/saved_models/model.pkl

# Copy config files if needed
COPY config/schema.yaml /app/config/schema.yaml

# Install only the requirements.txt from flask_app
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose Flask port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]

#local
# CMD ["python", "app.py"]

# Production entry point
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
