FROM registry.access.redhat.com/ubi8/python-38

# Install system dependencies as root
USER root
RUN yum install -y git-lfs && \
    git lfs install && \
    yum clean all

# Upgrade pip and pre-install heavy libraries that don't change often
COPY requirements.txt .
RUN pip install -U pip
RUN pip install bitsandbytes huggingface_hub

# Install the rest of the Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Copy the application source code
COPY . .

# Copy the entrypoint script and make sure it's executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch back to non-root user
USER 1001

# Use the entrypoint script to login to Hugging Face and then run the application
ENTRYPOINT ["/entrypoint.sh"]

# Default command runs the Python application
CMD ["python", "app.py"]
