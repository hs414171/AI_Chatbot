#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv ~/.venvs/rag_agent

# Activate the virtual environment
echo "Activating virtual environment..."
source ~/.venvs/rag_agent/bin/activate

# Install libraries from requirements.txt 
echo "Installing libraries from requirements.txt..."
pip install -r requirements.txt

# Copy env backup to .env file
if [ -f ".env" ]; then
    echo "Copying .env to .env..."
    cp .env .env
else
    echo "No .env file found. Creating a new .env file..."
    touch .env
fi

# Prompt user to fill the .env file
echo "Please fill in the .env file with the necessary environment variables."

echo "Setup completed successfully!"

