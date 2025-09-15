#!/bin/bash
# Start script for Professional UI

echo "Starting RELO Classifier Professional UI..."
echo "================================"
echo "Denali Advanced Integration"
echo "================================"
echo ""

# Check if python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Start a simple HTTP server for testing
echo "Starting HTTP server on port 8080..."
echo "Access the Professional UI at: http://localhost:8080/index-professional.html"
echo "Access the Test Preview at: http://localhost:8080/test-professional.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python3 -m http.server 8080