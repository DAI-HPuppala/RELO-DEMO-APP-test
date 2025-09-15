#!/bin/bash

echo "======================================"
echo "RELO Classifier - Professional UI Test"
echo "Denali Advanced Integration"
echo "======================================"
echo ""

# Check if backend is running
echo "Checking backend status..."
if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✓ Backend is running"
else
    echo "✗ Backend is not running"
    echo "Starting backend..."
    cd backend
    source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate
    python src/api/main.py &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
    sleep 3
fi

# Start frontend server
echo ""
echo "Starting frontend server..."
cd frontend
echo ""
echo "Available UIs:"
echo "1. Professional UI: http://localhost:8080/index-professional.html"
echo "2. Integrated UI: http://localhost:8080/index-integrated.html"  
echo "3. Original UI: http://localhost:8080/index-v1-original.html"
echo "4. Test Preview: http://localhost:8080/test-professional.html"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 -m http.server 8080