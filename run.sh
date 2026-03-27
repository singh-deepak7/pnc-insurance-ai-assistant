#!/bin/bash

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd)

echo "Starting backend..."
cd backend
uvicorn app.main:app --reload &

sleep 3

echo "Starting frontend..."
cd ../frontend
python app.py