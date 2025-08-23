@echo off
echo Starting PRs4Dummies RAG API Server...
echo.
echo The server will be available at:
echo - API Documentation: http://localhost:8000/docs
echo - Health Check: http://localhost:8000/health
echo - API Info: http://localhost:8000/info
echo.
echo Press Ctrl+C to stop the server
echo.
python main.py
pause
