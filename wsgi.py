"""
WSGI entry point for Gunicorn production server.

This module exports the Flask application for use with Gunicorn.
"""
from server import app

if __name__ == "__main__":
    app.run()
