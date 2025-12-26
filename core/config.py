import os

from dotenv import load_dotenv

# Load environment variables from .env file for local development
# In production, Azure will provide the environment variables
load_dotenv()
if os.getenv("DEBUG") is not None:
    DEBUG = os.getenv("DEBUG").lower() in ("true", "1", "yes")
else:
    DEBUG = False
# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# Email configuration
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

