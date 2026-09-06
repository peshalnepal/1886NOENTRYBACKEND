"""Process-wide configuration read from the environment.

`load_dotenv()` runs on import so a local `.env` is picked up; in production the
platform supplies the variables directly and the call is a no-op.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "").strip().lower() in ("true", "1", "yes")
DATABASE_URL = os.getenv("DATABASE_URL")
