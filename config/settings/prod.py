from .base import *

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
LOCAL_MODE = os.getenv("LOCAL_MODE", "False").lower() in ("true", "1", "yes")

# Security Settings in Production
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
