"""
backend/app.py
==============
SchemeXpress — Flask Application Entry Point

This is the main file that starts the Flask web server.

How Flask works (briefly):
- Flask is a WSGI web framework. "WSGI" means it follows a standard
  Python interface for web servers and web applications.
- We create a Flask "app" object, register route blueprints on it,
  and run it. Flask then listens for HTTP requests and dispatches
  them to the correct route handler.

Why app.py and not run.py or main.py?
- Flask convention. The file that creates the Flask app object is
  typically called app.py. Some projects use an "application factory"
  pattern (create_app() function) for better testability — we'll
  migrate to that pattern in a later phase.

Environment variables:
- We use python-dotenv to load .env into os.environ before Flask reads it.
- This means SECRET_KEY, MONGO_URI etc. never appear in source code.
"""

import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ---------------------------------------------------------
# Load environment variables from .env file.
# load_dotenv() reads the .env file in the project root and
# sets each key-value pair as an environment variable.
# os.getenv() then reads those values safely.
# If .env doesn't exist (e.g. in production where env vars
# are set by the server), load_dotenv() silently does nothing.
# ---------------------------------------------------------
# override=True ensures the .env values take effect even if the variable
# was already set in the shell environment (important for the Flask
# debug reloader, which re-imports the module in the same process).
load_dotenv(override=True)


def create_app():
    """
    Application factory function.

    Why a factory function instead of a global app object?
    - Testability: tests can call create_app() to get a fresh app
      instance with a test configuration, without affecting the
      real application state.
    - Flexibility: different configurations (development, testing,
      production) can be passed to the factory.

    For now this is simple. We'll expand it in Phase 11.
    """

    # Flask(__name__) creates the app.
    # __name__ tells Flask where to look for templates and static files
    # relative to this file's location.
    # template_folder and static_folder point one level up because
    # our templates/ and static/ directories are at the project root,
    # not inside backend/.
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static"
    )

    # ---------------------------------------------------------
    # Configuration
    # app.config is a dictionary Flask uses for settings.
    # SECRET_KEY is required for session management and security.
    # os.getenv() reads from environment — set in .env file.
    # The second argument is a fallback for development only.
    # NEVER use a simple fallback string in production.
    # ---------------------------------------------------------
    app.config["SECRET_KEY"] = os.getenv(
        "FLASK_SECRET_KEY",
        "dev-fallback-key-replace-in-production"
    )
    app.config["ENV"] = os.getenv("FLASK_ENV", "development")
    app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "1") == "1"

    # ---------------------------------------------------------
    # CORS (Cross-Origin Resource Sharing)
    # -------------------------------------------------------
    # In local development the Next.js frontend runs on
    # http://localhost:3000 while Flask runs on :5000.
    # Browsers block cross-origin requests unless the server
    # explicitly allows them via the Access-Control-Allow-Origin
    # response header.
    #
    # flask-cors handles this automatically.
    #
    # FRONTEND_ORIGIN in .env controls which origin is allowed.
    # Default: http://localhost:3000  (safe for local dev only).
    #
    # In production set FRONTEND_ORIGIN to your real domain, e.g.:
    #   FRONTEND_ORIGIN=https://schemegenie.vercel.app
    # ---------------------------------------------------------
    frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    # Support comma-separated list of origins (e.g. localhost:3000,localhost:3001)
    origins = [o.strip() for o in frontend_origin.split(",") if o.strip()]
    CORS(
        app,
        resources={r"/api/*": {"origins": origins}},
        supports_credentials=False,
    )

    # ---------------------------------------------------------
    # Register Blueprints
    # ---------------------------------------------------------
    from backend.routes.api import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # Phase 11+: uncomment as routes are built
    # from backend.routes.main import main_bp
    # from backend.routes.recommendation import recommendation_bp
    # from backend.routes.schemes import schemes_bp
    # app.register_blueprint(main_bp)
    # app.register_blueprint(recommendation_bp)
    # app.register_blueprint(schemes_bp)

    # ---------------------------------------------------------
    # Temporary route — confirms Flask is working
    # This will be replaced by the proper home page in Phase 12.
    # ---------------------------------------------------------
    @app.route("/")
    def index():
        """
        Temporary home route.
        Returns a plain HTML confirmation that the server is running.
        In Phase 12 this becomes render_template('home.html').
        """
        return """
        <html>
          <body style="font-family: sans-serif; max-width: 600px; margin: 60px auto; text-align: center;">
            <h1>SchemeXpress</h1>
            <p>Smart Government Scheme Recommendation Platform</p>
            <hr>
            <p style="color: green;">Flask server is running successfully.</p>
            <p style="color: #666; font-size: 0.9em;">
              Phase 1 complete. Frontend will be built in Phase 12.
            </p>
          </body>
        </html>
        """

    # ---------------------------------------------------------
    # Custom error handlers
    # These catch HTTP errors and return friendly pages instead
    # of Flask's default white error pages.
    # We define them here so they apply to the whole application.
    # ---------------------------------------------------------
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        return """
        <html>
          <body style="font-family: sans-serif; max-width: 600px; margin: 60px auto; text-align: center;">
            <h2>Page Not Found</h2>
            <p>The page you requested does not exist.</p>
            <a href="/">Return to Home</a>
          </body>
        </html>
        """, 404

    @app.errorhandler(500)
    def server_error(error):
        """Handle 500 Internal Server Error."""
        return """
        <html>
          <body style="font-family: sans-serif; max-width: 600px; margin: 60px auto; text-align: center;">
            <h2>Something went wrong</h2>
            <p>We encountered an internal error. Please try again.</p>
            <a href="/">Return to Home</a>
          </body>
        </html>
        """, 500

    return app


# ---------------------------------------------------------
# Entry point
# This block only runs when you execute: python backend/app.py
# It does NOT run when Flask is imported as a module (e.g. in tests).
# This is standard Python — the if __name__ == "__main__" guard
# prevents code from running on import.
# ---------------------------------------------------------
if __name__ == "__main__":
    app = create_app()

    # host="0.0.0.0" makes Flask accessible on all network interfaces,
    # not just localhost. Useful for testing on your local network.
    # port=5000 is Flask's default port.
    # debug mode is read from the app config (set via .env).
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=app.config["DEBUG"]
    )
