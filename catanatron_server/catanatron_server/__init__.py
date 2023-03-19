import os

from flask import Flask
from flask_cors import CORS


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # ===== Load base configuration
    database_url = os.environ.get("DATABASE_URL", "sqlite:///:memory:")
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    secret_key = os.environ.get("SECRET_KEY", "dev")
    app.config.from_mapping(
        SECRET_KEY=secret_key,
        SQLALCHEMY_DATABASE_URI=database_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    if test_config is not None:
        app.config.update(test_config)

    # ===== Initialize Database
    from catanatron_server.models import db

    with app.app_context():
        db.init_app(app)
        db.create_all()

    # ===== Initialize Routes
    from . import api

    app.register_blueprint(api.bp)

    return app
