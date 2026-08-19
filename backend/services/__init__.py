# backend/services/__init__.py
# Service layer package.
# Services contain business logic and orchestrate the recommendation pipeline.
# Flask routes call services — services call src/ modules.
# This separation keeps routes thin and logic testable.
