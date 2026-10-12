# Genesis AI code-execution sandbox image.
# Locked down on purpose: no network tools, no shell utilities beyond what's needed,
# runs as a non-root user, and is meant to be launched with --network none,
# --read-only, and explicit --memory/--cpus limits (see docker_sandbox.py).
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    xgboost==2.0.3

# Non-root user - if the generated code has a bug that lets something escape the
# intended script, it still can't do anything privileged inside the container.
RUN useradd --no-create-home --uid 1000 sandboxuser
USER sandboxuser

WORKDIR /sandbox
ENTRYPOINT ["python", "pipeline.py"]
