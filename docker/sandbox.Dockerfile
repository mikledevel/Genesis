# Genesis AI code-execution sandbox image.
# Locked down on purpose: no network tools, no shell utilities beyond what's needed,
# runs as a non-root user, and is meant to be launched with --network none,
# --read-only, and explicit --memory/--cpus limits (see docker_sandbox.py).
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    pandas==3.0.2 \
    numpy==2.4.4 \
    scikit-learn==1.8.0 \
    xgboost-cpu==3.4.1

# Non-root user - if the generated code has a bug that lets something escape the
# intended script, it still can't do anything privileged inside the container.
RUN useradd --no-create-home --uid 1000 sandboxuser
USER sandboxuser

WORKDIR /sandbox
ENTRYPOINT ["python", "pipeline.py"]
