"""
DockerSandbox - production-grade isolation for executing AI-generated ML pipeline code,
replacing the dev-only subprocess sandbox in ai_generator.py for real, multi-tenant use.

Isolation applied on every run:
  - network_mode="none"      - the container cannot reach the internet or the host network
  - mem_limit / nano_cpus    - hard resource caps, one runaway script can't starve the host
  - read_only root filesystem - only /sandbox (the mounted workdir) is writable
  - non-root user (baked into the image, see sandbox.Dockerfile)
  - a wall-clock timeout enforced from the Python side, independent of the container

Falls back cleanly to the (weaker) subprocess sandbox in ai_generator.py if Docker isn't
installed/running - this keeps local single-user development working without requiring
Docker, while real deployments serving other users should always have Docker available.
"""
import os
import tempfile
import shutil
from typing import Dict

IMAGE_NAME = "genesis-sandbox:latest"
DOCKERFILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docker", "sandbox.Dockerfile")


def docker_available() -> bool:
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def ensure_image_built() -> bool:
    """Build the sandbox image if it doesn't exist yet. Returns True if the image is
    ready to use, False if Docker itself isn't available or the build failed."""
    try:
        import docker
        client = docker.from_env()
        try:
            client.images.get(IMAGE_NAME)
            return True
        except docker.errors.ImageNotFound:
            pass
        build_context = os.path.dirname(DOCKERFILE_PATH)
        client.images.build(path=build_context, dockerfile="sandbox.Dockerfile", tag=IMAGE_NAME, rm=True)
        return True
    except Exception as e:
        print(f"[DockerSandbox] could not build sandbox image: {e}")
        return False


class DockerSandbox:
    TIMEOUT_SECONDS = 45
    MEMORY_LIMIT = "512m"
    NANO_CPUS = 1_000_000_000  # 1 full CPU core

    def execute(self, code: str, workdir: str) -> Dict:
        """Same interface/return shape as AICodeGenerator.execute() so it's a drop-in
        replacement: {"success": bool, "stdout": str, "stderr": str, "returncode": int|None}."""
        import docker
        from docker.errors import ContainerError, ImageNotFound, APIError

        os.makedirs(os.path.join(workdir, "models"), exist_ok=True)
        with open(os.path.join(workdir, "pipeline.py"), "w", encoding="utf-8") as f:
            f.write(code)

        client = docker.from_env()
        container = None
        try:
            container = client.containers.run(
                IMAGE_NAME,
                volumes={os.path.abspath(workdir): {"bind": "/sandbox", "mode": "rw"}},
                working_dir="/sandbox",
                network_mode="none",
                mem_limit=self.MEMORY_LIMIT,
                nano_cpus=self.NANO_CPUS,
                read_only=True,
                tmpfs={"/tmp": "size=64m"},  # sklearn/xgboost sometimes need scratch space
                detach=True,
                stdout=True,
                stderr=True,
            )
            result = container.wait(timeout=self.TIMEOUT_SECONDS)
            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            returncode = result.get("StatusCode", 1)
            success = returncode == 0 and "PIPELINE_SUCCESS" in logs
            return {"success": success, "stdout": logs[-4000:], "stderr": "" if success else logs[-4000:], "returncode": returncode}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": f"Sandbox execution error: {e}", "returncode": None}
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
