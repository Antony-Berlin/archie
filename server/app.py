"""
FastAPI application for the NeuralArch-Bench Environment.

Endpoints:
    POST /reset   — start a new episode
    POST /step    — submit new model code, get observation + reward
    GET  /state   — inspect current state
    GET  /schema  — action/observation JSON schemas
    WS   /ws      — WebSocket for persistent sessions

Run locally:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv-core is required. Run: pip install openenv-core") from e

try:
    from ..core.models import NeuralArchAction, NeuralArchObservation
    from .neural_arch_environment import NeuralArchEnvironment
except ImportError:
    from core.models import NeuralArchAction, NeuralArchObservation
    from server.neural_arch_environment import NeuralArchEnvironment


app = create_app(
    NeuralArchEnvironment,
    NeuralArchAction,
    NeuralArchObservation,
    env_name="neural-arch-bench",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for: uv run --project . server  OR  python -m server.app"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
