"""
Expose the LightX2V-compatible API server as `infinitetalk.serving`.

Implementation lives in the top-level `serving` package (kept for backward compatibility).
"""

def run_server(args):
    # Lazy import: avoid importing heavy deps (librosa/torch) on `import infinitetalk.serving`.
    from serving.main import run_server as _run_server

    return _run_server(args)

__all__ = ["run_server"]

