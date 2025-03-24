import asyncio

from . import env, server


def main():
    """Main entry point for the package."""
    asyncio.run(server.main())


__all__ = ["main", "server"]
