from .fastgs import FastGSAdapter
from .lightx2v import LightX2VAdapter
from .ultrashape import UltraShapeAdapter
from .world import WorldAdapter


def build_registry() -> dict[str, object]:
    return {
        "ultrashape": UltraShapeAdapter(),
        "world": WorldAdapter(),
        "lightx2v": LightX2VAdapter(),
        "fastgs": FastGSAdapter(),
    }


__all__ = ["build_registry"]
