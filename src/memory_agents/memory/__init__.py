"""Pluggable memory backends for the memory-augmented agents.

The `Memory` ABC is the seam every method shares: Reflexion writes verbal
reflections keyed by puzzle id here; MemoryBank / MemP will plug in the same way.
"""

from .base import Memory
from .reflexion import Reflector, ReflexionMemory

__all__ = ["Memory", "Reflector", "ReflexionMemory"]
