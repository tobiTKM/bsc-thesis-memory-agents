"""Pluggable memory backends for the memory-augmented agents.

The `Memory` ABC is the seam every method shares: Reflexion writes verbal
reflections keyed by puzzle id, while Memp (procedural) and MemoryBank
(declarative) are cross-task banks retrieved by embedding similarity -- the
latter a subclass of the former, since the two differ in what a memory *says*,
not in how it is retrieved.
"""

from .base import Memory
from .memorybank import CaseDistiller, MemoryBankMemory
from .memp import Distiller, MempMemory
from .reflexion import Reflector, ReflexionMemory

__all__ = ["Memory", "CaseDistiller", "MemoryBankMemory", "Distiller",
           "MempMemory", "Reflector", "ReflexionMemory"]
