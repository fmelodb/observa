"""Reflective, MCP-backed specialist agents used by the turn-based graph.

The ``REGISTRY`` maps each agent's ``name`` (matching ``config.agents_enabled``
keys) to its class, so the turn controller can instantiate only the agents the
user has enabled.
"""
from __future__ import annotations

from observa.agents.ash import ASHAgent
from observa.agents.base import ReflectiveAgent
from observa.agents.concurrency import ConcurrencyAgent
from observa.agents.dg import DGAgent
from observa.agents.exadata import ExadataAgent
from observa.agents.file_reader import FileReaderAgent
from observa.agents.infra import InfraAgent
from observa.agents.io_storage import IOStorageAgent
from observa.agents.memory import MemoryAgent
from observa.agents.rac import RACAgent
from observa.agents.segment_object import SegmentObjectAgent
from observa.agents.sql import SQLAgent
from observa.agents.wait_time import WaitTimeAgent

REGISTRY: dict[str, type[ReflectiveAgent]] = {
    WaitTimeAgent.name: WaitTimeAgent,
    ASHAgent.name: ASHAgent,
    InfraAgent.name: InfraAgent,
    SQLAgent.name: SQLAgent,
    MemoryAgent.name: MemoryAgent,
    IOStorageAgent.name: IOStorageAgent,
    ConcurrencyAgent.name: ConcurrencyAgent,
    SegmentObjectAgent.name: SegmentObjectAgent,
    RACAgent.name: RACAgent,
    ExadataAgent.name: ExadataAgent,
    DGAgent.name: DGAgent,
    FileReaderAgent.name: FileReaderAgent,
}

__all__ = [
    "REGISTRY",
    "ReflectiveAgent",
    "ASHAgent",
    "ConcurrencyAgent",
    "DGAgent",
    "ExadataAgent",
    "FileReaderAgent",
    "InfraAgent",
    "IOStorageAgent",
    "MemoryAgent",
    "RACAgent",
    "SegmentObjectAgent",
    "SQLAgent",
    "WaitTimeAgent",
]
