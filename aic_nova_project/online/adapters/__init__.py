"""Database SDK adapters for Online read-only access."""

from typing import TYPE_CHECKING, Any

from .contract_validator import (
    CheckStatus,
    ContractCheck,
    ContractValidationReport,
    OfflineContractValidator,
    ValidationStatus,
)
from .elasticsearch import ElasticsearchSearchAdapter
from .evidence import ElasticsearchEvidenceHydrator
from .images import FilesystemImageResolver
from .milvus import MilvusSearchAdapter
from .manifest import DatasetManifestGate
from .sqlite import SQLiteReadAdapter
from .visual_corpus import MilvusSQLiteVisualCorpusAdapter

if TYPE_CHECKING:
    from .qwen_vlm import QwenVLMAdapter


def __getattr__(name: str) -> Any:
    """Load provider-specific adapters only when explicitly requested."""
    if name == "QwenVLMAdapter":
        from .qwen_vlm import QwenVLMAdapter

        return QwenVLMAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ElasticsearchSearchAdapter",
    "ElasticsearchEvidenceHydrator",
    "FilesystemImageResolver",
    "CheckStatus",
    "ContractCheck",
    "ContractValidationReport",
    "MilvusSearchAdapter",
    "DatasetManifestGate",
    "OfflineContractValidator",
    "QwenVLMAdapter",
    "SQLiteReadAdapter",
    "MilvusSQLiteVisualCorpusAdapter",
    "ValidationStatus",
]
