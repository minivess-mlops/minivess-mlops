from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single entry in the SaMD audit trail."""

    timestamp: str
    event_type: str
    actor: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    data_hash: str | None = None


@dataclass
class AuditTrail:
    """IEC 62304 compliant audit trail for SaMD lifecycle."""

    entries: list[AuditEntry] = field(default_factory=list)

    def log_event(
        self,
        event_type: str,
        description: str,
        *,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
        data_hash: str | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=datetime.now(UTC).isoformat(),
            event_type=event_type,
            actor=actor,
            description=description,
            metadata=metadata or {},
            data_hash=data_hash,
        )
        self.entries.append(entry)
        logger.info("Audit [%s]: %s - %s", event_type, description, actor)
        return entry

    def log_data_access(
        self,
        dataset_name: str,
        file_paths: list[str],
        *,
        actor: str = "system",
    ) -> AuditEntry:
        content = "\n".join(sorted(file_paths))
        data_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.log_event(
            "DATA_ACCESS",
            f"Accessed dataset: {dataset_name}",
            actor=actor,
            metadata={"dataset": dataset_name, "num_files": len(file_paths)},
            data_hash=data_hash,
        )

    def log_model_training(
        self,
        model_name: str,
        config: dict[str, Any],
        *,
        actor: str = "system",
    ) -> AuditEntry:
        return self.log_event(
            "MODEL_TRAINING",
            f"Trained model: {model_name}",
            actor=actor,
            metadata={"model": model_name, "config": config},
        )

    def log_model_deployment(
        self,
        model_name: str,
        version: str,
        *,
        actor: str = "system",
    ) -> AuditEntry:
        return self.log_event(
            "MODEL_DEPLOYMENT",
            f"Deployed {model_name} v{version}",
            actor=actor,
            metadata={"model": model_name, "version": version},
        )

    def log_test_evaluation(
        self,
        model_name: str,
        metrics: dict[str, float],
        *,
        actor: str = "system",
    ) -> AuditEntry:
        return self.log_event(
            "TEST_EVALUATION",
            f"Evaluated {model_name} on test set",
            actor=actor,
            metadata={"model": model_name, "metrics": metrics},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self.entries]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Audit trail saved to %s (%d entries)", path, len(self.entries))

    @classmethod
    def load(cls, path: Path) -> AuditTrail:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [AuditEntry(**entry) for entry in data]
        trail = cls(entries=entries)
        logger.info("Audit trail loaded from %s (%d entries)", path, len(entries))
        return trail
