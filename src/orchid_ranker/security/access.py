"""Lightweight role-based access control used by Orchid CLI and experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Set


Action = str
Role = str


DEFAULT_POLICY: Dict[Role, Set[Action]] = {
    "admin": {"preprocess", "experiment", "dp_sensitive", "read_logs"},
    "ml_engineer": {"preprocess", "experiment", "read_logs"},
    "analyst": {"experiment", "read_logs"},
    "viewer": {"read_logs"},
}


@dataclass
class AccessControl:
    """Role-based access guard with immutable policy."""

    policy: Mapping[Role, Iterable[Action]] = field(default_factory=lambda: DEFAULT_POLICY)

    def _to_set(self, role: Role) -> Set[Action]:
        actions = self.policy.get(role) or set()
        return {str(a) for a in actions}

    def can(self, role: Role, action: Action) -> bool:
        """Return ``True`` if ``role`` may perform ``action``."""
        return action in self._to_set(role) or "*" in self._to_set(role)

    def require(self, role: Role, action: Action) -> None:
        """Raise ``PermissionError`` when ``role`` lacks ``action``."""
        if not self.can(role, action):
            allowed = ", ".join(sorted(self._to_set(role))) or "<none>"
            raise PermissionError(f"Role '{role}' is not permitted to perform '{action}'. Allowed: {allowed}")


__all__ = [
    "Action",
    "Role",
    "DEFAULT_POLICY",
    "AccessControl",
]
