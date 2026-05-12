"""Lightweight role-based access control used by Orchid CLI and experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Mapping, Set

Action = str
Role = str


DEFAULT_POLICY: Mapping[Role, FrozenSet[Action]] = {
    "admin": frozenset({"preprocess", "experiment", "dp_sensitive", "read_logs"}),
    "ml_engineer": frozenset({"preprocess", "experiment", "read_logs"}),
    "analyst": frozenset({"experiment", "read_logs"}),
    "viewer": frozenset({"read_logs"}),
}


@dataclass
class AccessControl:
    """Role-based access guard with immutable policy."""

    policy: Mapping[Role, Iterable[Action]] = field(
        default_factory=lambda: {role: set(actions) for role, actions in DEFAULT_POLICY.items()}
    )

    def _to_set(self, role: Role) -> Set[Action]:
        actions = self.policy.get(role) or set()
        return {str(a) for a in actions}

    def can(self, role: Role, action: Action) -> bool:
        """Return ``True`` if ``role`` may perform ``action``."""
        actions = self._to_set(role)
        if "*" in actions:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Role '%s' has wildcard ('*') permission — grants access to ALL actions", role
            )
        return action in actions or "*" in actions

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
