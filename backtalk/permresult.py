# backtalk: talk to your opencode agent out loud.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Permission decisions — the tiny contract the spoken gate returns.

These used to be `claude_agent_sdk.PermissionResultAllow` /
`PermissionResultDeny`. opencode has no SDK and no such classes: the gate's
answer is turned into a `{"response": "once" | "reject"}` POST by
brain._handle_permission. The gate in main.py still wants to express "allow",
"deny with this reason", so the shape is kept — as plain dataclasses, with
no dependency at all.

Kept in its own module so main.py's gate and brain's translation cannot
drift: both speak this vocabulary.

Note: opencode's permission reply takes only {"response": "once" | "always" |
"reject"} with no reason field, so a Deny's `message` is logged rather than
delivered. The gate still says the denial out loud, so the person hears it.
"""
from dataclasses import dataclass


@dataclass
class PermissionResultAllow:
    behavior: str = "allow"
    updated_input: dict | None = None
    updated_permissions: list | None = None


@dataclass
class PermissionResultDeny:
    behavior: str = "deny"
    message: str = ""
    interrupt: bool = False
