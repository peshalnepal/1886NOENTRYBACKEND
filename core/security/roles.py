"""
Role definitions for the multi-tenant RBAC system.

The system uses three distinct scopes:

  1. Platform scope  -> `PlatformRole`
       The root of the hierarchy. A Platform Admin can create
       Organizations and seed their first Org Admin. This role is
       represented by the boolean flag `users.is_platform_admin` so
       that a single user identity can hold platform-scope privileges
       independent of any specific organization.

  2. Organization scope -> `OrgRole`
       Maps a user to an organization (tenant). Stored in
       `org_memberships`. The Operator role is intentionally
       organization-scoped: an operator is the first reviewer of
       AI camera alerts for *all* sites within their org.

  3. Site scope -> `SiteRole`
       Maps a user to a single site inside an organization. Stored
       in `site_memberships`. Drives day-to-day permissions such as
       arming/disarming and viewing approved alerts.

Important invariants enforced by the application layer (see
`AuthzService`):

  * A `SiteMembership` only makes sense if the user also has an
    `OrgMembership` for the site's organization. The membership
    repository validates this before insert.
  * Org Admins implicitly have full access to every site in their
    organization, even without an explicit `SiteMembership` row.
  * Platform Admins bypass all org/site checks.
"""
from __future__ import annotations

from enum import Enum


class PlatformRole(str, Enum):
    """
    Platform-level role.

    Only one value exists today; the enum is kept for symmetry with
    `OrgRole`/`SiteRole` and to leave room for future platform-level
    distinctions (e.g. read-only auditor).
    """

    SUPER_ADMIN = "super_admin"


class OrgRole(str, Enum):
    """
    Organization-level role.

    * ADMIN     - manages the org: creates sites, devices, cameras and
                  invites/removes users. Has implicit access to every
                  site in the org.
    * OPERATOR  - reviews pending AI camera alerts for the whole org
                  and approves/rejects them. May also generate and
                  send reports to site members.
    * MEMBER    - a plain organization member. Site-level access
                  must be granted via a `SiteMembership` row.
    """

    ADMIN = "admin"
    OPERATOR = "operator"
    MEMBER = "member"


class SiteRole(str, Enum):
    """
    Site-level role assigned via `site_memberships`.

    * ADMIN      - can edit the site, its cameras and its members.
                   Identical to an Org Admin's implicit access but
                   scoped to a single site (useful when an Org Admin
                   wants to delegate one site without granting org
                   admin rights everywhere).
    * ARM_DISARM - can arm or disarm the site and receives approved
                   notifications for it.
    * READ_ONLY  - can view the site (live streams, approved alerts)
                   but cannot change its armed state or settings.
    """

    ADMIN = "admin"
    ARM_DISARM = "arm_disarm"
    READ_ONLY = "read_only"


# =========================================================================
# Permission catalog (ACL-style RBAC)
# =========================================================================
class RoleScope(str, Enum):
    """The context a role applies to. Stored on `roles.scope`.

    A role name is only unique *within* a scope: both an org and a site
    have a role literally named ``admin`` with different powers.
    """

    ORG = "org"
    SITE = "site"


class Permission(str, Enum):
    """Atomic, checkable privileges granted to a role.

    Names are namespaced ``<resource>:<action>`` so the catalog reads
    cleanly. A permission is checked in a context (an org or a site) via
    `AuthzService.has_permission`.
    """

    # --- Organization scope ---
    ORG_READ = "org:read"
    ORG_MANAGE_MEMBERS = "org:manage_members"
    ORG_MANAGE_SITES = "org:manage_sites"
    ORG_MANAGE_DEVICES = "org:manage_devices"
    ORG_MANAGE_CAMERAS = "org:manage_cameras"
    ORG_MANAGE_SETTINGS = "org:manage_settings"
    ALERTS_APPROVE = "alerts:approve"
    REPORTS_SEND = "reports:send"

    # --- Site scope ---
    SITE_READ = "site:read"
    SITE_ARM_DISARM = "site:arm_disarm"
    SITE_MANAGE = "site:manage"
    SITE_MANAGE_MEMBERS = "site:manage_members"
    SITE_MANAGE_CAMERAS = "site:manage_cameras"


# Every org/site permission as a frozenset, used to build the admin grants.
_ORG_PERMISSIONS = frozenset(
    {
        Permission.ORG_READ,
        Permission.ORG_MANAGE_MEMBERS,
        Permission.ORG_MANAGE_SITES,
        Permission.ORG_MANAGE_DEVICES,
        Permission.ORG_MANAGE_CAMERAS,
        Permission.ORG_MANAGE_SETTINGS,
        Permission.ALERTS_APPROVE,
        Permission.REPORTS_SEND,
    }
)
_SITE_PERMISSIONS = frozenset(
    {
        Permission.SITE_READ,
        Permission.SITE_ARM_DISARM,
        Permission.SITE_MANAGE,
        Permission.SITE_MANAGE_MEMBERS,
        Permission.SITE_MANAGE_CAMERAS,
    }
)


# Single source of truth for "what can this role do". Keyed by
# (role_name, scope) -> set of Permission. An Org Admin carries every org
# *and* site permission, which encodes the product rule that an Org Admin
# implicitly has full access to every site in their org.
#
# This dict is the build-time seed: at startup it populates the
# `roles`/`permissions`/`role_permissions` tables (see core/database.py).
# At request time `AuthzService` answers permission checks by JOINing those
# rows against the caller's `access_grants`, not by reading this dict. Routes
# therefore never hard-code role names; they declare a `Permission`.
ROLE_PERMISSIONS: dict[tuple[str, str], frozenset[Permission]] = {
    (OrgRole.ADMIN.value, RoleScope.ORG.value): _ORG_PERMISSIONS | _SITE_PERMISSIONS,
    (OrgRole.OPERATOR.value, RoleScope.ORG.value): frozenset(
        {Permission.ORG_READ, Permission.ALERTS_APPROVE, Permission.REPORTS_SEND}
    ),
    (OrgRole.MEMBER.value, RoleScope.ORG.value): frozenset({Permission.ORG_READ}),
    (SiteRole.ADMIN.value, RoleScope.SITE.value): _SITE_PERMISSIONS,
    (SiteRole.ARM_DISARM.value, RoleScope.SITE.value): frozenset(
        {Permission.SITE_READ, Permission.SITE_ARM_DISARM}
    ),
    (SiteRole.READ_ONLY.value, RoleScope.SITE.value): frozenset({Permission.SITE_READ}),
}


# Human-readable descriptions for the seeded permission rows.
PERMISSION_DESCRIPTIONS: dict[str, str] = {
    Permission.ORG_READ.value: "View the organization and its resources.",
    Permission.ORG_MANAGE_MEMBERS.value: "Add, remove and re-role organization members.",
    Permission.ORG_MANAGE_SITES.value: "Create, edit and delete sites in the org.",
    Permission.ORG_MANAGE_DEVICES.value: "Create, edit and delete devices in the org.",
    Permission.ORG_MANAGE_CAMERAS.value: "Create, edit and delete cameras in the org.",
    Permission.ORG_MANAGE_SETTINGS.value: "Change organization-level settings.",
    Permission.ALERTS_APPROVE.value: "Review and approve/reject pending AI alerts.",
    Permission.REPORTS_SEND.value: "Generate and send reports to site members.",
    Permission.SITE_READ.value: "View a site, its live streams and approved alerts.",
    Permission.SITE_ARM_DISARM.value: "Arm or disarm a site.",
    Permission.SITE_MANAGE.value: "Edit a site and its settings.",
    Permission.SITE_MANAGE_MEMBERS.value: "Grant and revoke site-level roles.",
    Permission.SITE_MANAGE_CAMERAS.value: "Create, edit and delete a site's cameras.",
}
