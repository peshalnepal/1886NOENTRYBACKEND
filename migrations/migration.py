"""
Migrates an existing database (old ORM) to the new schema (new ORM).

Self-contained: it first brings the *schema* up to the new ORM, then seeds
and backfills data. The old `org_memberships` / `site_memberships` tables are
gone; membership is now a row in `access_grants` (user -> role -> context),
where `roles.scope` says whether the role is org-level (admin / operator /
member) or site-level (admin / arm_disarm / read_only), and a role's powers
come from the `role_permissions` links to the `permissions` catalog
(site:read, org:manage_sites, …).

Schema bring-up (idempotent, checks existence first). `create_all` only creates
whole missing tables — it NEVER ALTERs an existing one — so column additions on
pre-existing tables must be done by hand, each guarded by information_schema:
  - creates any missing tables from the ORM metadata (organizations, roles,
    permissions, role_permissions, access_grants, …)
  - adds any missing columns the new ORM introduced on existing tables
    (users.is_platform_admin, sites.org_id, sites.created_by, sites.is_deleted)
    + the sites.org_id FK

This script then runs a DATA backfill + fixup pass:

  - seeds the roles / permissions / role_permissions catalog idempotently
    from core.security.roles (the single source of truth)
  - one `organization` per user (named after the user)
  - an org-scoped `admin` access grant for that user on their org
  - links the user's sites to the org (sites.org_id)
  - a site-scoped access grant for every site the user owns

Site-role normalization (per the RBAC model in core/security/roles.py):
  - Platform admins (users.is_platform_admin) and Org Admins (an org-scoped
    `admin` grant on the site's org) keep their site grants AS-IS (admin).
  - Everyone else — Operators and plain Members — is reduced to a
    site-scoped `read_only` grant.

Usage:
    pip install sqlalchemy pymysql cryptography
    python migrate.py                        # uses hardcoded config below
    python migrate.py --dry-run              # rolls back, no changes committed
    python migrate.py --url "mysql+pymysql://user:pass@host/db"  # override URL
"""


import argparse
import logging
import os
import re
import sys
import urllib.parse
from datetime import datetime, timezone

from sqlalchemy import bindparam, create_engine, text

# Make the Backend package importable when this script is run standalone
# (e.g. `python migrations/migration.py`), so we can read the RBAC catalog
# (roles/permissions) from its single source of truth in core.security.roles.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)
 
 



# Connection comes from the environment (or --url). Never hardcode a password
# here: this file is committed.
DB_HOST     = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_NAME     = os.getenv("DB_NAME", "appdb")
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

 
def build_url(host, port, name, user, password):
    pwd = urllib.parse.quote(password, safe="")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}"
 
 
def utc_now():
    return datetime.now(timezone.utc)
 
 
def slugify(name: str) -> str:
    """'John Doe' → 'john-doe'"""
    s = re.sub(r"[\s_]+", "-", name.strip().lower())
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "org"
 
 
def unique_slug(conn, base: str) -> str:
    """Append -2, -3 … until the slug is not taken."""
    slug = base[:60]
    row = conn.execute(
        text("SELECT id FROM organizations WHERE slug = :s"), {"s": slug}
    ).fetchone()
    if not row:
        return slug
    counter = 2
    while True:
        candidate = f"{base[:57]}-{counter}"
        row = conn.execute(
            text("SELECT id FROM organizations WHERE slug = :s"), {"s": candidate}
        ).fetchone()
        if not row:
            return candidate
        counter += 1
 
 
def _table_exists(conn, table: str) -> bool:
    return (
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t LIMIT 1"
            ),
            {"t": table},
        ).fetchone()
        is not None
    )


def _column_exists(conn, table: str, column: str) -> bool:
    return (
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "  AND TABLE_NAME = :t AND COLUMN_NAME = :c LIMIT 1"
            ),
            {"t": table, "c": column},
        ).fetchone()
        is not None
    )


def _fk_exists(conn, table: str, column: str, ref_table: str) -> bool:
    return (
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.KEY_COLUMN_USAGE "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t "
                "  AND COLUMN_NAME = :c AND REFERENCED_TABLE_NAME = :rt LIMIT 1"
            ),
            {"t": table, "c": column, "rt": ref_table},
        ).fetchone()
        is not None
    )


# New columns the updated ORM added onto pre-existing tables. create_all()
# only creates whole missing tables — it never ALTERs an existing one — so
# these are added by hand, each guarded by information_schema.
#   (table, column, column DDL)
_NEW_COLUMNS = [
    ("users", "is_platform_admin", "BOOLEAN NOT NULL DEFAULT 0"),
    ("sites", "org_id", "INT NULL"),
    ("sites", "created_by", "INT NULL"),
    ("sites", "is_deleted", "BOOLEAN NOT NULL DEFAULT 0"),
]


def ensure_schema(engine, dry_run: bool = False):
    """Create any missing tables, then add any missing columns.

    Step 1 uses the ORM metadata so every new table (organizations, roles,
    permissions, role_permissions, access_grants, …) is created if absent
    and left untouched if present (create_all is checkfirst=True).

    Step 2 adds the columns the new ORM introduced on existing tables.

    NOTE: DDL auto-commits in MySQL and cannot be rolled back, so the schema
    changes here are applied even under --dry-run (only the data backfill is
    rolled back). This is logged loudly.
    """
    if dry_run:
        log.warning(
            "DRY RUN: schema DDL (CREATE TABLE / ADD COLUMN) still applies — "
            "MySQL cannot roll back DDL. Only the data backfill is rolled back."
        )

    from core.database_orm import Base

    before = set(Base.metadata.tables.keys())
    with engine.connect() as conn:
        existing = {t for t in before if _table_exists(conn, t)}
    missing = sorted(before - existing)
    Base.metadata.create_all(engine)  # checkfirst=True by default
    if missing:
        log.info("Schema: created %d missing table(s): %s", len(missing), ", ".join(missing))
    else:
        log.info("Schema: all ORM tables already exist.")

    with engine.begin() as conn:
        for table, column, ddl in _NEW_COLUMNS:
            if not _table_exists(conn, table):
                log.warning("Schema: table '%s' missing — skipping column '%s'.", table, column)
                continue
            if _column_exists(conn, table, column):
                continue
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
            log.info("Schema: added column %s.%s", table, column)

        # Best-effort FK for the column the backfill writes to.
        if (
            _table_exists(conn, "sites")
            and _table_exists(conn, "organizations")
            and _column_exists(conn, "sites", "org_id")
            and not _fk_exists(conn, "sites", "org_id", "organizations")
        ):
            try:
                conn.execute(
                    text(
                        "ALTER TABLE sites ADD CONSTRAINT fk_sites_org_id "
                        "FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE"
                    )
                )
                log.info("Schema: added FK sites.org_id -> organizations.id")
            except Exception as exc:  # noqa: BLE001 — FK is non-critical for backfill
                log.warning("Schema: could not add FK sites.org_id: %s", exc)


def seed_rbac_catalog(conn):
    """Idempotently seed the roles / permissions / role_permissions tables.

    Mirrors the app's startup seed (core/database.py) and pulls the role
    names, scopes and permission names from the single source of truth in
    ``core.security.roles`` (ROLE_PERMISSIONS / PERMISSION_DESCRIPTIONS).

    Running this first guarantees the backfill's
    ``INSERT ... SELECT FROM roles WHERE name=... AND scope=...`` lookups
    actually find a row instead of silently inserting nothing when the
    catalog has not been seeded yet.
    """
    from core.security.roles import PERMISSION_DESCRIPTIONS, ROLE_PERMISSIONS

    for perm_name, perm_desc in PERMISSION_DESCRIPTIONS.items():
        conn.execute(
            text(
                "INSERT INTO permissions (name, description) "
                "SELECT :n, :d FROM DUAL "
                "WHERE NOT EXISTS (SELECT 1 FROM permissions WHERE name = :n)"
            ),
            {"n": perm_name, "d": perm_desc},
        )

    # The same name (e.g. "admin") exists in both the "org" and "site" scope
    # with different powers, so uniqueness is on the (name, scope) pair.
    role_keys = set(ROLE_PERMISSIONS.keys())
    for (role_name, scope) in role_keys:
        conn.execute(
            text(
                "INSERT INTO roles (name, scope, description) "
                "SELECT :n, :s, :d FROM DUAL "
                "WHERE NOT EXISTS (SELECT 1 FROM roles WHERE name = :n AND scope = :s)"
            ),
            {"n": role_name, "s": scope, "d": f"{scope} role: {role_name}"},
        )

    link_rules = 0
    for (role_name, scope), perms in ROLE_PERMISSIONS.items():
        for perm in perms:
            conn.execute(
                text(
                    "INSERT INTO role_permissions (role_id, permission_id) "
                    "SELECT r.id, p.id FROM roles r, permissions p "
                    "WHERE r.name = :rn AND r.scope = :rs AND p.name = :pn "
                    "  AND NOT EXISTS (SELECT 1 FROM role_permissions rp "
                    "    WHERE rp.role_id = r.id AND rp.permission_id = p.id)"
                ),
                {"rn": role_name, "rs": scope, "pn": perm.value},
            )
            link_rules += 1

    log.info(
        "RBAC catalog ensured: %d permission(s), %d role(s), %d role->perm rule(s).",
        len(PERMISSION_DESCRIPTIONS), len(role_keys), link_rules,
    )


def _downgrade_site_roles(conn, now):
    """Reduce site grants to ``read_only`` for everyone who is not an admin.

    Keep AS-IS when the grant holder is a platform admin or holds an
    org-scoped ``admin`` grant on the org that owns the site. Otherwise
    (Operators / plain Members) the site grant is changed to ``read_only``.

    The candidate rows are resolved with a plain SELECT first, then the
    UPDATE/DELETE run by primary key. This avoids MySQL error 1093
    ("can't specify target table for update in FROM clause") that a
    self-referential UPDATE-with-subquery would otherwise raise.
    """
    ro = conn.execute(
        text("SELECT id FROM roles WHERE name = 'read_only' AND scope = 'site'")
    ).fetchone()
    if not ro:
        log.warning("No site 'read_only' role found — skipping site-role downgrade.")
        return
    read_only_id = ro.id

    grants = conn.execute(
        text("""
            SELECT
                g.id                  AS grant_id,
                g.user_id             AS user_id,
                g.site_uuid           AS site_uuid,
                g.role_id             AS role_id,
                u.is_platform_admin   AS is_platform_admin,
                EXISTS (
                    SELECT 1
                    FROM access_grants oag
                    JOIN roles orr ON orr.id = oag.role_id AND orr.scope = 'org'
                    WHERE oag.user_id = g.user_id
                      AND oag.org_id  = s.org_id
                      AND orr.name    = 'admin'
                )                     AS is_org_admin
            FROM access_grants g
            JOIN roles r ON r.id = g.role_id AND r.scope = 'site'
            JOIN sites s ON s.site_uuid = g.site_uuid
            JOIN users u ON u.id = g.user_id
            WHERE g.site_uuid IS NOT NULL
        """)
    ).fetchall()

    # (user_id, site_uuid) pairs that already hold a read_only grant, so a
    # downgrade does not collide with uq_access_grant_user_role_context.
    existing_ro = {
        (row.user_id, row.site_uuid)
        for row in grants
        if row.role_id == read_only_id
    }

    to_downgrade: list = []
    to_delete: list = []
    for row in grants:
        # Admins (platform or org) keep their grant exactly as it is.
        if bool(row.is_platform_admin) or bool(row.is_org_admin):
            continue
        if row.role_id == read_only_id:
            continue
        key = (row.user_id, row.site_uuid)
        if key in existing_ro:
            # A read_only grant already exists for this user+site; drop the
            # redundant (admin/arm_disarm) one instead of creating a dupe.
            to_delete.append(row.grant_id)
        else:
            to_downgrade.append(row.grant_id)
            existing_ro.add(key)  # guard against a sibling grant colliding

    if to_delete:
        conn.execute(
            text("DELETE FROM access_grants WHERE id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": to_delete},
        )
    if to_downgrade:
        conn.execute(
            text(
                "UPDATE access_grants SET role_id = :ro, updated_at = :now "
                "WHERE id IN :ids"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ro": read_only_id, "now": now, "ids": to_downgrade},
        )

    log.info(
        "Site-role normalization: %d grant(s) reduced to read_only, "
        "%d redundant grant(s) removed.",
        len(to_downgrade), len(to_delete),
    )


def run_seed(db_url: str, dry_run: bool = False):
    engine = create_engine(db_url, echo=False)
    now = utc_now()

    # Schema first: create missing tables + columns before any data.
    ensure_schema(engine, dry_run=dry_run)

    with engine.begin() as conn:

        # The backfill below resolves role ids by (name, scope); seed the
        # catalog first so those lookups never silently no-op.
        seed_rbac_catalog(conn)

        users = conn.execute(
            text("SELECT id, user_name, email FROM users ORDER BY id")
        ).fetchall()
 
        log.info("Found %d user(s) to process.", len(users))
 
        for user in users:
            user_id   = user.id
            user_name = user.user_name or user.email.split("@")[0]
 
            log.info("  [user %d] %s", user_id, user_name)
 
            existing_mem = conn.execute(
                text("""
                    SELECT g.id, g.org_id
                    FROM access_grants g
                    JOIN roles r ON r.id = g.role_id AND r.scope = 'org'
                    WHERE g.user_id = :uid AND g.org_id IS NOT NULL
                    LIMIT 1
                """),
                {"uid": user_id},
            ).fetchone()
 
            if existing_mem:
                org_id = existing_mem.org_id
                log.info("    ✓ org_membership already exists  org_id=%d — skipped", org_id)
            else:
                org_name = user_name
                slug     = unique_slug(conn, slugify(org_name))
 
                result = conn.execute(
                    text("""
                        INSERT INTO organizations
                            (name, slug, owner_user_id, is_active, created_at, updated_at)
                        VALUES
                            (:name, :slug, :owner, 1, :now, :now)
                    """),
                    {"name": org_name, "slug": slug, "owner": user_id, "now": now},
                )
                org_id = result.lastrowid
                log.info("    + organization created  id=%d  slug=%s", org_id, slug)
 
                conn.execute(
                    text("""
                        INSERT INTO access_grants
                            (user_id, role_id, org_id, created_at, updated_at)
                        SELECT :uid, r.id, :oid, :now, :now
                        FROM roles r
                        WHERE r.name = 'admin' AND r.scope = 'org'
                    """),
                    {"uid": user_id, "oid": org_id, "now": now},
                )
                log.info("    + org access grant created  role=admin")
 
            updated = conn.execute(
                text("""
                    UPDATE sites
                    SET org_id = :oid, updated_at = :now
                    WHERE user_id = :uid
                      AND (org_id IS NULL OR org_id != :oid)
                """),
                {"oid": org_id, "uid": user_id, "now": now},
            ).rowcount
            if updated:
                log.info("    + %d site(s) linked to org %d", updated, org_id)
            else:
                log.info("    ✓ all sites already linked to org %d — skipped", org_id)
 
            # Every site the user owns gets a site-scoped admin grant; the
            # normalization pass below then reduces it where appropriate.
            sites = conn.execute(
                text("SELECT site_uuid FROM sites WHERE user_id = :uid"),
                {"uid": user_id},
            ).fetchall()

            sm_inserted = sm_skipped = 0
            for site_row in sites:
                site_uuid = site_row.site_uuid

                already = conn.execute(
                    text("""
                        SELECT g.id FROM access_grants g
                        JOIN roles r ON r.id = g.role_id AND r.scope = 'site'
                        WHERE g.user_id = :uid AND g.site_uuid = :suuid
                    """),
                    {"uid": user_id, "suuid": site_uuid},
                ).fetchone()

                if already:
                    sm_skipped += 1
                    continue

                conn.execute(
                    text("""
                        INSERT INTO access_grants
                            (user_id, role_id, site_uuid, created_at, updated_at)
                        SELECT :uid, r.id, :suuid, :now, :now
                        FROM roles r
                        WHERE r.name = 'admin' AND r.scope = 'site'
                    """),
                    {"uid": user_id, "suuid": site_uuid, "now": now},
                )
                sm_inserted += 1

            if sm_inserted:
                log.info("    + %d site access grant(s) created  role=admin", sm_inserted)
            if sm_skipped:
                log.info("    ✓ %d site access grant(s) already exist — skipped", sm_skipped)

        # Normalize site roles by each holder's org standing:
        # admin / platform admin -> kept as-is, everyone else -> read_only.
        _downgrade_site_roles(conn, now)

        if dry_run:
            raise Exception("DRY RUN — rolling back.")
 
    log.info("Seed complete ✓")
 
 
def print_summary(db_url: str):
    """Post-apply verification: one row per user with their org, role and counts."""
    engine = create_engine(db_url, echo=False)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                u.id                        AS user_id,
                u.user_name,
                o.id                        AS org_id,
                o.name                      AS org_name,
                org_role.name               AS org_role,
                COUNT(DISTINCT s.site_uuid) AS site_count,
                COUNT(DISTINCT sg.id)       AS site_grants
            FROM users u
            LEFT JOIN access_grants oag
                   ON oag.user_id = u.id AND oag.org_id IS NOT NULL
            LEFT JOIN roles org_role
                   ON org_role.id = oag.role_id AND org_role.scope = 'org'
            LEFT JOIN organizations o ON o.id = oag.org_id
            LEFT JOIN sites s
                   ON s.org_id = o.id AND s.is_deleted = 0
            LEFT JOIN access_grants sg
                   ON sg.user_id = u.id AND sg.site_uuid IS NOT NULL
            GROUP BY u.id, o.id, org_role.name
            ORDER BY u.id
        """)).fetchall()
 
    log.info("")
    log.info("%-8s %-25s %-8s %-20s %-8s %-6s %-14s",
             "user_id", "user_name", "org_id", "org_name", "org_role", "sites", "site_grants")
    log.info("-" * 100)
    for r in rows:
        log.info("%-8s %-25s %-8s %-20s %-8s %-6s %-14s",
                 r.user_id, r.user_name, r.org_id, r.org_name,
                 r.org_role, r.site_count, r.site_grants)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill orgs from existing users")
    parser.add_argument("--url", default=None,
                        help="Full SQLAlchemy URL; overrides the DB_* env vars.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done and roll back.")
    args = parser.parse_args()
 
    url = args.url or build_url(DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
 
    try:
        run_seed(url, dry_run=args.dry_run)
        if not args.dry_run:
            print_summary(url)
    except Exception as exc:
        if args.dry_run:
            log.info("Rolled back (dry run). Details: %s", exc)
        else:
            log.exception("Seed failed: %s", exc)
            raise SystemExit(1)
 