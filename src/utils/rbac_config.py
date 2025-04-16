import os
import logging
from typing import Dict, List
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.collections.classes.tenants import TenantCreate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Tenant definitions
TENANTS = {
    "bain": {
        "name": "Bain & Company",
        "allowed_roles": ["bain_admin", "bain_reader", "bain_writer"],
    },
    "bcg": {
        "name": "Boston Consulting Group",
        "allowed_roles": ["bcg_admin", "bcg_reader", "bcg_writer"],
    },
    "mck": {
        "name": "McKinsey & Company",
        "allowed_roles": ["mck_admin", "mck_reader", "mck_writer"],
    },
}

# Role definitions with tenant-specific permissions
ROLES = {
    # Bain roles
    "bain_admin": {
        "permissions": ["read", "write", "delete", "create", "update"],
        "classes": ["DocumentChunk"],
        "tenant": "bain",
    },
    "bain_reader": {
        "permissions": ["read"],
        "classes": ["DocumentChunk"],
        "tenant": "bain",
    },
    "bain_writer": {
        "permissions": ["read", "write", "create"],
        "classes": ["DocumentChunk"],
        "tenant": "bain",
    },
    # BCG roles
    "bcg_admin": {
        "permissions": ["read", "write", "delete", "create", "update"],
        "classes": ["DocumentChunk"],
        "tenant": "bcg",
    },
    "bcg_reader": {
        "permissions": ["read"],
        "classes": ["DocumentChunk"],
        "tenant": "bcg",
    },
    "bcg_writer": {
        "permissions": ["read", "write", "create"],
        "classes": ["DocumentChunk"],
        "tenant": "bcg",
    },
    # McKinsey roles
    "mck_admin": {
        "permissions": ["read", "write", "delete", "create", "update"],
        "classes": ["DocumentChunk"],
        "tenant": "mck",
    },
    "mck_reader": {
        "permissions": ["read"],
        "classes": ["DocumentChunk"],
        "tenant": "mck",
    },
    "mck_writer": {
        "permissions": ["read", "write", "create"],
        "classes": ["DocumentChunk"],
        "tenant": "mck",
    },
}


def setup_rbac(client: WeaviateClient) -> None:
    """Set up RBAC roles and permissions with tenant isolation"""
    try:
        collection = client.collections.get("DocumentChunk")

        # Create tenants if they don't exist
        for tenant_id, tenant_info in TENANTS.items():
            try:
                # Check if tenant exists first
                try:
                    collection.tenants.get(tenant_id)
                    logging.info(f"Tenant {tenant_id} already exists")
                except:
                    # Create tenant if it doesn't exist
                    collection.tenants.create(tenant_id)
                    logging.info(f"Created tenant: {tenant_info['name']}")
            except Exception as e:
                if "already exists" in str(e):
                    logging.info(f"Tenant {tenant_id} already exists")
                else:
                    logging.error(f"Error creating tenant {tenant_id}: {str(e)}")
                    raise

        # Create roles if they don't exist
        for role_name, role_config in ROLES.items():
            try:
                # Check if role exists
                try:
                    client.roles.get(role_name)
                    logging.info(f"Role already exists: {role_name}")
                except:
                    # Create role with permissions
                    client.roles.create(
                        name=role_name,
                        permissions=role_config["permissions"],
                        tenant=role_config["tenant"],
                    )
                    logging.info(
                        f"Created role: {role_name} for tenant {role_config['tenant']}"
                    )
            except Exception as e:
                if "already exists" in str(e):
                    logging.info(f"Role {role_name} already exists")
                else:
                    logging.error(f"Error creating role {role_name}: {str(e)}")
                    raise

        # Log successful setup
        logging.info("Successfully set up RBAC with tenants and roles")

    except Exception as e:
        logging.error(f"Error setting up RBAC: {str(e)}")
        raise


def create_user(
    client: WeaviateClient, username: str, password: str, roles: List[str], tenant: str
) -> None:
    """Create a new user with specified roles for a specific tenant"""
    try:
        if tenant not in TENANTS:
            raise ValueError(f"Invalid tenant: {tenant}")

        # Validate roles belong to the tenant
        for role in roles:
            if not role.startswith(tenant + "_"):
                raise ValueError(f"Role {role} does not belong to tenant {tenant}")

        if not client.rbac.user_exists(username):
            client.rbac.create_user(username=username, password=password, roles=roles)
            logging.info(
                f"Created user: {username} with roles: {roles} for tenant {tenant}"
            )
        else:
            logging.warning(f"User already exists: {username}")
    except Exception as e:
        logging.error(f"Error creating user {username}: {str(e)}")
        raise


def get_user_permissions(client: WeaviateClient, username: str) -> Dict:
    """Get permissions for a specific user"""
    try:
        return client.rbac.get_user_permissions(username)
    except Exception as e:
        logging.error(f"Error getting permissions for user {username}: {str(e)}")
        raise


def validate_user_access(
    client: WeaviateClient,
    username: str,
    required_permission: str,
    class_name: str,
    tenant: str,
) -> bool:
    """Validate if a user has the required permission for a class within their tenant"""
    try:
        # Get user permissions
        permissions = (
            client.collections.get(class_name)
            .authorization.users.get(username)
            .permissions
        )

        # Check if user has permission for the specific tenant
        return any(
            perm["permission"] == required_permission
            and perm["class"] == class_name
            and perm.get("tenant") == tenant
            for perm in permissions
        )
    except Exception as e:
        logging.error(f"Error validating access for user {username}: {str(e)}")
        return False
