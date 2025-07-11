"""
Azure services integration.

This module provides clients for Azure Document Intelligence, Azure OpenAI,
Azure SQL Database, and Azure Blob Storage.
"""

from pathlib import Path

# Import AzWrap for Azure services
from azwrap import (
    Identity,
)
"""Azure Identity wrapper with fallback authentication.

This module provides a modified version of the Identity class that uses DefaultAzureCredential
with a fallback to ClientSecretCredential for more flexible authentication options.
"""

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.mgmt.resource import SubscriptionClient
# from azwrap import Identity as BaseIdentity

class DefaultIdentity(Identity):
    """Identity class that uses DefaultAzureCredential with fallback to ClientSecretCredential."""

    def __init__(self):
        """Initialize identity using DefaultAzureCredential with fallback.

        Authentication is attempted in the following order:
        1. DefaultAzureCredential (which includes):
           - Environment variables
           - Managed Identity
           - Visual Studio Code credentials
           - Azure CLI credentials
           - Azure PowerShell credentials
        2. Fallback to ClientSecretCredential if environment variables are present
        """
        try:
            self.credential = DefaultAzureCredential()
            # Test the credential by trying to list subscriptions
            SubscriptionClient(self.credential).subscriptions.list()
        except (ClientAuthenticationError, Exception):
            # Fallback to ClientSecretCredential if config variables are present
            if all([AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET]):
                self.credential = ClientSecretCredential(
                    tenant_id=AZURE_TENANT_ID,
                    client_id=AZURE_CLIENT_ID,
                    client_secret=AZURE_CLIENT_SECRET
                )
            else:
                raise Exception(
                    "Authentication failed with DefaultAzureCredential and "
                    "configuration variables for ClientSecretCredential are not set"
                )

        self.subscription_client = SubscriptionClient(self.credential)

    def get_credential(self):
        """Get the credential instance (either DefaultAzureCredential or ClientSecretCredential)."""
        return self.credential

from config import (
    # Add AzWrap configuration
    AZURE_TENANT_ID,
    AZURE_CLIENT_ID,
    AZURE_CLIENT_SECRET,
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP_NAME,
    # AZURE_OPENAI_SERVICE,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_SERVICE_NAME

)

class OpenAIService:
    """Client for Azure OpenAI service using AzWrap."""

    def __init__(self):
        """Initialize the Azure OpenAI client using AzWrap."""
        # Initialize AzWrap Identity
        self.identity = DefaultIdentity()
        # self.identity = Identity(
        #     tenant_id=AZURE_TENANT_ID,
        #     client_id=AZURE_CLIENT_ID,
        #     client_secret=AZURE_CLIENT_SECRET
        # )

        # Get subscription and resource group
        self.subscription = self.identity.get_subscription(AZURE_SUBSCRIPTION_ID)
        # print(100 * "-")
        # print([g.name.lower() for g in self.subscription.resource_client.resource_groups.list()])
        # print(100 * "-")
        self.resource_group = self.subscription.get_resource_group(AZURE_RESOURCE_GROUP_NAME)

        # Get OpenAI service
        self.ai_service = self.resource_group.get_ai_service(AZURE_OPENAI_SERVICE_NAME)

        # Get OpenAI client
        self.client = self.ai_service.get_OpenAIClient(api_version=AZURE_OPENAI_API_VERSION).openai_client

        # self.deployment = OPENAI_DEPLOYMENT
        # # logger.info("Azure OpenAI client initialized using AzWrap")

        # """Initialize the Azure OpenAI client."""
        # self.endpoint = OPENAI_ENDPOINT
        # self.key = OPENAI_KEY
        # if not self.endpoint or not self.key:
        #     raise ValueError("Azure OpenAI endpoint and key must be provided")
        # self.client = AzureOpenAI(
        #             azure_endpoint = self.endpoint,
        #             api_key=self.key,
        #             azure_deployment=self.deployment,
        #             api_version=OPENAI_API_VERSION
        #         )

