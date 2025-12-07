"""Prime Intellect Compute API client."""

import os
from dataclasses import dataclass
from typing import Any

import httpx

API_BASE = "https://api.primeintellect.ai/api/v1"


@dataclass
class GpuOffer:
    """Available GPU offer."""

    cloud_id: str
    gpu_type: str
    gpu_count: int
    provider: str
    data_center: str
    country: str
    price_per_hour: float | None
    stock_status: str
    images: list[str]
    socket: str | None
    security: str | None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "GpuOffer":
        # Handle both old and new API response formats
        prices = data.get("prices", {})
        price = prices.get("onDemand") if isinstance(prices, dict) else data.get("price")
        return cls(
            cloud_id=data.get("cloudId", ""),
            gpu_type=data.get("gpuType", ""),
            gpu_count=data.get("gpuCount", 1),
            provider=data.get("provider", ""),
            data_center=data.get("dataCenter", ""),
            country=data.get("country", ""),
            price_per_hour=price,
            stock_status=data.get("stockStatus", data.get("availability", "")),
            images=data.get("images", []),
            socket=data.get("socket"),
            security=data.get("security"),
        )


@dataclass
class Pod:
    """A provisioned GPU pod."""

    id: str
    name: str
    status: str
    gpu_type: str
    gpu_count: int
    provider: str
    ssh_command: str | None
    ip: str | None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Pod":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=data.get("status", ""),
            gpu_type=data.get("gpuType", ""),
            gpu_count=data.get("gpuCount", 1),
            provider=data.get("provider", ""),
            ssh_command=data.get("sshCommand"),
            ip=data.get("ip"),
        )


class PrimeClient:
    """Prime Intellect Compute API client."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("PRIME_COMPUTE_KEY") or os.environ.get("PRIME_API_KEY")
        if not self.api_key:
            raise ValueError(
                "PRIME_COMPUTE_KEY or PRIME_API_KEY environment variable required.\n"
                "Get your Compute API key from: https://app.primeintellect.ai/dashboard/account\n"
                "Make sure it has 'Instances -> Read and write' permission"
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_availability(self, gpu_type: str | None = None, gpu_count: int = 1) -> list[GpuOffer]:
        """List available GPU offers."""
        params = {"gpu_count": str(gpu_count)}
        if gpu_type:
            params["gpu_type"] = gpu_type

        with httpx.Client() as client:
            response = client.get(f"{API_BASE}/availability/gpus", params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()

        # API returns {items: [...], totalCount: N}
        items = data.get("items", data) if isinstance(data, dict) else data
        return [GpuOffer.from_api(offer) for offer in items]

    def list_pods(self) -> list[Pod]:
        """List user's pods."""
        with httpx.Client() as client:
            response = client.get(f"{API_BASE}/pods/", headers=self.headers)
            response.raise_for_status()
            data = response.json()

        return [Pod.from_api(pod) for pod in data.get("pods", [])]

    def get_pod(self, pod_id: str) -> Pod | None:
        """Get a specific pod by ID."""
        pods = self.list_pods()
        for pod in pods:
            if pod.id == pod_id:
                return pod
        return None

    def create_pod(
        self,
        name: str = "gpu-pod",
        gpu_type: str = "A100_PCIE_40GB",
        image: str = "pytorch",
    ) -> Pod:
        """Create a new GPU pod.

        Args:
            name: Pod name
            gpu_type: GPU type (e.g., A100_PCIE_40GB, H100_80GB)
            image: Container image (e.g., pytorch, vllm_llama_8b)

        Returns:
            Created Pod instance
        """
        # Find availability for this GPU type
        offers = self.list_availability(gpu_type=gpu_type)
        if not offers:
            raise ValueError(f"No {gpu_type} GPUs available")

        # Pick first available offer
        offer = next((o for o in offers if o.stock_status == "Available"), offers[0])

        # Check if image is available, fallback if not
        selected_image = image if image in offer.images else (offer.images[0] if offer.images else "pytorch")

        body = {
            "pod": {
                "name": name,
                "cloudId": offer.cloud_id,
                "gpuType": offer.gpu_type,
                "socket": offer.socket,
                "gpuCount": offer.gpu_count,
                "image": selected_image,
                "dataCenterId": offer.data_center,
                "country": offer.country,
                "security": offer.security or "secure_cloud",
            },
            "provider": {
                "type": offer.provider,
            },
        }

        with httpx.Client() as client:
            response = client.post(f"{API_BASE}/pods/", headers=self.headers, json=body)
            response.raise_for_status()
            data = response.json()

        return Pod.from_api(data)

    def delete_pod(self, pod_id: str) -> bool:
        """Delete a pod."""
        with httpx.Client() as client:
            response = client.delete(f"{API_BASE}/pods/{pod_id}", headers=self.headers)
            response.raise_for_status()
        return True

    def wait_for_pod(self, pod_id: str, timeout: int = 300, poll_interval: int = 10) -> Pod:
        """Wait for a pod to be ready.

        Args:
            pod_id: Pod ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Time between status checks

        Returns:
            Ready Pod instance

        Raises:
            TimeoutError: If pod doesn't become ready within timeout
        """
        import time

        start = time.time()
        while time.time() - start < timeout:
            pod = self.get_pod(pod_id)
            if pod and pod.status.lower() in ("running", "ready"):
                return pod
            time.sleep(poll_interval)

        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")
