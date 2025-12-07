"""Prime Intellect GPU provisioning CLI."""

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from .prime import PrimeClient

app = typer.Typer(help="Prime Intellect GPU provisioning")
console = Console()


def get_client() -> PrimeClient:
    """Get Prime client, exit on error."""
    try:
        return PrimeClient()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_cmd(
    gpu_type: str | None = typer.Argument(None, help="Filter by GPU type (e.g., A100_PCIE_40GB)"),
):
    """List available GPUs."""
    client = get_client()
    offers = client.list_availability(gpu_type=gpu_type)

    if not offers:
        console.print("[yellow]No GPUs available[/yellow]")
        return

    table = Table(title="Available GPUs")
    table.add_column("GPU Type", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Data Center")
    table.add_column("Price/hr", justify="right", style="yellow")
    table.add_column("Status")

    for offer in offers[:20]:
        price = f"${offer.price_per_hour:.2f}" if offer.price_per_hour else "N/A"
        table.add_row(
            offer.gpu_type,
            offer.provider,
            offer.data_center[:15],
            price,
            offer.stock_status,
        )

    console.print(table)
    if len(offers) > 20:
        console.print(f"\n[dim]Showing 20 of {len(offers)} offers[/dim]")


@app.command("pods")
def pods_cmd():
    """List your pods."""
    client = get_client()
    pods = client.list_pods()

    if not pods:
        console.print("[yellow]No pods found[/yellow]")
        console.print("Create one with: [cyan]infra create <name>[/cyan]")
        return

    table = Table(title="Your Pods")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column("Provider")
    table.add_column("SSH")

    for pod in pods:
        ssh = pod.ssh_command[:40] + "..." if pod.ssh_command and len(pod.ssh_command) > 40 else (pod.ssh_command or "-")
        table.add_row(
            pod.id[:12],
            pod.name,
            pod.status,
            f"{pod.gpu_type} x{pod.gpu_count}",
            pod.provider,
            ssh,
        )

    console.print(table)


@app.command("create")
def create_cmd(
    name: str = typer.Argument("gpu-pod", help="Pod name"),
    gpu_type: str = typer.Option("A100_PCIE_40GB", "--gpu", "-g", help="GPU type"),
    image: str = typer.Option("pytorch", "--image", "-i", help="Container image"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for pod to be ready"),
):
    """Create a new GPU pod."""
    client = get_client()

    console.print(f"\n[bold]Creating pod...[/bold]")
    console.print(f"  Name: {name}")
    console.print(f"  GPU: {gpu_type}")
    console.print(f"  Image: {image}\n")

    try:
        pod = client.create_pod(name=name, gpu_type=gpu_type, image=image)
    except Exception as e:
        console.print(f"[red]Failed to create pod:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]✓ Pod created![/green]")
    console.print(f"  ID: {pod.id}")
    console.print(f"  Status: {pod.status}")

    if wait:
        console.print("\n[dim]Waiting for pod to be ready...[/dim]")
        try:
            pod = client.wait_for_pod(pod.id)
            console.print(f"[green]✓ Pod ready![/green]")
            if pod.ssh_command:
                console.print(f"\nSSH: [cyan]{pod.ssh_command}[/cyan]")
        except TimeoutError:
            console.print("[yellow]Pod not ready yet. Check with: infra pods[/yellow]")
    else:
        console.print("\nCheck status with: [cyan]infra pods[/cyan]")


@app.command("delete")
def delete_cmd(
    pod_id: str = typer.Argument(..., help="Pod ID to delete"),
):
    """Delete a pod."""
    client = get_client()

    console.print(f"\n[bold]Deleting pod {pod_id}...[/bold]")

    try:
        client.delete_pod(pod_id)
        console.print(f"[green]✓ Pod deleted![/green]")
    except Exception as e:
        console.print(f"[red]Failed to delete pod:[/red] {e}")
        raise typer.Exit(1)


@app.command("ssh")
def ssh_cmd(
    pod_id: str = typer.Argument(..., help="Pod ID to SSH into"),
):
    """Get SSH command for a pod."""
    client = get_client()
    pod = client.get_pod(pod_id)

    if not pod:
        # Try partial match
        pods = client.list_pods()
        matches = [p for p in pods if p.id.startswith(pod_id)]
        if len(matches) == 1:
            pod = matches[0]
        elif len(matches) > 1:
            console.print(f"[yellow]Multiple pods match '{pod_id}':[/yellow]")
            for p in matches:
                console.print(f"  {p.id} ({p.name})")
            raise typer.Exit(1)
        else:
            console.print(f"[red]Pod not found: {pod_id}[/red]")
            raise typer.Exit(1)

    if pod.ssh_command:
        console.print(pod.ssh_command)
    else:
        console.print(f"[yellow]Pod {pod.id} has no SSH command (status: {pod.status})[/yellow]")


def main():
    app()


if __name__ == "__main__":
    main()
