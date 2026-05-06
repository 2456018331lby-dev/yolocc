"""
YOLOCC Launcher — Visual Dashboard & Quick Actions
Run: python launch.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.layout import Layout
    from rich import box
except ImportError:
    print("Installing rich...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich import box

console = Console()

# ── Helpers ──────────────────────────────────────────────────────────────────

def banner():
    art = r"""
 ██╗   ██╗ ██████╗ ██╗      ██████╗  ██████╗
 ╚██╗ ██╔╝██╔═══██╗██║     ██╔════╝ ██╔════╝
  ╚████╔╝ ██║   ██║██║     ██║  ███╗██║  ███╗
   ╚██╔╝  ██║   ██║██║     ██║   ██║██║   ██║
    ██║   ╚██████╔╝███████╗╚██████╔╝╚██████╔╝
    ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝
    """
    console.print(art, style="bold cyan")
    console.print("  Garbage Classification Detection System v2.0", style="bold white")
    console.print("  " + "─" * 50, style="dim")


def check_status():
    """Check project status."""
    status = {}

    # Dataset
    dataset_path = PROJECT_ROOT / "data" / "dataset" / "images" / "train"
    if dataset_path.exists():
        count = len(list(dataset_path.glob("*.jpg")))
        status["dataset"] = f"{count} images" if count > 0 else "empty"
    else:
        status["dataset"] = "not created"

    # Trained model
    best_pt = PROJECT_ROOT / "results" / "runs" / "train" / "garbage_detect" / "weights" / "best.pt"
    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        status["model_pt"] = f"{size_mb:.1f} MB"
    else:
        status["model_pt"] = "not trained"

    # ONNX model
    best_onnx = PROJECT_ROOT / "results" / "runs" / "train" / "garbage_detect" / "weights" / "best.onnx"
    if best_onnx.exists():
        size_mb = best_onnx.stat().st_size / (1024 * 1024)
        status["model_onnx"] = f"{size_mb:.1f} MB"
    else:
        status["model_onnx"] = "not exported"

    # Weights dir
    weights_dir = PROJECT_ROOT / "weights"
    if weights_dir.exists():
        weight_files = [f for f in weights_dir.iterdir() if f.is_file() and f.suffix in (".pt", ".onnx")]
        status["weights"] = f"{len(weight_files)} files" if weight_files else "empty"
    else:
        status["weights"] = "empty"

    return status


def show_status():
    """Display project status panel."""
    status = check_status()

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Component", style="cyan", width=20)
    table.add_column("Status", width=30)

    # Dataset
    ds_style = "green" if "images" in status["dataset"] else "yellow"
    table.add_row("Dataset", Text(status["dataset"], style=ds_style))

    # Model PT
    pt_style = "green" if "MB" in status["model_pt"] else "yellow"
    table.add_row("Model (.pt)", Text(status["model_pt"], style=pt_style))

    # Model ONNX
    onnx_style = "green" if "MB" in status["model_onnx"] else "yellow"
    table.add_row("Model (.onnx)", Text(status["model_onnx"], style=onnx_style))

    # Weights
    w_style = "green" if "files" in status["weights"] and "0" not in status["weights"] else "yellow"
    table.add_row("Weights/", Text(status["weights"], style=w_style))

    console.print(Panel(table, title="[bold]Project Status[/bold]", border_style="blue"))


def run_cmd(cmd, desc):
    """Run a command with spinner."""
    console.print(f"\n[bold cyan]>[/] {desc}")
    console.print(f"  [dim]{cmd}[/dim]\n")
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if result.returncode == 0:
        console.print(f"  [bold green]OK[/] — {desc}")
    else:
        console.print(f"  [bold red]FAILED[/] — {desc}")
    return result.returncode == 0


# ── Actions ──────────────────────────────────────────────────────────────────

def action_generate_data():
    """Generate synthetic dataset."""
    num = IntPrompt.ask("Number of images", default=200)
    run_cmd(
        f'{sys.executable} scripts/prepare_dataset.py generate --num {num}',
        f"Generate {num} synthetic images"
    )


def action_validate_data():
    """Validate dataset."""
    run_cmd(
        f'{sys.executable} scripts/prepare_dataset.py validate',
        "Validate dataset format"
    )


def action_train():
    """Train model."""
    epochs = IntPrompt.ask("Epochs (10=quick test, 100=full)", default=10)
    device = Prompt.ask("Device (0=GPU, cpu=CPU)", default="cpu")

    run_cmd(
        f'{sys.executable} -m src.train --cfg configs/train_cfg.yaml --epochs {epochs} --device {device}',
        f"Train YOLOv8 ({epochs} epochs, {device})"
    )


def action_export():
    """Export model."""
    best_pt = PROJECT_ROOT / "results" / "runs" / "train" / "garbage_detect" / "weights" / "best.pt"
    if not best_pt.exists():
        console.print("[yellow]No trained model found. Train first.[/yellow]")
        return

    fmt = Prompt.ask("Format", choices=["onnx", "onnx+int8", "all"], default="onnx")

    if fmt == "onnx":
        run_cmd(
            f'{sys.executable} -m src.export_model --weights {best_pt} --format onnx',
            "Export ONNX model"
        )
    elif fmt == "onnx+int8":
        run_cmd(
            f'{sys.executable} -m src.export_model --weights {best_pt} --format onnx --int8 --data configs/garbage.yaml',
            "Export ONNX INT8 model"
        )
    else:
        run_cmd(
            f'{sys.executable} -m src.export_model --weights {best_pt} --format all',
            "Export all formats"
        )


def action_detect_webcam():
    """Webcam detection."""
    weights = find_weights()
    if not weights:
        console.print("[yellow]No model weights found.[/yellow]")
        return
    run_cmd(
        f'{sys.executable} -m src.inference --source 0 --weights {weights}',
        "Webcam detection (press Q to quit)"
    )


def action_detect_image():
    """Image detection."""
    weights = find_weights()
    if not weights:
        console.print("[yellow]No model weights found.[/yellow]")
        return

    source = Prompt.ask("Image path", default="test.jpg")
    if not Path(source).exists():
        console.print(f"[yellow]File not found: {source}[/yellow]")
        return

    save = str(Path(source).stem + "_result.jpg")
    run_cmd(
        f'{sys.executable} -m src.inference --source {source} --weights {weights} --save {save}',
        f"Detect objects in {source}"
    )
    console.print(f"  Result saved to: {save}")


def action_benchmark():
    """Run benchmark."""
    best_pt = PROJECT_ROOT / "results" / "runs" / "train" / "garbage_detect" / "weights" / "best.pt"
    best_onnx = PROJECT_ROOT / "results" / "runs" / "train" / "garbage_detect" / "weights" / "best.onnx"

    weights = []
    if best_pt.exists():
        weights.append(str(best_pt))
    if best_onnx.exists():
        weights.append(str(best_onnx))

    if not weights:
        console.print("[yellow]No model weights found. Train and export first.[/yellow]")
        return

    runs = IntPrompt.ask("Number of runs", default=50)
    run_cmd(
        f'{sys.executable} -m src.benchmark --weights {" ".join(weights)} --runs {runs}',
        f"Benchmark ({runs} runs)"
    )


def action_api_server():
    """Start REST API."""
    weights = find_weights()
    if not weights:
        console.print("[yellow]No model weights found.[/yellow]")
        return
    port = IntPrompt.ask("Port", default=8000)
    console.print(f"\n[bold green]Starting API server on port {port}...[/]")
    console.print(f"  Docs: http://localhost:{port}/docs")
    console.print(f"  Health: http://localhost:{port}/health")
    console.print("  Press Ctrl+C to stop\n")
    subprocess.run(
        f'{sys.executable} -m src.api --weights {weights} --port {port}',
        shell=True, cwd=str(PROJECT_ROOT)
    )


def action_streamlit():
    """Launch Streamlit demo."""
    console.print("\n[bold green]Launching Streamlit demo...[/]")
    console.print("  Open: http://localhost:8501")
    console.print("  Press Ctrl+C to stop\n")
    subprocess.run(
        "streamlit run deploy/app.py --server.headless true",
        shell=True, cwd=str(PROJECT_ROOT)
    )


def action_full_pipeline():
    """Run full pipeline."""
    console.print("\n[bold yellow]Full Pipeline: generate → validate → train → export → benchmark[/]")
    if not Confirm.ask("This will take a while. Continue?", default=True):
        return

    steps = [
        (f'{sys.executable} scripts/prepare_dataset.py generate --num 200', "Generate dataset"),
        (f'{sys.executable} scripts/prepare_dataset.py validate', "Validate dataset"),
        (f'{sys.executable} -m src.train --cfg configs/train_cfg.yaml --epochs 10 --device cpu', "Train model"),
        (f'{sys.executable} -m src.validate --weights results/runs/train/garbage_detect/weights/best.pt', "Evaluate model"),
        (f'{sys.executable} -m src.export_model --weights results/runs/train/garbage_detect/weights/best.pt --format onnx', "Export ONNX"),
    ]

    for cmd, desc in steps:
        if not run_cmd(cmd, desc):
            console.print(f"\n[red]Pipeline stopped at: {desc}[/red]")
            return

    console.print("\n[bold green]Full pipeline complete![/bold green]")


def action_show_config():
    """Show current configuration."""
    import yaml
    cfg_path = PROJECT_ROOT / "configs" / "train_cfg.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    table = Table(title="Training Configuration", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    for k, v in cfg.items():
        if k.startswith("#"):
            continue
        table.add_row(str(k), str(v))

    console.print(table)


def find_weights():
    """Find best available weights."""
    candidates = [
        PROJECT_ROOT / "results/runs/train/garbage_detect/weights/best.onnx",
        PROJECT_ROOT / "results/runs/train/garbage_detect/weights/best.pt",
        PROJECT_ROOT / "weights/best.onnx",
        PROJECT_ROOT / "weights/best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


# ── Main Menu ────────────────────────────────────────────────────────────────

def main():
    while True:
        console.clear()
        banner()
        show_status()

        console.print()
        menu = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        menu.add_column("Key", style="bold yellow", width=4)
        menu.add_column("Action", width=40)
        menu.add_column("Description", style="dim")

        menu.add_row("1", "Generate test dataset", "Create synthetic images for testing")
        menu.add_row("2", "Validate dataset", "Check label format & integrity")
        menu.add_row("3", "Train model", "Train YOLOv8 on dataset")
        menu.add_row("4", "Export model", "Convert to ONNX / INT8")
        menu.add_row("5", "Detect (webcam)", "Real-time detection via webcam")
        menu.add_row("6", "Detect (image)", "Run detection on a single image")
        menu.add_row("7", "Benchmark", "Compare inference speed")
        menu.add_row("8", "API server", "Start REST API (FastAPI)")
        menu.add_row("9", "Web demo", "Launch Streamlit UI")
        menu.add_row("0", "Full pipeline", "Run all steps end-to-end")
        menu.add_row("c", "Show config", "Display training config")
        menu.add_row("q", "Quit", "")

        console.print(Panel(menu, title="[bold]Quick Actions[/bold]", border_style="green"))

        choice = Prompt.ask(
            "\n[bold]Choose action[/bold]",
            choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "c", "q"],
            default="1",
        )

        actions = {
            "1": action_generate_data,
            "2": action_validate_data,
            "3": action_train,
            "4": action_export,
            "5": action_detect_webcam,
            "6": action_detect_image,
            "7": action_benchmark,
            "8": action_api_server,
            "9": action_streamlit,
            "0": action_full_pipeline,
            "c": action_show_config,
        }

        if choice == "q":
            console.print("\n[bold cyan]Bye![/bold cyan]")
            break

        action = actions.get(choice)
        if action:
            action()
            console.print()
            Prompt.ask("[dim]Press Enter to continue...[/dim]")


if __name__ == "__main__":
    main()
