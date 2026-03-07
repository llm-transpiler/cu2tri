#!/usr/bin/env python3
"""Auto-detect GPU environment and generate NVGPU configuration file.

This script detects all available GPUs on the system using nvidia-smi/pynvml
and generates a YAML configuration file for the NVGPU server.

Usage:
    python -m server.nvgpu.tools.gpu_detect
    python -m server.nvgpu.tools.gpu_detect --output custom_config.yml
    python -m server.nvgpu.tools.gpu_detect --max-tasks 5 --mode exclusive
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def detect_gpus_nvidia_smi():
    """Detect GPUs using nvidia-smi command.

    Returns:
        List of GPU info dicts or None if nvidia-smi not available
    """
    try:
        # Get GPU details first
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                idx = int(parts[0])
                name = parts[1]
                uuid = parts[2]
                memory_mb = int(parts[3])

                gpus.append({
                    'logical_id': idx,
                    'nvidia_smi_id': idx,
                    'cuda_visible_id': idx,
                    'name': name,
                    'uuid': uuid,
                    'memory_gb': round(memory_mb / 1024, 2),
                })

        return gpus

    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure NVIDIA drivers are installed.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return None


def detect_gpus_pynvml():
    """Detect GPUs using pynvml library.

    Returns:
        List of GPU info dicts or None if pynvml not available
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            print("No GPUs detected via NVML")
            return []

        gpus = []

        for idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gb = round(memory_info.total / (1024**3), 2)

            gpus.append({
                'logical_id': idx,
                'nvidia_smi_id': idx,
                'cuda_visible_id': idx,
                'name': name,
                'uuid': uuid,
                'memory_gb': memory_gb,
            })

        pynvml.nvmlShutdown()
        return gpus

    except ImportError:
        print("pynvml not available. Install with: pip install nvidia-ml-py3")
        return None
    except Exception as e:
        print(f"Error detecting GPUs with pynvml: {e}")
        return None


def generate_yaml_config(gpus, default_mode="shared", memory_threshold=0.75,
                         max_concurrent_tasks=3, hostname=None):
    """Generate YAML configuration content.

    Args:
        gpus: List of GPU info dicts
        default_mode: Default GPU mode (shared/exclusive)
        memory_threshold: Memory threshold (0.0-1.0)
        max_concurrent_tasks: Max concurrent tasks per GPU
        hostname: Hostname for config filename

    Returns:
        YAML content as string
    """
    lines = [
        "# Auto-generated NVGPU GPU configuration",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total GPUs: {len(gpus)}",
        "",
    ]

    # Add hostname comment if provided
    if hostname:
        lines.append(f"# Hostname: {hostname}")
        lines.append("")

    lines.append("gpus:")
    for gpu in gpus:
        lines.extend([
            f"  - logical_id: {gpu['logical_id']}",
            f"    nvidia_smi_id: {gpu['nvidia_smi_id']}",
            f"    cuda_visible_id: {gpu['cuda_visible_id']}",
            f"    name: \"{gpu['name']}\"",
            f"    uuid: \"{gpu['uuid']}\"",
            f"    memory_gb: {gpu['memory_gb']}",
            "    enabled: true",
            f"    default_mode: \"{default_mode}\"",
            f"    memory_threshold: {memory_threshold}",
            f"    max_concurrent_tasks: {max_concurrent_tasks}",
            "",
        ])

    lines.extend([
        "server:",
        "  auto_register_gpus: true",
    ])

    return '\n'.join(lines)


def get_hostname():
    """Get system hostname."""
    try:
        import socket
        return socket.gethostname()
    except:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect GPUs and generate NVGPU configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output config file path (default: auto-generated hostname-based)"
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Base directory for config files (default: server/nvgpu/configs/gpu_resources/)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["shared", "exclusive"],
        default="shared",
        help="Default GPU mode"
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=0.75,
        help="Memory threshold (0.0-1.0, default: 0.75)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=3,
        help="Max concurrent tasks per GPU (default: 3)"
    )
    parser.add_argument(
        "--method",
        choices=["auto", "nvidia-smi", "pynvml"],
        default="auto",
        help="GPU detection method"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config to stdout instead of writing file"
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the nvgpu-server startup command after generating config"
    )

    args = parser.parse_args()

    # Define project root for startup commands
    project_root = Path("/cu2tri")

    # Detect GPUs
    gpus = None

    if args.method == "auto":
        gpus = detect_gpus_pynvml()
        if gpus is None:
            gpus = detect_gpus_nvidia_smi()
    elif args.method == "pynvml":
        gpus = detect_gpus_pynvml()
    else:  # nvidia-smi
        gpus = detect_gpus_nvidia_smi()

    if gpus is None:
        sys.exit(1)

    if not gpus:
        print("No GPUs detected. Exiting.")
        sys.exit(1)

    print(f"Detected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  GPU {gpu['logical_id']}: {gpu['name']} "
              f"({gpu['memory_gb']} GB, UUID: {gpu['uuid']})")

    # Generate YAML
    hostname = get_hostname()
    yaml_content = generate_yaml_config(
        gpus,
        default_mode=args.mode,
        memory_threshold=args.memory_threshold,
        max_concurrent_tasks=args.max_tasks,
        hostname=hostname
    )

    if args.dry_run:
        print("\n" + "="*60)
        print("Generated Configuration:")
        print("="*60)
        print(yaml_content)
        # Still print startup command in dry-run mode
        if args.print_command:
            print("\n" + "="*60)
            print("NVGPU Server Startup Command:")
            print("="*60)
            print(f"cd {project_root}")
            print(f"nvgpu-server --gpu-config <config_file_path>")
            print("="*60)
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
        config_name = output_path.name
    else:
        # Auto-generate filename based on hostname
        config_name = f"{hostname or 'local'}_gpu_config.yml"
        if args.config_dir:
            config_dir = Path(args.config_dir)
        else:
            # Default to project relative path
            config_dir = Path("server/nvgpu/configs/gpu_resources")
        output_path = config_dir / config_name

    output_path = output_path.expanduser().resolve()

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write config file
    with open(output_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nConfiguration written to: {output_path}")

    # Calculate relative path from project root for startup command
    project_root = Path("/cu2tri")
    try:
        rel_path = output_path.relative_to(project_root)
        startup_config = str(rel_path)
    except ValueError:
        # Not under project root, use absolute path
        startup_config = str(output_path)

    # Print startup command
    if args.print_command:
        print("\n" + "="*60)
        print("NVGPU Server Startup Command:")
        print("="*60)
        print(f"cd {project_root}")
        print(f"nvgpu-server --gpu-config {startup_config}")
        print("="*60)

    return str(output_path)


if __name__ == "__main__":
    main()
