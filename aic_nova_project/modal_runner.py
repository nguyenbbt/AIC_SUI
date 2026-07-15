
import modal
import os
import sys
import subprocess
import threading
import time
import signal
import socket


def _print_metrics():
    import subprocess as _sp, os as _os
    print()
    print("=" * 65)
    print("  CONTAINER HARDWARE METRICS")
    print("=" * 65)
    try:
        r = _sp.run(["nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,"
            "utilization.memory,temperature.gpu,power.draw,power.limit,driver_version",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for i, line in enumerate(r.stdout.strip().split("\n")):
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 10:
                    print(f"  GPU {i}: {p[0]}")
                    print(f"    Driver:       {p[9]}")
                    print(f"    VRAM:         {p[2]} / {p[1]} MiB  (free: {p[3]} MiB)")
                    print(f"    GPU Util:     {p[4]}%")
                    print(f"    Mem Util:     {p[5]}%")
                    print(f"    Temperature:  {p[6]} C")
                    print(f"    Power:        {p[7]} / {p[8]} W")
    except Exception as e:
        print(f"  [GPU] error: {e}")
    try:
        with open("/proc/cpuinfo") as _f:
            _ci = _f.read()
        _cores = _ci.count("processor\t:")
        _mn = ""
        for _ln in _ci.split("\n"):
            if _ln.startswith("model name"):
                _mn = _ln.split(":", 1)[1].strip()
                break
        print(f"  CPU: {_mn}  ({_cores} cores)")
    except:
        pass
    try:
        with open("/proc/meminfo") as _f:
            _mi = {}
            for _ln in _f:
                if ":" in _ln:
                    _k, _v = _ln.split(":", 1)
                    _mi[_k.strip()] = _v.strip()
        _t = int(_mi.get("MemTotal", "0 kB").split()[0])
        _a = int(_mi.get("MemAvailable", "0 kB").split()[0])
        _u = _t - _a
        print(f"  Memory: {_u / 1048576:.1f} / {_t / 1048576:.1f} GB  ({_u * 100 // _t}% used)")
    except:
        pass
    try:
        _st = _os.statvfs("/")
        _td = _st.f_blocks * _st.f_frsize
        _fd = _st.f_bavail * _st.f_frsize
        _ud = _td - _fd
        print(f"  Disk:   {_ud / (1024**3):.1f} / {_td / (1024**3):.1f} GB  (free: {_fd / (1024**3):.1f} GB)")
    except:
        pass
    print("=" * 65)
    print()

def _monitor_metrics(interval=30):
    import subprocess as _sp, time as _time, threading as _th
    def _loop():
        while True:
            _time.sleep(interval)
            try:
                r = _sp.run(["nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    for line in r.stdout.strip().split("\n"):
                        p = [x.strip() for x in line.split(",")]
                        if len(p) >= 5:
                            print(f"[METRICS] GPU {p[0]}% | VRAM {p[1]}/{p[2]} MiB | {p[3]}C | {p[4]}W", flush=True)
            except:
                pass
            try:
                with open("/proc/loadavg") as _f:
                    _la = _f.read().split()[:3]
                print(f"[METRICS] CPU load {_la[0]} {_la[1]} {_la[2]}", flush=True)
            except:
                pass
    _th.Thread(target=_loop, daemon=True).start()


MAX_RESTARTS = 5  # per service

app = modal.App("m-gpux-compose")
workspace_volume = modal.Volume.from_name("m-gpux-compose-aic-nova-project-c536a3b21a", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install(['bash', 'build-essential', 'ca-certificates', 'curl', 'git', 'nano', 'net-tools', 'procps'])
    
    .pip_install_from_requirements("D:/Project/AI Challenge 2026/aic_nova_project/feature_extraction/asr_transcript/requirements.txt")
    .add_local_dir("D:/Project/AI Challenge 2026/aic_nova_project", remote_path="/workspace_seed", ignore=['.venv', 'venv', '__pycache__', '.git', 'node_modules', '.mypy_cache', '.pytest_cache', '*.egg-info', '.tox'])
    
)

# Service command registry
svc_commands = {
    "ocr_app": {"cmd": 'uvicorn app.main:app --host 0.0.0.0 --port 8000', "cwd": "/workspace", "port": 8000, "infra": False},
}

INFRA_ORDER = []
APP_ORDER = ['ocr_app']

def _prepare_workspace():
    os.makedirs("/workspace", exist_ok=True)
    subprocess.run(["cp", "-a", "/workspace_seed/.", "/workspace/"], check=False)
    workspace_volume.commit()

def _setup_volume_mounts():
    """Setup volume mounts (model_repository, etc.)."""
    pass

def _start_workspace_autocommit(interval=20):
    def _loop():
        while True:
            time.sleep(interval)
            try:
                workspace_volume.commit()
            except Exception:
                pass
    threading.Thread(target=_loop, daemon=True).start()

def _start_workspace_autoreload(interval=3):
    """Periodically reload the volume to pick up external changes (from compose sync)."""
    def _loop():
        while True:
            time.sleep(interval)
            try:
                workspace_volume.reload()
            except Exception:
                pass
    threading.Thread(target=_loop, daemon=True).start()

def _write_hosts():
    """Add service name entries to /etc/hosts for local resolution."""
    entries = [
    "127.0.0.1 ocr_app",
    ]
    with open("/etc/hosts", "a") as f:
        f.write("\n# m-gpux compose services\n")
        for entry in entries:
            f.write(entry + "\n")
    print("[COMPOSE] /etc/hosts updated with service entries:", entries, flush=True)

def _setup_nginx():
    """Setup nginx config: use workspace copy if present, else generate minimal one."""
    import glob
    # Look for user's nginx config in workspace (common patterns)
    candidates = (
        glob.glob("/workspace/nginx/nginx.conf") +
        glob.glob("/workspace/nginx/*.conf") +
        glob.glob("/workspace/config/nginx*.conf") +
        glob.glob("/workspace/nginx.conf")
    )
    if candidates:
        conf_src = candidates[0]
        print(f"[COMPOSE] Using nginx config: {conf_src}", flush=True)
        subprocess.run(["cp", conf_src, "/tmp/nginx_compose.conf"], check=False)
    else:
        # Generate a minimal reverse-proxy config
        conf = (
            "user root;\n"
            "worker_processes auto;\n"
            "error_log /var/log/nginx/error.log warn;\n"
            "pid /tmp/nginx.pid;\n"
            "events { worker_connections 1024; }\n"
            "http {\n"
            "    access_log /var/log/nginx/access.log;\n"
            "    server {\n"
            "        listen 80;\n"
            "        location / {\n"
            "            proxy_pass http://127.0.0.1:8000;\n"
            "            proxy_set_header Host $$host;\n"
            "            proxy_set_header X-Real-IP $$remote_addr;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        with open("/tmp/nginx_compose.conf", "w") as f:
            f.write(conf)
        print("[COMPOSE] Generated default nginx proxy config on port 80", flush=True)
    # Ensure log dir exists and fix permissions
    os.makedirs("/var/log/nginx", exist_ok=True)
    # Fix nginx user issue on non-alpine
    subprocess.run(["sed", "-i", "s/^user nginx/user root/", "/tmp/nginx_compose.conf"], check=False)

def _setup_environment():
    """Set all environment variables from compose config."""
    pass

def _is_idle_command(cmd):
    """Return True if the command is a placeholder/idle command that won't bind a port."""
    import re
    cmd_stripped = cmd.strip().lower()
    idle_patterns = [
        r'^(bash|sh|/bin/bash|/bin/sh)(\s+-\w+)*\s*$',
        r'^sleep\s+',
        r'^tail\s+-f',
        r'^cat\s*$',
        r'^bash\s+-lc\s+.sleep\s+',
    ]
    for pattern in idle_patterns:
        if re.match(pattern, cmd_stripped):
            return True
    return False

def _wait_for_port(port, timeout=30):
    """Block until a port is accepting connections or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False

def _start_service(name, procs):
    """Start a single service by name. Returns the Popen object."""
    info = svc_commands[name]
    cmd = info["cmd"]
    cwd = info["cwd"]
    print(f"[COMPOSE] Starting {name}: {cmd}", flush=True)
    log_path = f"/tmp/compose_{name}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    def _fwd():
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
        except Exception:
            pass
        finally:
            log_f.close()
    threading.Thread(target=_fwd, daemon=True).start()
    procs[name] = proc
    return proc

def _show_recent_log(name, n=20):
    """Print the last n lines from a service's log file."""
    log_path = f"/tmp/compose_{name}.log"
    try:
        with open(log_path) as f:
            lines = f.readlines()
        recent = lines[-n:] if len(lines) > n else lines
        if recent:
            print(f"[COMPOSE] Recent output from {name}:", flush=True)
            for line in recent:
                print(f"  | {line.rstrip()}", flush=True)
        else:
            print(f"[COMPOSE] No output captured from {name} yet", flush=True)
    except FileNotFoundError:
        print(f"[COMPOSE] No log file for {name}", flush=True)

def _supervisor_loop(procs, svc_commands, restart_counts):
    """Monitor all processes; auto-restart crashed ones up to MAX_RESTARTS."""
    while True:
        time.sleep(5)
        for name, proc in list(procs.items()):
            ret = proc.poll()
            if ret is not None:
                count = restart_counts.get(name, 0)
                if count >= MAX_RESTARTS:
                    if count == MAX_RESTARTS:  # print only once
                        print(f"[COMPOSE] FATAL: {name} crashed {MAX_RESTARTS} times. Giving up.", flush=True)
                        restart_counts[name] = count + 1
                    continue
                restart_counts[name] = count + 1
                print(f"[COMPOSE] {name} exited (code {ret}). Restarting ({count+1}/{MAX_RESTARTS})...", flush=True)
                time.sleep(2)  # back-off before restart
                _start_service(name, procs)

@app.function(image=image, gpu="A10G", timeout=86400, volumes={"/workspace": workspace_volume})
def run_compose():
    _print_metrics()
    _prepare_workspace()
    _setup_volume_mounts()
    _start_workspace_autocommit()
    _start_workspace_autoreload()

    # Setup hostname resolution and environment
    _write_hosts()
    _setup_environment()


    os.chdir("/workspace")
    procs = {}
    restart_counts = {}

    # --- Phase 1: Start infrastructure services and wait for ports ---
    for name in INFRA_ORDER:
        _start_service(name, procs)
        port = svc_commands[name]["port"]
        if port:
            print(f"[COMPOSE] Waiting for {name} on port {port}...", flush=True)
            if _wait_for_port(port, timeout=30):
                print(f"[COMPOSE] {name} ready (port {port})", flush=True)
            else:
                print(f"[COMPOSE] WARNING: {name} port {port} not responding after 30s", flush=True)
                _show_recent_log(name)
        else:
            time.sleep(2)

    # --- Phase 2: Start application services ---
    for name in APP_ORDER:
        _start_service(name, procs)
        port = svc_commands[name]["port"]
        cmd = svc_commands[name]["cmd"]
        if port and not _is_idle_command(cmd):
            print(f"[COMPOSE] Waiting for {name} on port {port}...", flush=True)
            if _wait_for_port(port, timeout=30):
                print(f"[COMPOSE] {name} ready (port {port})", flush=True)
            else:
                # Check if process already died
                if procs[name].poll() is not None:
                    print(f"[COMPOSE] ERROR: {name} exited immediately (code {procs[name].returncode})", flush=True)
                    _show_recent_log(name)
                else:
                    print(f"[COMPOSE] WARNING: {name} port {port} not responding (may still be starting)", flush=True)
                    _show_recent_log(name)
        else:
            # No port defined or idle command — background/worker service, just let it start
            time.sleep(2)
            if procs[name].poll() is not None:
                print(f"[COMPOSE] WARNING: {name} exited immediately (code {procs[name].returncode})", flush=True)
                _show_recent_log(name)
            else:
                print(f"[COMPOSE] {name} started (no port check)", flush=True)

    with modal.forward(8000, unencrypted=True) as tunnel:
        print("\n" + "=" * 60)
        print(f"[COMPOSE READY] {tunnel.url}")
        print(f"  Main service: ocr_app (port 8000)")
        print(f"  Services running: {', '.join(procs.keys())}")
        print(f"  Workspace: /workspace")
        print(f"  Volume: m-gpux-compose-aic-nova-project-c536a3b21a")
        print("=" * 60 + "\n", flush=True)

        _supervisor_loop(procs, svc_commands, restart_counts)

