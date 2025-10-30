#!/usr/bin/env python3
# train.py - Wrapper script for flexible YOLOv8 training (Detect, Segment, etc.)

import argparse, json, subprocess, sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# UTILITY FUNCTIONS 

def ensure_dir(p: Path):
    """Ensures a directory exists"""
    p.mkdir(parents=True, exist_ok=True)

def save_params(params: dict, out_dir: Path):
    """Saves training parameters to a JSON file"""
    ensure_dir(out_dir)
    serializable_params = {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}
    with open(out_dir / "params.json", "w") as f:
        json.dump(serializable_params, f, indent=2)

def run_cmd(cmd: List[str], log_file: Optional[Path] = None):
    """
    Executes a shell command and streams output to console and log file, 
    handling potential encoding issues.
    """
    print("[CMD]", " ".join(cmd))
    
    # Open log file using UTF-8 encoding
    lf = open(log_file, "a", encoding="utf-8") if log_file else None
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )
        
        # Stream output
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush() 
            if lf: 
                lf.write(line)
                lf.flush()
        
        proc.wait()
        
    finally:
        if lf: lf.close()
        
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with return code {proc.returncode}. Check log file.")


# MAIN TRAINING LOGIC

def start_yolov8_training(args_obj):
    """
    Configures and starts the YOLOv8 training process based on args_obj.
    Requires args_obj to have a 'task' attribute (e.g., 'detect', 'segment').
    """
    args = args_obj

    # 1. Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.project) / (args.name or timestamp)
    ensure_dir(exp_dir)
    save_params(vars(args), exp_dir)
    log_file = exp_dir / "train.log"

    # 2. Build the YOLO CLI command
    task_name = getattr(args, 'task', 'detect') 
    cmd = ["yolo", f"task={task_name}", "mode=train"] 

    # Mandatory parameters
    cmd += [
        f"data={args.data}", 
        f"model={args.model}", 
        f"epochs={args.epochs}", 
        f"imgsz={args.imgsz}", 
        f"batch={args.batch}"
    ]

    # Optional parameters
    if getattr(args, 'patience', None) is not None:
        cmd += [f"patience={args.patience}"]
    if getattr(args, 'device', ''):
        cmd += [f"device={args.device}"]
    if getattr(args, 'exist_ok', False):
        cmd += ["exist_ok=True"]

    # Worker handling (reduce to prevent Windows paging issues)
    workers_val = getattr(args, 'workers', None)
    if workers_val is None:
        cmd += ["workers=0"]
    else:
        cmd += [f"workers={workers_val}"]

    # Project and Name
    cmd += [f"project={args.project}", f"name={args.name}"]

    if getattr(args, 'extra', ''):
        cmd += args.extra.split()

    # 3. Execute training command
    try:
        run_cmd(cmd, log_file=log_file)
    except RuntimeError as e:
        print(f"\n[ERROR] Training command failed. Check log file: {log_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)

    print(f"\n[Done] Training finished successfully. Check logs & weights in: {exp_dir}")
    return exp_dir