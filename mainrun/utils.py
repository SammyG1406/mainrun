# Devcontainer environment guard and structured JSON logging helper.

import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm


## Exits with an instructions message if the required devcontainer marker is absent.
def _check_devcontainer():
    if not all([
        Path("/root/.mainrun").exists()
    ]):
        os.system('cls' if os.name == 'nt' else 'clear')

        print("""
🚨 DEVCONTAINER REQUIRED 🚨

This project must run in its devcontainer for:

✓ Assessment submission  ✓ Metrics collection  ✓ Review process

Setup Instructions:

📖 https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container

📋 IMPORTANT: Read README.md for Mainrun instructions and review process

☠️☠️  Running outside devcontainer = broken submission & metrics  ☠️☠️
        """)
        sys.exit(1)

_check_devcontainer()


## Opens log_file for writing and returns a DualLogger that mirrors output to stdout.
def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = open(log_file, 'w')

    ### Writes each event as a JSON line to file and as formatted text via tqdm.
    class DualLogger:
        #### Stores the open file handle used by all subsequent log calls.
        def __init__(self, file_handler):
            self.file_handler = file_handler

        #### Serialises the event to a JSON log line and prints if prnt=True.
        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()

            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)

    return DualLogger(file_handler)
