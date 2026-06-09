# Pilotless Poses

Batch pose detection web app using [VitPose](https://github.com/ViTAE-Transformer/ViTPose) (Vision Transformer for Pose Estimation). Point it at a folder of images, and it detects human pose keypoints for every person in every image — with real-time progress via WebSocket and an interactive skeleton overlay preview.

## Architecture

```
Browser (Vanilla JS) ←→ WebSocket ←→ Deno Server ←→ Python (VitPose inference)
                                          │
                                    JSON results on disk
```

- **Backend:** Deno.js — HTTP + WebSocket server, spawns Python child processes
- **ML inference:** Python — VitPose (COCO 17-keypoint format)
- **Frontend:** Vanilla JS — folder scan page + pose preview page with canvas skeleton overlay
- **Storage:** One JSON file per image + a `manifest.json`

## Docker isolation (recommended)

Claude Code runs inside a container so the `--dangerously-skip-permissions` flag cannot reach outside the workspace. You edit files on the host with VS Code — changes sync instantly via bind mount.

### Quick start

```bash
# Build and start the container
docker compose up -d --build

# Enter the container and launch Claude Code
./enter-claude.sh

# Or pass a one-shot prompt
./enter-claude.sh -p "Your prompt here"
```

### How it works

```
Bare Metal (VS Code)           Docker Container
─────────────────────          ─────────────────
~/code/pilotless_poses/  ←──→  /workspace/     (bind mount, real-time)
                                    │
Only /workspace is shared           ├── Node.js + Claude Code
Container CANNOT reach              ├── Deno + Python
anything else on host               └── Isolated network
```

### Daily workflow

```bash
# Start the container (once per boot, or leave it running)
docker compose up -d

# Edit files normally in VS Code — changes appear instantly in /workspace

# When you want Claude Code, run:
./enter-claude.sh

# Stop when done for the day
docker compose down
```

The container persists between Claude sessions. `enter-claude.sh` auto-detects whether the container is running and starts it if needed.

### GPU support (optional)

If you have an NVIDIA GPU and `nvidia-container-toolkit` installed, uncomment the `deploy` block in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Build arguments

| Arg | Default | Purpose |
|---|---|---|
| `HOST_UID` | `1000` | Match your host user ID (set in `.env`) |
| `HOST_GID` | `1000` | Match your host group ID (set in `.env`) |

The container creates a `developer` user with these IDs so files created inside the container are owned by you on the host — no permission headaches.

### What the container can and cannot do

| Can | Cannot |
|---|---|
| Read/write files in `/workspace` | Touch files outside `/workspace` |
| Run shell commands inside the container | Execute commands on your host OS |
| Install packages inside the container | Modify host system configuration |
| Access the internet | Access host network services unless exposed |

## Manual setup (without Docker)

### Prerequisites

- **Deno** ≥ 2.7
- **Python** ≥ 3.10
- **Node.js** ≥ 22 (for Claude Code)
- **Claude Code CLI** (`npm install -g @anthropic-ai/claude-code`)

### Install

```bash
# Python dependencies
pip install -r py-requirements.txt
mim install mmcv mmpose mmdet mmengine

# Deno caches dependencies on first run — no install step needed
```

### Run

```bash
# Start the server
deno run --allow-net --allow-read --allow-write --allow-run server.ts
```

Then open `http://localhost:8000` in your browser.

### Project structure

```
pilotless_poses/
├── server.ts            # Deno HTTP + WebSocket server
├── inference.py         # VitPose inference script
├── public/
│   ├── index.html       # Frontend (two pages)
│   ├── app.js           # WebSocket client + canvas rendering
│   └── style.css        # Dark theme styles
├── py-requirements.txt  # Python dependencies
├── Dockerfile           # Isolated container definition
├── docker-compose.yml   # Container orchestration
├── enter-claude.sh      # Launch Claude Code in the container
└── README.md
```
