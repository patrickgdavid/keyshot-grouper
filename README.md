# Keyshot Grouper

Groups VFX shot thumbnails by camera angle and organises them into keyshots with child shots. Useful for bidding, ShotGrid setup, and client-facing angle breakdowns.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)

## How it works

1. Each thumbnail is passed through a computer vision model to extract a feature vector representing its visual structure
2. Those features are clustered — shots with similar angles land in the same group
3. A web interface opens for review: drag shots between groups, rename groups, promote any shot to keyshot
4. Export to JSON when done

The default backend is **DINOv2 + UMAP + HDBSCAN**, which finds natural groupings without requiring you to specify a number of clusters. Shots that don't clearly belong anywhere are flagged as **Ungrouped**. Falls back to HOG or colour histograms if PyTorch isn't installed.

## Installation

```bash
git clone https://github.com/patrickgdavid/keyshot-grouper
cd keyshot-grouper

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Optional but recommended — enables DINOv2 (best accuracy):
.venv/bin/pip install torch torchvision
```

## Usage

```bash
./keyshot_grouper.sh --dir /path/to/thumbnails
```

Then open **http://localhost:6003**.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | *(required)* | Directory of thumbnail images |
| `--k` | auto | Number of clusters (k-means) or min cluster size (HDBSCAN) |
| `--output` | `keyshot_groups.json` | Output JSON path |
| `--port` | `6003` | Web UI port |
| `--method` | `auto` | Feature backend: `dinov2`, `resnet`, `hog`, `numpy` |

### Supported image formats

`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.webp`

## Web UI

- **Method dropdown** — switch feature backend; re-extracts and re-clusters automatically
- **min_size / k field** — adjust clustering granularity; board updates 700ms after you stop typing
- **Drag and drop** — move shots between groups
- **★ key** (hover a thumbnail) — promote that shot to keyshot for its group
- **Ungrouped card** — shots HDBSCAN couldn't confidently assign; drag them into groups manually
- **Export JSON** — saves the final grouping

## Output format

```json
{
  "groups": [
    {
      "keyshot": "SEQ_0010",
      "label": "Wide exterior",
      "shots": ["SEQ_0010", "SEQ_0020", "SEQ_0030"]
    }
  ]
}
```

## Feature backends

| Method | Model | Clustering | Notes |
|--------|-------|------------|-------|
| `dinov2` | DINOv2 ViT-S/14 | UMAP + HDBSCAN | Best accuracy. Requires PyTorch (~84 MB download on first run) |
| `resnet` | ResNet18 | k-means | Good accuracy. Requires PyTorch |
| `hog` | HOG | k-means | No PyTorch needed. Requires scikit-image |
| `numpy` | Colour histograms | k-means | No extra dependencies |

`auto` selects the best available backend.
