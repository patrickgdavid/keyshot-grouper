#!/usr/bin/env python3
"""
keyshot_grouper.py — Group VFX shot thumbnails by camera angle.

Feature backends (best → fallback):
  dinov2  — DINOv2 ViT-S/14 + UMAP + HDBSCAN (automatic k, outlier detection)
  resnet  — ResNet18 + k-means
  hog     — HOG + k-means
  numpy   — spatial colour histograms + k-means

Usage:
    python keyshot_grouper.py --dir /path/to/thumbnails [--k 8] [--port 6003]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from flask import Flask, jsonify, render_template, request, send_from_directory

THUMB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

app = Flask(__name__)
STATE = {}


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_dinov2(image_paths):
    import torch
    from torchvision import transforms

    print("  Loading DINOv2 ViT-S/14 (downloads ~84 MB on first run)...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = []
    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            t = transform(img).unsqueeze(0)
            feat = model(t).squeeze().numpy()
            features.append(feat)
            sys.stdout.write(".")
            sys.stdout.flush()
    print()
    return np.array(features)


def _extract_resnet(image_paths):
    import torch
    import torchvision.models as models
    from torchvision import transforms

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = []
    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            t = transform(img).unsqueeze(0)
            feat = model(t).squeeze().numpy()
            features.append(feat)
            sys.stdout.write(".")
            sys.stdout.flush()
    print()
    return np.array(features)


def _extract_hog(image_paths):
    from skimage.feature import hog
    from skimage.transform import resize as sk_resize
    import skimage.color

    features = []
    for p in image_paths:
        img = np.array(Image.open(p).convert("RGB"))
        img = sk_resize(img, (128, 128), anti_aliasing=True)
        gray = skimage.color.rgb2gray(img)
        h = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), feature_vector=True)
        features.append(h)
        sys.stdout.write(".")
        sys.stdout.flush()
    print()
    return np.array(features)


def _extract_numpy(image_paths):
    """Fallback: spatial colour histograms on a 4×4 grid."""
    features = []
    for p in image_paths:
        img = np.array(Image.open(p).convert("RGB").resize((64, 64)))
        feat = []
        h, w = img.shape[:2]
        for i in range(4):
            for j in range(4):
                patch = img[h*i//4:h*(i+1)//4, w*j//4:w*(j+1)//4]
                for c in range(3):
                    hist, _ = np.histogram(patch[:, :, c], bins=16, range=(0, 256))
                    feat.extend(hist / (hist.sum() + 1e-8))
        features.append(feat)
        sys.stdout.write(".")
        sys.stdout.flush()
    print()
    return np.array(features)


def resolve_method(method_arg):
    """Return (feature_method, cluster_method) for the given --method arg."""
    if method_arg == "auto":
        try:
            import torch, torchvision  # noqa: F401
            return "dinov2", "hdbscan"
        except ImportError:
            pass
        try:
            import skimage  # noqa: F401
            return "hog", "kmeans"
        except ImportError:
            pass
        return "numpy", "kmeans"
    elif method_arg == "dinov2":
        return "dinov2", "hdbscan"
    else:
        return method_arg, "kmeans"


def extract_features(image_paths, feature_method):
    print(f"Feature extraction: {feature_method}", flush=True)
    raw = {
        "dinov2": _extract_dinov2,
        "resnet":  _extract_resnet,
        "hog":     _extract_hog,
        "numpy":   _extract_numpy,
    }[feature_method](image_paths)
    return normalize(raw)


# ── Clustering ─────────────────────────────────────────────────────────────────

def _shots_for_indices(indices, image_paths):
    return [
        {"name": Path(image_paths[i]).stem, "file": Path(image_paths[i]).name}
        for i in sorted(indices)
    ]


def _centroid_keyshot(indices, features_or_embedding, centroid):
    dists = [np.linalg.norm(features_or_embedding[i] - centroid) for i in indices]
    return indices[int(np.argmin(dists))]


def cluster_kmeans(features, image_paths, k):
    k = min(k, len(image_paths))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(features)
    groups = []
    for g in range(k):
        indices = [i for i, lbl in enumerate(labels) if lbl == g]
        if not indices:
            continue
        key_i = _centroid_keyshot(indices, features, km.cluster_centers_[g])
        groups.append({
            "id": f"group_{g}",
            "label": f"Group {g + 1}",
            "keyshot": Path(image_paths[key_i]).stem,
            "shots": _shots_for_indices(indices, image_paths),
        })
    groups.sort(key=lambda g: len(g["shots"]), reverse=True)
    return groups


def cluster_hdbscan(features, image_paths, min_cluster_size=2):
    try:
        from umap import UMAP
    except ImportError:
        sys.exit("umap-learn not installed: pip install umap-learn")
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        sys.exit("hdbscan not installed: pip install hdbscan")

    n = len(features)
    n_components = max(2, min(20, n - 2))
    n_neighbors  = max(2, min(15, n - 1))

    print(f"  UMAP: {features.shape[1]}d → {n_components}d "
          f"(n_neighbors={n_neighbors})...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric="cosine",
            random_state=42,
        ).fit_transform(features)

    print(f"  HDBSCAN (min_cluster_size={min_cluster_size})...", flush=True)
    labels = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
    ).fit_predict(embedding)

    unique = sorted(set(labels) - {-1})
    groups = []
    for g in unique:
        indices = [i for i, lbl in enumerate(labels) if lbl == g]
        centroid = embedding[indices].mean(axis=0)
        key_i = _centroid_keyshot(indices, embedding, centroid)
        groups.append({
            "id": f"group_{g}",
            "label": f"Group {g + 1}",
            "keyshot": Path(image_paths[key_i]).stem,
            "shots": _shots_for_indices(indices, image_paths),
        })
    groups.sort(key=lambda g: len(g["shots"]), reverse=True)

    outlier_idx = [i for i, lbl in enumerate(labels) if lbl == -1]
    if outlier_idx:
        shots = _shots_for_indices(outlier_idx, image_paths)
        groups.append({
            "id": "group_outliers",
            "label": f"Ungrouped",
            "keyshot": None,
            "shots": shots,
        })

    n_clusters = len(unique)
    n_outliers = len(outlier_idx)
    print(f"  → {n_clusters} groups, {n_outliers} ungrouped shots")
    return groups


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    return send_from_directory(STATE["thumb_dir"], filename)


@app.route("/api/info")
def get_info():
    return jsonify({
        "cluster_method":  STATE["cluster_method"],
        "feature_method":  STATE["feature_method"],
        "cluster_param":   STATE["cluster_param"],
    })


@app.route("/api/groups")
def get_groups():
    return jsonify(STATE["groups"])


@app.route("/api/groups", methods=["POST"])
def update_groups():
    STATE["groups"] = request.json
    return jsonify({"ok": True})


@app.route("/api/recluster", methods=["POST"])
def recluster():
    param = request.args.get("k", type=int)
    if not param or param < 1:
        return jsonify({"error": "invalid param"}), 400

    STATE["cluster_param"] = param
    if STATE["cluster_method"] == "hdbscan":
        print(f"Re-clustering (HDBSCAN min_cluster_size={param})...", flush=True)
        groups = cluster_hdbscan(STATE["features"], STATE["image_paths"],
                                 min_cluster_size=max(2, param))
    else:
        print(f"Re-clustering (k-means k={param})...", flush=True)
        groups = cluster_kmeans(STATE["features"], STATE["image_paths"], k=param)

    STATE["groups"] = groups
    return jsonify(groups)


@app.route("/api/reextract", methods=["POST"])
def reextract():
    method = request.args.get("method", "auto")
    feature_method, cluster_method = resolve_method(method)

    print(f"Re-extracting features: {feature_method}...", flush=True)
    features = extract_features(STATE["image_paths"], feature_method)

    param = STATE["cluster_param"]
    if cluster_method == "hdbscan":
        groups = cluster_hdbscan(features, STATE["image_paths"], min_cluster_size=param)
    else:
        groups = cluster_kmeans(features, STATE["image_paths"], k=param)

    STATE.update({
        "features":       features,
        "feature_method": feature_method,
        "cluster_method": cluster_method,
        "groups":         groups,
    })
    return jsonify({
        "groups":         groups,
        "feature_method": feature_method,
        "cluster_method": cluster_method,
        "cluster_param":  param,
    })


@app.route("/api/export", methods=["POST"])
def export_groups():
    out_path = STATE["output"]
    export_data = {
        "groups": [
            {
                "keyshot": g["keyshot"],
                "label":   g["label"],
                "shots":   [s["name"] for s in g["shots"]],
            }
            for g in STATE["groups"]
            if g["id"] != "group_outliers" or g["shots"]  # always include non-empty outliers
        ]
    }
    with open(out_path, "w") as f:
        json.dump(export_data, f, indent=2)
    return jsonify({"ok": True, "path": out_path})


# ── Entry point ────────────────────────────────────────────────────────────────

def run(args):
    thumb_dir = Path(args.dir).resolve()
    image_paths = sorted([
        str(p) for p in thumb_dir.iterdir()
        if p.suffix.lower() in THUMB_EXTENSIONS
    ])
    if not image_paths:
        sys.exit(f"No images found in {thumb_dir}")

    feature_method, cluster_method = resolve_method(args.method)

    print(f"Found {len(image_paths)} thumbnails.")
    features = extract_features(image_paths, feature_method)

    if cluster_method == "hdbscan":
        min_cs = max(2, args.k) if args.k else 2
        print(f"Clustering: HDBSCAN (min_cluster_size={min_cs})...")
        groups = cluster_hdbscan(features, image_paths, min_cluster_size=min_cs)
        cluster_param = min_cs
    else:
        k = args.k if args.k else max(2, min(10, len(image_paths) // 3))
        print(f"Clustering: k-means (k={k})...")
        groups = cluster_kmeans(features, image_paths, k)
        cluster_param = k

    STATE.update({
        "thumb_dir":      str(thumb_dir),
        "image_paths":    image_paths,
        "features":       features,
        "groups":         groups,
        "feature_method": feature_method,
        "cluster_method": cluster_method,
        "cluster_param":  cluster_param,
        "output":         str(Path(args.output).resolve()),
    })

    print(f"\nOpen: http://localhost:{args.port}")
    print(f"Output: {STATE['output']}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group shot thumbnails by camera angle")
    parser.add_argument("--dir",    required=True)
    parser.add_argument("--k",      type=int, default=None,
                        help="Clusters for k-means, or min_cluster_size for HDBSCAN")
    parser.add_argument("--port",   type=int, default=6003)
    parser.add_argument("--output", default="keyshot_groups.json")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "dinov2", "resnet", "hog", "numpy"])
    run(parser.parse_args())
