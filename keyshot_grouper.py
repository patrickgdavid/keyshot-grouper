#!/usr/bin/env python3
"""
keyshot_grouper.py — Group VFX shot thumbnails by camera angle.

Modes:
  --dir /path   Local folder of images (original workflow)
  (no --dir)    ShotGrid mode — pick a project from the UI, pulls live thumbnails

Feature backends (best → fallback):
  dinov2  DINOv2 ViT-S/14 + UMAP + HDBSCAN (automatic k, outlier detection)
  resnet  ResNet18 + k-means
  hog     HOG + k-means
  numpy   spatial colour histograms + k-means
"""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from flask import Flask, jsonify, render_template, request, send_from_directory

THUMB_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
CACHE_DIR = Path.home() / ".cache" / "keyshot-grouper"
INACTIVE_STATUSES = ["omit", "hdn", "void", "na", "bid", "archive"]

app = Flask(__name__)
STATE = {}


def _set_progress(message, current=0, total=0):
    STATE["progress"] = {"message": message, "current": current, "total": total}


# ── ShotGrid connection ────────────────────────────────────────────────────────

_RDO_PATHS = [
    "/System/Volumes/Data/rdo/software/rez/packages/rdo_shotgun_core/1.13.0/python",
    "/System/Volumes/Data/rdo/software/rez/packages/rdo_logging/1.7.2/python",
    "/System/Volumes/Data/rdo/software/rez/packages/rdo_site/0.5.1/python",
]


def _load_env():
    script_dir = Path(__file__).parent
    for candidate in [script_dir / ".env", script_dir.parent / "sg_note_drafter" / ".env"]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def sg_connect():
    _load_env()
    login = os.environ.get("SG_USER_LOGIN")
    if not login:
        sys.exit("SG_USER_LOGIN not set in .env")

    for p in _RDO_PATHS:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import rdo_shotgun_core
        return rdo_shotgun_core.connect(scriptName="Delivery", cache=False, username=login)
    except Exception:
        pass

    import shotgun_api3
    url    = os.environ.get("SG_URL")
    script = os.environ.get("SG_SCRIPT_NAME")
    key    = os.environ.get("SG_API_KEY")
    cred_file = Path.home() / ".sg_timelog.json"
    if cred_file.exists():
        d = json.loads(cred_file.read_text())
        url    = url    or d.get("sg_url")
        script = script or d.get("sg_script_name")
        key    = key    or d.get("sg_api_key")
    if not all([url, script, key]):
        sys.exit("Missing ShotGrid credentials — set SG_URL, SG_SCRIPT_NAME, SG_API_KEY in .env")
    return shotgun_api3.Shotgun(url, script_name=script, api_key=key, sudo_as_login=login)


def fetch_active_projects(sg):
    projects = sg.find(
        "Project",
        [["sg_status", "is", "Active"]],
        ["id", "name", "code"],
        order=[{"field_name": "name", "direction": "asc"}],
    )
    return [{"id": p["id"], "name": p["name"], "code": p.get("code") or p["name"]}
            for p in projects]


def fetch_project_shots(sg, project_id):
    return sg.find(
        "Shot",
        [
            ["project", "is", {"type": "Project", "id": project_id}],
            ["sg_status_list", "not_in", INACTIVE_STATUSES],
        ],
        ["code", "sg_sequence", "image"],
        order=[
            {"field_name": "sg_sequence.Sequence.code", "direction": "asc"},
            {"field_name": "code", "direction": "asc"},
        ],
    )


def _download_one(shot, project_code):
    url = shot.get("image")
    seq = shot.get("sg_sequence")
    seq_name = seq.get("name", "No Sequence") if isinstance(seq, dict) else "No Sequence"
    if not url:
        return shot["code"], seq_name, None
    cache_dir = CACHE_DIR / project_code
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{shot['code']}.jpg"
    if not path.exists():
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(path))
        except Exception as e:
            print(f"  Warning: {shot['code']}: {e}", flush=True)
            return shot["code"], seq_name, None
    return shot["code"], seq_name, path


def download_thumbnails(project_code, shots):
    """Download thumbnails in parallel; return sequences dict."""
    total = len(shots)
    completed = []
    sequences = {}
    _set_progress("Downloading thumbnails", 0, total)

    def _track(future):
        result = future.result()
        completed.append(1)
        _set_progress("Downloading thumbnails", len(completed), total)
        return result

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(_download_one, shot, project_code) for shot in shots]
        results = [_track(f) for f in as_completed(futures)]

    for shot_code, seq_name, path in sorted(results, key=lambda r: r[0]):
        if seq_name not in sequences:
            sequences[seq_name] = {"shots": [], "features": None, "groups": None, "cluster_param": None}
        if path:
            sequences[seq_name]["shots"].append({
                "name": shot_code,
                "file": f"{shot_code}.jpg",
                "file_path": str(path),
            })

    return sequences


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
    n = len(image_paths)
    with torch.no_grad():
        for i, p in enumerate(image_paths):
            t = transform(Image.open(p).convert("RGB")).unsqueeze(0)
            features.append(model(t).squeeze().numpy())
            _set_progress("Extracting features · dinov2", i + 1, n)
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
    n = len(image_paths)
    with torch.no_grad():
        for i, p in enumerate(image_paths):
            t = transform(Image.open(p).convert("RGB")).unsqueeze(0)
            features.append(model(t).squeeze().numpy())
            _set_progress("Extracting features · resnet", i + 1, n)
    print()
    return np.array(features)


def _extract_hog(image_paths):
    from skimage.feature import hog
    from skimage.transform import resize as sk_resize
    import skimage.color
    features = []
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        img = sk_resize(np.array(Image.open(p).convert("RGB")), (128, 128), anti_aliasing=True)
        h = hog(skimage.color.rgb2gray(img), orientations=9,
                pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        features.append(h)
        _set_progress("Extracting features · hog", i + 1, n)
    print()
    return np.array(features)


def _extract_numpy(image_paths):
    features = []
    n = len(image_paths)
    for idx, p in enumerate(image_paths):
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
        _set_progress("Extracting features · numpy", idx + 1, n)
    print()
    return np.array(features)


def resolve_method(method_arg):
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
    print(f"  Feature extraction: {feature_method} ({len(image_paths)} shots)...", flush=True)
    raw = {"dinov2": _extract_dinov2, "resnet": _extract_resnet,
           "hog": _extract_hog, "numpy": _extract_numpy}[feature_method](image_paths)
    return normalize(raw)


# ── Clustering ─────────────────────────────────────────────────────────────────

def _shots_for_indices(indices, image_paths):
    return [{"name": Path(image_paths[i]).stem, "file": Path(image_paths[i]).name}
            for i in sorted(indices)]


def _centroid_keyshot(indices, embedding, centroid):
    dists = [np.linalg.norm(embedding[i] - centroid) for i in indices]
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
        groups.append({"id": f"group_{g}", "label": f"Group {g+1}",
                       "keyshot": Path(image_paths[key_i]).stem,
                       "shots": _shots_for_indices(indices, image_paths)})
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
    # UMAP needs at least n_neighbors+1 samples; fall back to k-means for tiny sequences
    if n < 6:
        return cluster_kmeans(features, image_paths, k=min(2, n))
    n_components = max(2, min(20, n - 2))
    n_neighbors  = max(2, min(15, n - 1))
    print(f"  UMAP: {features.shape[1]}d → {n_components}d...", flush=True)
    _set_progress(f"Reducing dimensions · UMAP ({features.shape[1]}d → {n_components}d)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                         metric="cosine", random_state=42).fit_transform(features)
    print(f"  HDBSCAN (min_cluster_size={min_cluster_size})...", flush=True)
    _set_progress(f"Finding clusters · HDBSCAN (min_size={min_cluster_size})")
    labels = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=1, metric="euclidean").fit_predict(embedding)
    unique = sorted(set(labels) - {-1})
    groups = []
    for g in unique:
        indices = [i for i, lbl in enumerate(labels) if lbl == g]
        centroid = embedding[indices].mean(axis=0)
        key_i = _centroid_keyshot(indices, embedding, centroid)
        groups.append({"id": f"group_{g}", "label": f"Group {g+1}",
                       "keyshot": Path(image_paths[key_i]).stem,
                       "shots": _shots_for_indices(indices, image_paths)})
    groups.sort(key=lambda g: len(g["shots"]), reverse=True)
    outlier_idx = [i for i, lbl in enumerate(labels) if lbl == -1]
    if outlier_idx:
        groups.append({"id": "group_outliers", "label": "Ungrouped",
                       "keyshot": None,
                       "shots": _shots_for_indices(outlier_idx, image_paths)})
    print(f"  → {len(unique)} groups, {len(outlier_idx)} ungrouped", flush=True)
    return groups


def _do_cluster(features, image_paths):
    if STATE["cluster_method"] == "hdbscan":
        return cluster_hdbscan(features, image_paths, min_cluster_size=STATE["cluster_param"])
    return cluster_kmeans(features, image_paths, k=STATE["cluster_param"])


# ── State helpers ──────────────────────────────────────────────────────────────

def _active_seq():
    return STATE.get("active_sequence") if STATE.get("sg_mode") else None


def _get_groups():
    seq = _active_seq()
    if seq:
        return STATE["sequences"].get(seq, {}).get("groups") or []
    return STATE.get("groups", [])


def _set_groups(groups):
    seq = _active_seq()
    if seq:
        STATE["sequences"][seq]["groups"] = groups
    else:
        STATE["groups"] = groups


def _get_features():
    seq = _active_seq()
    if seq:
        return STATE["sequences"].get(seq, {}).get("features")
    return STATE.get("features")


def _get_image_paths():
    seq = _active_seq()
    if seq:
        return [s["file_path"] for s in STATE["sequences"].get(seq, {}).get("shots", [])
                if s.get("file_path")]
    return STATE.get("image_paths", [])


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    if STATE.get("sg_mode") and STATE.get("project"):
        return send_from_directory(str(CACHE_DIR / STATE["project"]["code"]), filename)
    return send_from_directory(STATE["thumb_dir"], filename)


@app.route("/api/progress")
def get_progress():
    return jsonify(STATE.get("progress", {"message": "", "current": 0, "total": 0}))


@app.route("/api/info")
def get_info():
    return jsonify({
        "sg_mode":        STATE.get("sg_mode", False),
        "cluster_method": STATE.get("cluster_method", "kmeans"),
        "feature_method": STATE.get("feature_method", "hog"),
        "cluster_param":  STATE.get("cluster_param", 2),
    })


# ── ShotGrid endpoints ─────────────────────────────────────────────────────────

@app.route("/api/projects")
def get_projects():
    if not STATE.get("sg_mode"):
        return jsonify([])
    try:
        return jsonify(fetch_active_projects(STATE["sg"]))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load_project", methods=["POST"])
def load_project():
    if not STATE.get("sg_mode"):
        return jsonify({"error": "not in SG mode"}), 400
    body = request.json
    project_id   = body["id"]
    project_code = body["code"]
    project_name = body["name"]
    print(f"\nLoading project: {project_name} ({project_code})", flush=True)
    _set_progress(f"Fetching shots from ShotGrid…")
    shots = fetch_project_shots(STATE["sg"], project_id)
    print(f"  {len(shots)} active shots found.", flush=True)
    sequences = download_thumbnails(project_code, shots)
    STATE["project"]          = {"id": project_id, "code": project_code, "name": project_name}
    STATE["sequences"]        = sequences
    STATE["active_sequence"]  = sorted(sequences.keys())[0] if sequences else None
    return jsonify({
        "sequences": [
            {"name": name, "shot_count": len(data["shots"]), "clustered": False}
            for name, data in sorted(sequences.items())
        ]
    })


@app.route("/api/sequences")
def get_sequences():
    sequences = STATE.get("sequences", {})
    return jsonify([
        {"name": name, "shot_count": len(d["shots"]), "clustered": d["groups"] is not None}
        for name, d in sorted(sequences.items())
    ])


@app.route("/api/load_sequence", methods=["POST"])
def load_sequence():
    seq_name = request.json.get("sequence")
    sequences = STATE.get("sequences", {})
    if not seq_name or seq_name not in sequences:
        return jsonify({"error": "unknown sequence"}), 400
    STATE["active_sequence"] = seq_name
    seq = sequences[seq_name]
    if seq["groups"] is None:
        paths = [s["file_path"] for s in seq["shots"] if s.get("file_path")]
        if not paths:
            seq["groups"] = []
        else:
            try:
                print(f"\nClustering {seq_name} ({len(paths)} shots)...", flush=True)
                seq["features"] = extract_features(paths, STATE["feature_method"])
                seq["groups"]   = _do_cluster(seq["features"], paths)
            except Exception as exc:
                print(f"Clustering error for {seq_name}: {exc}", flush=True)
                seq["groups"] = []
    seq_param = seq.get("cluster_param") or STATE["cluster_param"]
    return jsonify({
        "groups":         seq["groups"],
        "cluster_method": STATE["cluster_method"],
        "cluster_param":  seq_param,
    })


# ── Grouping endpoints (work in both modes) ────────────────────────────────────

@app.route("/api/groups")
def get_groups():
    return jsonify(_get_groups())


@app.route("/api/groups", methods=["POST"])
def update_groups():
    _set_groups(request.json)
    return jsonify({"ok": True})


@app.route("/api/recluster", methods=["POST"])
def recluster():
    param = request.args.get("k", type=int)
    if not param or param < 1:
        return jsonify({"error": "invalid param"}), 400
    STATE["cluster_param"] = param
    # Also persist k per-sequence so switching tabs preserves it
    seq_name = _active_seq()
    if seq_name and seq_name in STATE.get("sequences", {}):
        STATE["sequences"][seq_name]["cluster_param"] = param
    features    = _get_features()
    image_paths = _get_image_paths()
    if features is None:
        return jsonify({"error": "features not yet extracted"}), 400
    groups = _do_cluster(features, image_paths)
    _set_groups(groups)
    return jsonify(groups)


@app.route("/api/reextract", methods=["POST"])
def reextract():
    method = request.args.get("method", "auto")
    feature_method, cluster_method = resolve_method(method)
    STATE["feature_method"] = feature_method
    STATE["cluster_method"] = cluster_method

    if STATE.get("sg_mode"):
        # Invalidate other sequences; re-extract active one now
        for name, seq in STATE.get("sequences", {}).items():
            if name != STATE.get("active_sequence"):
                seq["features"] = None
                seq["groups"]   = None
        seq_name = STATE.get("active_sequence")
        if seq_name:
            seq   = STATE["sequences"][seq_name]
            paths = [s["file_path"] for s in seq["shots"] if s.get("file_path")]
            seq["features"] = extract_features(paths, feature_method)
            seq["groups"]   = _do_cluster(seq["features"], paths)
    else:
        STATE["features"] = extract_features(STATE["image_paths"], feature_method)
        STATE["groups"]   = _do_cluster(STATE["features"], STATE["image_paths"])

    return jsonify({
        "groups":         _get_groups(),
        "feature_method": feature_method,
        "cluster_method": cluster_method,
        "cluster_param":  STATE["cluster_param"],
    })


@app.route("/api/export", methods=["POST"])
def export_groups():
    out_path = STATE["output"]
    if STATE.get("sg_mode"):
        project = STATE.get("project", {})
        data = {
            "project": project,
            "sequences": {
                seq_name: {
                    "groups": [
                        {"keyshot": g["keyshot"], "label": g["label"],
                         "shots": [s["name"] for s in g["shots"]]}
                        for g in seq_data["groups"]
                    ]
                }
                for seq_name, seq_data in sorted(STATE.get("sequences", {}).items())
                if seq_data["groups"] is not None
            },
        }
    else:
        data = {
            "groups": [
                {"keyshot": g["keyshot"], "label": g["label"],
                 "shots": [s["name"] for s in g["shots"]]}
                for g in STATE.get("groups", [])
            ]
        }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"ok": True, "path": out_path})


# ── Entry point ────────────────────────────────────────────────────────────────

def run(args):
    feature_method, cluster_method = resolve_method(args.method)
    cluster_param = args.k if args.k else 2

    if args.dir:
        thumb_dir   = Path(args.dir).resolve()
        image_paths = sorted([str(p) for p in thumb_dir.iterdir()
                               if p.suffix.lower() in THUMB_EXTENSIONS])
        if not image_paths:
            sys.exit(f"No images found in {thumb_dir}")
        print(f"Found {len(image_paths)} thumbnails.")
        features = extract_features(image_paths, feature_method)
        if cluster_method == "hdbscan":
            cluster_param = args.k or 2
        else:
            cluster_param = args.k or max(2, min(10, len(image_paths) // 3))
        groups = _do_cluster_with(features, image_paths, cluster_method, cluster_param)
        STATE.update({
            "sg_mode":        False,
            "thumb_dir":      str(thumb_dir),
            "image_paths":    image_paths,
            "features":       features,
            "groups":         groups,
            "feature_method": feature_method,
            "cluster_method": cluster_method,
            "cluster_param":  cluster_param,
            "output":         str(Path(args.output).resolve()),
        })
    else:
        print("ShotGrid mode. Connecting...", flush=True)
        sg = sg_connect()
        print("Connected.", flush=True)
        STATE.update({
            "sg_mode":        True,
            "sg":             sg,
            "project":        None,
            "sequences":      {},
            "active_sequence": None,
            "feature_method": feature_method,
            "cluster_method": cluster_method,
            "cluster_param":  cluster_param,
            "output":         str(Path(args.output).resolve()),
        })

    print(f"\nOpen: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


def _do_cluster_with(features, image_paths, cluster_method, cluster_param):
    if cluster_method == "hdbscan":
        return cluster_hdbscan(features, image_paths, min_cluster_size=cluster_param)
    return cluster_kmeans(features, image_paths, k=cluster_param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group shot thumbnails by camera angle")
    parser.add_argument("--dir",    default=None, help="Local thumbnail directory (omit for ShotGrid mode)")
    parser.add_argument("--k",      type=int, default=None)
    parser.add_argument("--port",   type=int, default=6003)
    parser.add_argument("--output", default="keyshot_groups.json")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "dinov2", "resnet", "hog", "numpy"])
    run(parser.parse_args())
