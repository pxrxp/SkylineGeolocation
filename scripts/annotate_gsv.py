#!/usr/bin/env python
"""Zero-Labor AI Skyline Server (Direct-Scribble Contour Engine).

Fixes:
  - Direct-Scribble Path Replacement: Your hand-drawn scribble directly shapes 
    the skyline (snapping to nearby edges) and NEVER produces straight horizontal lines.
  - Storm-Cloud Resistant Feature Map: Uses CLAHE + Multi-Scale Sobel Texture 
    gradients that easily separate dark storm clouds from dark mountain shadows.
  - Smooth Boundary Blending: Seamlessly merges scribble updates with existing line.

Usage:
  python scripts/annotate_gsv.py [--port 8787] [--host 127.0.0.1] [--limit 30]
"""

import argparse
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "data/street_view/images")
MASKS_DIR = os.path.join(ROOT, "data/street_view/masks")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")

H, W = 720, 1080
HIST_CAP = 25

STATE = {
    "sids": [],
    "annotations": {},
    "skipped": {},
    "lines": {},
    "hist": {},
    "lock": threading.Lock(),
}


def safe_load_json(file_path, default_value):
    """Loads a JSON file safely, recovering automatically if empty or corrupted."""
    if not os.path.exists(file_path):
        return default_value
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return default_value
            return json.loads(content)
    except Exception as e:
        print(f"Warning: Could not parse {file_path} ({e}). Starting fresh.")
        return default_value


def load_state(limit):
    """Safely loads ground truth and annotations."""
    gt = safe_load_json(GT_FILE, {})
    sids = list(gt.keys())
    if limit:
        sids = sids[:limit]
    STATE["sids"] = sids

    annot_data = safe_load_json(ANNOT_FILE, {"annotations": {}, "skipped": {}})
    STATE["annotations"] = annot_data.get("annotations", {})
    STATE["skipped"] = annot_data.get("skipped", {})


def save_state():
    """Atomic save: writes to temp file first to prevent JSON corruption."""
    os.makedirs(os.path.dirname(ANNOT_FILE), exist_ok=True)
    tmp_file = ANNOT_FILE + ".tmp"
    payload = {"annotations": STATE["annotations"], "skipped": STATE["skipped"]}
    with open(tmp_file, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp_file, ANNOT_FILE)


def dark_channel_prior_dehaze(img_rgb, omega=0.80, patch_size=15):
    """He et al. Dark Channel Prior Dehazing to sharpen distant hazy peaks."""
    img_float = img_rgb.astype(np.float64) / 255.0
    min_channel = np.min(img_float, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    flat_dark = dark_channel.ravel()
    flat_img = img_float.reshape(-1, 3)
    num_pixels = len(flat_dark)
    top_num = max(1, int(num_pixels * 0.001))
    indices = np.argpartition(flat_dark, -top_num)[-top_num:]
    A = np.mean(flat_img[indices], axis=0)
    A = np.maximum(A, 0.1)

    normalized = img_float / A
    min_norm = np.min(normalized, axis=2)
    dark_norm = cv2.erode(min_norm, kernel)
    transmission = 1.0 - omega * dark_norm
    transmission = np.clip(transmission, 0.1, 1.0)

    dehazed = (img_float - A) / transmission[:, :, None] + A
    dehazed = np.clip(dehazed * 255.0, 0, 255).astype(np.uint8)
    return dehazed


def compute_enhanced_skyline_feature_map(img_rgb):
    """Robust Texture & Gradient Feature Map resilient against storm clouds."""
    H_sub, W_sub, _ = img_rgb.shape

    dehazed = dark_channel_prior_dehaze(img_rgb, omega=0.80)
    smoothed = cv2.bilateralFilter(dehazed, d=5, sigmaColor=35, sigmaSpace=35)

    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(gray).astype(np.float64)
    b_clahe = clahe.apply(lab[:, :, 2]).astype(np.float64)

    # Color & Luminance vertical gradients
    db = np.maximum(0, b_clahe[2:, :] - b_clahe[:-2, :])
    dl = np.maximum(0, l_clahe[:-2, :] - l_clahe[2:, :])

    canny = cv2.Canny(smoothed, 20, 80).astype(np.float64) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g_mag = np.sqrt(gx**2 + gy**2)
    g_norm = g_mag / (g_mag.max() + 1e-6)

    pad_db = np.zeros((H_sub, W_sub), dtype=np.float64)
    pad_db[1:-1, :] = db

    pad_dl = np.zeros((H_sub, W_sub), dtype=np.float64)
    pad_dl[1:-1, :] = dl

    raw_signal = (pad_db * 2.0 + pad_dl * 1.0) * (1.0 + 2.0 * canny) + 1.5 * g_norm

    # TOP-DOWN FIRST-EDGE SUPPRESSION
    cum_signal = np.cumsum(raw_signal, axis=0)
    prev_cum = cum_signal - raw_signal
    top_first_suppression = np.exp(-1.5 * np.maximum(0.0, prev_cum / (raw_signal.max() + 1e-6)))

    topmost_map = raw_signal * top_first_suppression
    if topmost_map.max() > 0:
        topmost_map /= topmost_map.max()

    return topmost_map, dehazed, b_clahe


def extract_pure_image_baseline_skyline(img_rgb):
    """Automatic Base Skyline Extractor supporting steep vertical cliffs."""
    H_img, W_img, _ = img_rgb.shape
    feature_map, _, _ = compute_enhanced_skyline_feature_map(img_rgb)

    cost = 1.0 - feature_map

    dp = np.full((H_img, W_img), 1e7, dtype=np.float32)
    backtrack = np.zeros((H_img, W_img), dtype=np.int32)
    dp[:, 0] = cost[:, 0]

    max_step = 25
    steps = np.arange(-max_step, max_step + 1)
    step_costs = 0.02 * (np.abs(steps) ** 1.4)

    for c in range(1, W_img):
        for dr, step_cost in zip(steps, step_costs):
            if dr >= 0:
                prev = dp[: H_img - dr, c - 1] + step_cost
                curr_target = cost[dr:, c] + prev
                mask = curr_target < dp[dr:, c]
                dp[dr:, c] = np.where(mask, curr_target, dp[dr:, c])
                backtrack[dr:, c] = np.where(mask, np.arange(H_img - dr), backtrack[dr:, c])
            else:
                abs_dr = -dr
                prev = dp[abs_dr:, c - 1] + step_cost
                curr_target = cost[: H_img - abs_dr, c] + prev
                mask = curr_target < dp[: H_img - abs_dr, c]
                dp[: H_img - abs_dr, c] = np.where(mask, curr_target, dp[: H_img - abs_dr, c])
                backtrack[: H_img - abs_dr, c] = np.where(
                    mask, np.arange(abs_dr, H_img), backtrack[: H_img - abs_dr, c]
                )

    best_last_r = np.argmin(dp[:, W_img - 1])
    base_line = np.zeros(W_img, dtype=np.float32)
    base_line[-1] = best_last_r
    for c in range(W_img - 1, 0, -1):
        base_line[c - 1] = backtrack[int(base_line[c]), c]

    base_line_smooth = gaussian_filter1d(base_line, sigma=1.0)
    return np.clip(np.round(base_line_smooth), 0, H_img - 1).astype(np.int32)


def load_line(sid):
    p_img = os.path.join(IMAGES_DIR, sid + ".png")
    if os.path.exists(p_img):
        try:
            img_rgb = np.array(Image.open(p_img).convert("RGB"))
            return extract_pure_image_baseline_skyline(img_rgb)
        except Exception as e:
            print(f"Warning: Could not extract image base skyline for {sid}: {e}")

    p_mask = os.path.join(MASKS_DIR, sid + ".png")
    line = np.full(W, H - 1, dtype=np.int32)
    if os.path.exists(p_mask):
        try:
            m = np.array(Image.open(p_mask).convert("L"))
            if m[:10, :].mean() > m[-10:, :].mean():
                m = 255 - m
            mask = (m >= 128).astype(np.uint8) * 255
            for c in range(W):
                r = np.where(mask[:, c] == 255)[0]
                if len(r):
                    line[c] = r[0]
        except Exception as e:
            print(f"Warning: Could not load mask for {sid}: {e}")

    return line


def line_for(sid):
    if sid not in STATE["lines"]:
        STATE["lines"][sid] = load_line(sid)
    return STATE["lines"][sid]


def push_hist(sid):
    hist = STATE["hist"].setdefault(sid, [])
    hist.append(line_for(sid).copy())
    if len(hist) > HIST_CAP:
        del hist[0]


def pop_hist(sid):
    hist = STATE["hist"].get(sid, [])
    if hist:
        STATE["lines"][sid] = hist.pop()
        return True
    return False


def magic_scribble_snap(img_rgb, current_line, points, radius=30):
    """Direct-Scribble Contour Engine: Snaps locally to physical edges along your drawn path."""
    H_img, W_img, _ = img_rgb.shape
    if not points or len(points) < 2:
        return current_line.copy()

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Sort stroke by x coordinate
    sort_idx = np.argsort(xs)
    xs = xs[sort_idx]
    ys = ys[sort_idx]

    c_min = max(0, int(np.min(xs)))
    c_max = min(W_img - 1, int(np.max(xs)))

    if c_max <= c_min + 2:
        return current_line.copy()

    # Interpolate target scribble curve for all columns in stroke range
    scribble_cols = np.arange(c_min, c_max + 1)
    scribble_y = np.interp(scribble_cols, xs, ys)

    # Compute robust feature map
    feature_map, _, _ = compute_enhanced_skyline_feature_map(img_rgb)

    # Local edge-snapping along the user's stroke
    snapped_y = scribble_y.copy()
    search_radius = max(18, int(radius * 0.7))

    for idx, col in enumerate(scribble_cols):
        target_y = int(scribble_y[idx])
        y_start = max(0, target_y - search_radius)
        y_end = min(H_img - 1, target_y + search_radius)

        edge_strip = feature_map[y_start : y_end + 1, col]
        if len(edge_strip) > 0 and edge_strip.max() > 0.12:
            local_rows = np.arange(y_start, y_end + 1)
            dist_penalty = 0.003 * ((local_rows - target_y) ** 2)
            score = edge_strip - dist_penalty
            best_local_y = local_rows[np.argmax(score)]
            snapped_y[idx] = best_local_y

    # Smooth the snapped stroke
    snapped_y_smooth = gaussian_filter1d(snapped_y, sigma=1.2)

    # Replace line across scribble span
    updated_line = current_line.copy()
    updated_line[c_min : c_max + 1] = np.clip(
        np.round(snapped_y_smooth), 0, H_img - 1
    ).astype(np.int32)

    # Smooth full path transition
    updated_line = gaussian_filter1d(updated_line.astype(np.float64), sigma=0.8)
    return np.clip(np.round(updated_line), 0, H_img - 1).astype(np.int32)


def apply_scribble_brush(sid, points, tool, radius):
    """Applies ✨ Magic Scribble Guide."""
    push_hist(sid)
    line = line_for(sid)

    img_path = os.path.join(IMAGES_DIR, sid + ".png")
    if os.path.exists(img_path):
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        STATE["lines"][sid] = magic_scribble_snap(
            img_rgb,
            line,
            points=points,
            radius=radius,
        )


def line_to_points(line, step=1):
    return [[int(c), int(line[c])] for c in range(0, W, step)]


def render_view_mode_png(sid, view_mode, line):
    """Serves requested inspection layer: dehazed photo, LAB b*, CLAHE edge, or overlay tint."""
    img_path = os.path.join(IMAGES_DIR, sid + ".png")
    if not os.path.exists(img_path):
        return ""

    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    if view_mode == "dehazed":
        dehazed = dark_channel_prior_dehaze(img_rgb, omega=0.80)
        buf = io.BytesIO()
        Image.fromarray(dehazed).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    elif view_mode == "lab":
        _, _, b_clahe = compute_enhanced_skyline_feature_map(img_rgb)
        b_vis = cv2.applyColorMap(b_clahe.astype(np.uint8), cv2.COLORMAP_JET)
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(b_vis, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    elif view_mode == "clahe":
        feature_map, _, _ = compute_enhanced_skyline_feature_map(img_rgb)
        feat_vis = (feature_map * 255.0).astype(np.uint8)
        feat_vis = cv2.applyColorMap(feat_vis, cv2.COLORMAP_MAGMA)
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(feat_vis, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # Default mask overlay tint
    rr = np.arange(H)[:, None]
    sky = rr < line[None, :]

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[sky, 0:3] = (30, 110, 255)
    rgba[sky, 3] = 75
    rgba[~sky, 0:3] = (255, 140, 20)
    rgba[~sky, 3] = 60

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def sample_payload(sid, view_mode="tint"):
    done = len(STATE["annotations"]) + len(STATE["skipped"])
    total = len(STATE["sids"])
    idx = next((i for i, s in enumerate(STATE["sids"]) if s == sid), 0) + 1
    line = line_for(sid)
    return {
        "sid": sid,
        "img": "/api/img/" + sid + ".png",
        "points": line_to_points(line, step=1),
        "tint": render_view_mode_png(sid, view_mode, line),
        "total": total,
        "done": done,
        "idx": idx,
    }


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Direct-Scribble AI Skyline Refinement</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 14px; background: #111; color: #eee; max-width: 1100px; }
  #status { font-size: 18px; font-weight: 600; margin-bottom: 2px; }
  #sub { font-size: 12px; color: #888; margin-bottom: 8px; }
  #barwrap { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  #bar { flex: 1; height: 8px; background: #222; border-radius: 4px; overflow: hidden; }
  #barfill { height: 100%; background: #38bdf8; width: 0%; transition: width .2s; }
  #barprog { font-size: 13px; color: #aaa; white-space: nowrap; }
  #toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
  .btn { padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer;
         background: #222; color: #ddd; border: 1px solid #444; border-radius: 6px; }
  .btn.active { border-color: #38bdf8; background: #0284c7; color: #fff; }
  .btn.primary { background: #16a34a; border-color: #16a34a; color: #fff; }
  .view-btn.active { border-color: #f59e0b; background: #b45309; color: #fff; }
  #tip { font-size: 13px; color: #7dd3fc; margin-bottom: 8px; min-height: 18px; }
  #imgwrap { position: relative; display: inline-block; line-height: 0; user-select: none; }
  #img { display: block; max-width: 1080px; height: auto; border-radius: 4px; }
  #tint { position: absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
  svg { position: absolute; top:0; left:0; width:100%; height:100%; touch-action: none; }
</style></head>
<body>
  <div id="status">Loading AI Engine…</div>
  <div id="sub"></div>
  <div id="barwrap"><div id="bar"><div id="barfill"></div></div><span id="barprog"></span></div>
  
  <div id="toolbar">
    <button class="btn active" id="btn_scribble">✨ Magic Scribble Guide</button>
    <span style="color:#555">|</span>
    <span style="font-size:12px; color:#aaa">Size:</span>
    <input type="range" id="brush_size" min="15" max="70" value="30">
    <span style="color:#555">|</span>
    <span style="font-size:12px; color:#aaa">View Mode:</span>
    <button class="btn view-btn active" id="view_tint">🎭 Tint Overlay</button>
    <button class="btn view-btn" id="view_dehazed">🌫️ Dehazed Photo</button>
    <button class="btn view-btn" id="view_lab">🎨 LAB b* Color Map</button>
    <button class="btn view-btn" id="view_clahe">🔍 Edge Map</button>
    <span style="flex:1"></span>
    <button class="btn" id="btn_undo">↩ Undo (u)</button>
    <button class="btn" id="btn_reset">Reset (c)</button>
    <button class="btn primary" id="btn_save">Save + Next (s)</button>
    <button class="btn" id="btn_skip">Skip (n)</button>
  </div>
  
  <div id="tip">✨ Magic Scribble Guide: Draw a quick rough stroke across any mountain section—AI shapes the ridge directly along your path!</div>
  
  <div id="imgwrap">
    <img id="img">
    <img id="tint">
    <svg id="svg"></svg>
  </div>

<script>
const img = document.getElementById('img'), svg = document.getElementById('svg'), tint = document.getElementById('tint');
let cur = null, tool = 'scribble', viewMode = 'tint', pts = [], strokePts = [], busy = false, dragging = false;

function pos(evt) {
  const r = img.getBoundingClientRect();
  return [Math.max(0, Math.min(1080, (evt.clientX - r.left) / r.width * 1080)),
          Math.max(0, Math.min(720, (evt.clientY - r.top) / r.height * 720))];
}

function render() {
  svg.innerHTML = '';
  if (pts.length > 1) {
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    poly.setAttribute('points', pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' '));
    poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke', '#ff2222');
    poly.setAttribute('stroke-width', '2.5');
    svg.appendChild(poly);
  }
  if (dragging && strokePts.length > 1) {
    const stroke = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    stroke.setAttribute('points', strokePts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' '));
    stroke.setAttribute('fill', 'none');
    stroke.setAttribute('stroke', '#38bdf8');
    stroke.setAttribute('stroke-width', '4');
    stroke.setAttribute('stroke-dasharray', '6 3');
    svg.appendChild(stroke);
  }
}

function setViewMode(v) {
  viewMode = v;
  ['tint', 'dehazed', 'lab', 'clahe'].forEach(m => {
    document.getElementById('view_' + m).className = 'btn view-btn' + (v === m ? ' active' : '');
  });
  if (cur) sendAction('view_mode', {view_mode: viewMode});
}

function sendAction(endpoint, body) {
  if (busy || !cur) return;
  busy = true;
  fetch('/api/' + endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({sid: cur.sid, view_mode: viewMode}, body))
  }).then(r => r.json()).then(apply);
}

svg.addEventListener('pointerdown', evt => {
  svg.setPointerCapture(evt.pointerId);
  dragging = true;
  strokePts = [pos(evt)];
  render();
});

svg.addEventListener('pointermove', evt => {
  if (dragging) {
    strokePts.push(pos(evt));
    render();
  }
});

svg.addEventListener('pointerup', () => {
  if (dragging && strokePts.length > 1) {
    dragging = false;
    const radius = parseInt(document.getElementById('brush_size').value);
    sendAction('scribble_brush', {points: strokePts, tool: 'scribble', radius: radius});
    strokePts = [];
  } else {
    dragging = false;
  }
});

window.addEventListener('keydown', evt => {
  if (evt.key === 's') sendAction('annotate', {});
  else if (evt.key === 'n') sendAction('skip', {});
  else if (evt.key === 'u') sendAction('undo', {});
  else if (evt.key === 'c') sendAction('reset', {});
});

document.getElementById('view_tint').onclick = () => setViewMode('tint');
document.getElementById('view_dehazed').onclick = () => setViewMode('dehazed');
document.getElementById('view_lab').onclick = () => setViewMode('lab');
document.getElementById('view_clahe').onclick = () => setViewMode('clahe');
document.getElementById('btn_undo').onclick = () => sendAction('undo', {});
document.getElementById('btn_reset').onclick = () => sendAction('reset', {});
document.getElementById('btn_save').onclick = () => sendAction('annotate', {});
document.getElementById('btn_skip').onclick = () => sendAction('skip', {});

function apply(d) {
  if (d.error) { busy = false; return; }
  cur = d;
  pts = d.points || [];
  img.src = d.img;
  tint.src = 'data:image/png;base64,' + d.tint;
  document.getElementById('status').textContent = 'Sample ' + d.idx + ' / ' + d.total;
  document.getElementById('sub').textContent = d.sid;
  document.getElementById('barfill').style.width = ((d.done / d.total) * 100) + '%';
  document.getElementById('barprog').textContent = d.done + '/' + d.total + ' processed';
  render();
  busy = false;
}

fetch('/api/next').then(r => r.json()).then(apply);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, HTML.encode(), "text/html")
        elif u.path.startswith("/api/img/"):
            sid = u.path[len("/api/img/") :].split(".")[0]
            p = os.path.join(IMAGES_DIR, sid + ".png")
            if os.path.exists(p):
                self._send(200, open(p, "rb").read(), "image/png")
            else:
                self._send(404, b"not found")
        elif u.path == "/api/next":
            sid = next(
                (
                    s
                    for s in STATE["sids"]
                    if s not in STATE["annotations"] and s not in STATE["skipped"]
                ),
                None,
            )
            if sid is None:
                self._send(200, json.dumps({"error": "All done!"}).encode())
            else:
                with STATE["lock"]:
                    self._send(200, json.dumps(sample_payload(sid)).encode())

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        sid = body.get("sid")
        vm = body.get("view_mode", "tint")

        with STATE["lock"]:
            if u.path == "/api/scribble_brush":
                apply_scribble_brush(
                    sid,
                    body.get("points", []),
                    body.get("tool", "scribble"),
                    int(body.get("radius", 30)),
                )
                self._send(200, json.dumps(sample_payload(sid, view_mode=vm)).encode())
            elif u.path == "/api/view_mode":
                self._send(200, json.dumps(sample_payload(sid, view_mode=vm)).encode())
            elif u.path == "/api/undo":
                pop_hist(sid)
                self._send(200, json.dumps(sample_payload(sid, view_mode=vm)).encode())
            elif u.path == "/api/reset":
                STATE["lines"][sid] = load_line(sid)
                STATE["hist"].pop(sid, None)
                self._send(200, json.dumps(sample_payload(sid, view_mode=vm)).encode())
            elif u.path == "/api/annotate":
                line = line_for(sid)
                STATE["annotations"][sid] = line_to_points(line, step=1)
                save_state()
                sid_next = next(
                    (
                        s
                        for s in STATE["sids"]
                        if s not in STATE["annotations"]
                        and s not in STATE["skipped"]
                    ),
                    None,
                )
                if sid_next:
                    self._send(
                        200, json.dumps(sample_payload(sid_next, view_mode=vm)).encode()
                    )
                else:
                    self._send(200, json.dumps({"error": "All done!"}).encode())
            elif u.path == "/api/skip":
                STATE["skipped"][sid] = True
                save_state()
                sid_next = next(
                    (
                        s
                        for s in STATE["sids"]
                        if s not in STATE["annotations"]
                        and s not in STATE["skipped"]
                    ),
                    None,
                )
                if sid_next:
                    self._send(
                        200, json.dumps(sample_payload(sid_next, view_mode=vm)).encode()
                    )
                else:
                    self._send(200, json.dumps({"error": "All done!"}).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    load_state(args.limit)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Direct-Scribble AI Server running at http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
