#!/usr/bin/env python
"""Adaptive GSV Crop Dashboard & Storage-Aware Auto-Cropper.

Launch GUI:   python scripts/gsv_crop_dashboard.py [--port 8765]
Batch Mode:   python scripts/gsv_crop_dashboard.py --auto-detect-all --auto-crop-all [--target-crops 3]
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import webbrowser
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.streetview_utils import slice_perspective

DEFAULT_PORT = 8765
OUT_W, OUT_H = 1080, 720
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"


def get_pano_list(panos_dir):
    panos_dir = Path(panos_dir)
    panos = []
    for p in sorted(list(panos_dir.glob("*.jpg")) + list(panos_dir.glob("*.png"))):
        panos.append({"id": p.stem, "path": str(p)})
    return panos


def is_valid_sky_crop(img, min_sky_frac=0.15, min_relief=2.0):
    """Check that crop has actual sky at top and is not a ground/bush close-up."""
    arr = np.array(img)
    top_hsv = cv2.cvtColor(arr[: int(OUT_H * 0.25), :, :], cv2.COLOR_RGB2HSV)
    sky_pixel_frac = float(((top_hsv[:, :, 2] > 120) & (top_hsv[:, :, 1] < 55)).mean())

    top_row = arr[:5, :, :].mean(axis=(0, 2))
    has_relief = np.std(top_row) >= min_relief

    return (sky_pixel_frac >= min_sky_frac) and has_relief


def _sky_mask(img_region):
    """Return a boolean mask of pixels that look like actual sky.

    Uses both HSV and RGB heuristics to separate real sky (blue / bright
    overcast) from white walls, concrete, snow, etc.
    """
    arr = np.array(img_region)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)

    # Blue sky: hue in blue-cyan range (80-130 in OpenCV 0-180), decent brightness
    blue_sky = (h_ch >= 80) & (h_ch <= 130) & (v_ch > 100) & (s_ch > 15)

    # Bright overcast / light sky: high brightness, low saturation, blue-ish tint
    # The blue channel must be >= green to avoid warm-toned walls
    overcast = (v_ch > 140) & (s_ch < 40) & (b >= g) & (b >= r * 0.9)

    return blue_sky | overcast


def _perspective_crop_quality(img):
    """Evaluate whether a *rendered perspective crop* is good for skyline matching.

    Very strict — returns True only for crops that clearly contain
    sky AND a visible skyline (sky-to-ground transition).
    When in doubt, returns False (mark as bad).

    The top and bottom ~15%% of the perspective crop are excluded from
    evaluation because they sample from near zenith/nadir in the
    equirectangular projection and are distorted or cut.
    """
    arr = np.array(img)
    H, W, _ = arr.shape

    # ── Exclude distorted top/bottom bands ──
    band = max(1, int(H * 0.15))
    usable = arr[band : H - band, :, :]  # middle ~70%% of crop
    uh, uw, _ = usable.shape

    # Divide the usable region into thirds: sky zone (top), horizon zone (mid), ground zone (bot)
    sky_zone = usable[: uh // 3, :, :]           # upper third of usable
    horizon_zone = usable[uh // 3 : 2 * uh // 3, :, :]  # middle third
    ground_zone = usable[2 * uh // 3 :, :, :]   # lower third

    sky_mask = _sky_mask(sky_zone)
    sky_frac = float(sky_mask.mean())

    # ── 1) Must have real sky in the upper usable region ──
    if sky_frac < 0.30:
        return False

    # ── 2) Sky must dominate the upper usable — not just scattered pixels ──
    row_sky_frac = sky_mask.astype(float).mean(axis=1)
    rows_with_sky = float((row_sky_frac > 0.20).mean())
    if rows_with_sky < 0.50:
        return False

    # ── 3) Upper zone must be bright — dark = ground/building/wall ──
    sky_hsv = cv2.cvtColor(sky_zone, cv2.COLOR_RGB2HSV)
    mean_v = float(sky_hsv[:, :, 2].mean())
    if mean_v < 50:
        return False

    # ── 4) Must NOT be all cloud/fog — sky needs actual blue ──
    blue_only = float(((
        (sky_hsv[:, :, 0] >= 80) & (sky_hsv[:, :, 0] <= 130) &
        (sky_hsv[:, :, 2] > 100) & (sky_hsv[:, :, 1] > 15)
    ).mean()))
    if sky_frac > 0.05:
        cloud_ratio = 1.0 - (blue_only / sky_frac)
        if cloud_ratio > 0.75:
            return False

    # ── 5) Must have a skyline — edges in the horizon zone ──
    hor_gray = cv2.cvtColor(horizon_zone, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(hor_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edge_frac = float((edges > 0).mean())
    if edge_frac < 0.02:
        return False

    # ── 6) Sky-to-ground transition — sky must drop from sky_zone to ground_zone ──
    ground_sky = _sky_mask(ground_zone)
    ground_sky_frac = float(ground_sky.mean())
    if ground_sky_frac > sky_frac * 0.70:
        return False

    # ── 7) Minimum contrast in usable region — flat/uniform = bad ──
    usable_gray = cv2.cvtColor(usable, cv2.COLOR_RGB2GRAY)
    if float(usable_gray.std()) < 25:
        return False

    return True


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GSV Crop Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; height: 100vh; }
#app { display: flex; height: 100vh; }
#sidebar { width: 280px; background: #16213e; overflow-y: auto; flex-shrink: 0; border-right: 1px solid #333; }
#pano-list { padding: 4px; }
.pano-item { padding: 6px 10px; cursor: pointer; border-radius: 4px; font-size: 12px; font-family: monospace; display: flex; justify-content: space-between; }
.pano-item:hover { background: #1a1a4e; }
.pano-item.active { background: #0f3460; color: #e94560; }
#viewer { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#top-bar { display: flex; align-items: center; padding: 8px 16px; background: #16213e; border-bottom: 1px solid #333; gap: 8px; font-size: 13px; flex-wrap: wrap; }
#top-bar button { background: #0f3460; border: 1px solid #333; color: #eee; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 12px; }
#top-bar button.save { background: #e94560; border-color: #e94560; }
#top-bar button.active-mode { background: #e94560; border-color: #e94560; }
#top-bar .sep { width: 1px; height: 24px; background: #444; }
#top-bar .info { color: #888; }
.badge { background: #e94560; color: #fff; border-radius: 8px; padding: 1px 6px; font-size: 10px; margin-left: 4px; }
#canvas-row { flex: 1; display: flex; overflow: hidden; }
#pano-container { flex: 1; position: relative; overflow: hidden; cursor: grab; }
#pano-canvas { position: absolute; top: 0; left: 0; }
#preview-panel { width: 360px; background: #16213e; border-left: 1px solid #333; display: flex; flex-direction: column; padding: 12px; }
#preview-canvas { width: 100%; border: 1px solid #333; border-radius: 4px; background: #000; }
#toast { position: fixed; bottom: 20px; right: 20px; background: #0f3460; color: #eee; padding: 10px 20px; border-radius: 6px; font-size: 13px; display: none; z-index: 100; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <input type="text" id="search" placeholder="Search panoramas..." oninput="filterPanos()">
    <div id="pano-list"></div>
  </div>
  <div id="viewer">
    <div id="top-bar">
      <span id="pano-id" class="info">Select a panorama</span>
      <button onclick="resetView()">Reset View</button>
      <div class="sep"></div>
      <button id="btn-mark-bad" onclick="toggleMarkBadMode()">Mark Bad Region</button>
      <button id="btn-del-region" onclick="deleteSelectedRegion()" style="display:none">Delete Region</button>
      <div class="sep"></div>
      <button class="auto" onclick="autoCrop()">Auto Crop</button>
      <button class="save" onclick="saveCrop()">Save Crop</button>
    </div>
    <div id="canvas-row">
      <div id="pano-container">
        <canvas id="pano-canvas"></canvas>
      </div>
      <div id="preview-panel">
        <h3>PERSPECTIVE PREVIEW</h3>
        <span id="preview-info" style="font-size:11px;color:#888;margin:4px 0"></span>
        <canvas id="preview-canvas" width="360" height="240"></canvas>
        <div id="forbidden-info" style="margin-top:8px;font-size:11px;color:#888"></div>
      </div>
    </div>
  </div>
</div>
<div id="toast"></div>

<script>
let panos = []; let currentPano = null; let savedCrops = {};
let heading = 0, pitch = 0, fovY = 65, dragging = false, dragStartX = 0, dragStartHeading = 0;
let panoImg = null;

// ── Forbidden regions state ──
let forbiddenRegions = [];  // [{x1,y1,x2,y2,source}, ...]  normalized 0-1
let markBadMode = false;
let selectedRegionIdx = -1;

// Region interaction state
let regionDragMode = null;  // 'move' | 'resize-tl' | 'resize-tr' | 'resize-bl' | 'resize-br' | null
let regionDragStart = null; // {mx, my, origRegion}

const panoCanvas = document.getElementById('pano-canvas');
const panoCtx = panoCanvas.getContext('2d');
const container = document.getElementById('pano-container');

async function init() {
    const resp = await fetch('/api/panos'); panos = await resp.json();
    const stateResp = await fetch('/api/state'); savedCrops = await stateResp.json();
    renderPanoList(panos);
    if (panos.length > 0) selectPano(0);
}

function renderPanoList(list) {
    const el = document.getElementById('pano-list');
    el.innerHTML = list.map(p => `
        <div class="pano-item ${currentPano && currentPano.id === p.id ? 'active' : ''}" onclick="selectPanoById('${p.id}')">
            <span>${p.id.substring(0, 20)}</span>
        </div>
    `).join('');
}

function selectPano(idx) { selectPanoById(panos[idx].id); }

function selectPanoById(id) {
    currentPano = panos.find(p => p.id === id);
    if (!currentPano) return;
    document.getElementById('pano-id').textContent = currentPano.id;
    selectedRegionIdx = -1;
    document.getElementById('btn-del-region').style.display = 'none';

    // Fetch forbidden regions for this pano
    fetchForbiddenRegions(currentPano.id).then(() => {
        panoImg = new Image();
        panoImg.onload = () => {
            resizePanoCanvas();
            // Start at the best (first non-forbidden) heading
            heading = getBestHeading();
            drawPano();
        };
        panoImg.src = '/api/pano/' + currentPano.id + '.jpg';
    });
    renderPanoList(panos);
}

async function fetchForbiddenRegions(panoId) {
    try {
        const resp = await fetch('/api/forbidden/' + panoId);
        const data = await resp.json();
        forbiddenRegions = data.regions || [];
    } catch(e) {
        forbiddenRegions = [];
    }
}

async function saveForbiddenRegions() {
    if (!currentPano) return;
    await fetch('/api/forbidden/' + currentPano.id, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ regions: forbiddenRegions })
    });
    updateForbiddenInfo();
}

function getBestHeading() {
    // Find the first heading (in 15-deg steps) that's NOT in a forbidden region
    for (let h = 0; h < 360; h += 15) {
        if (!headingInForbidden(h)) return h;
    }
    return 0; // all forbidden, just use 0
}

function headingInForbidden(hDeg) {
    const hNorm = ((hDeg % 360) + 360) % 360 / 360.0;
    for (const r of forbiddenRegions) {
        const xMin = Math.min(r.x1, r.x2);
        const xMax = Math.max(r.x1, r.x2);
        if (hNorm >= xMin && hNorm <= xMax) return true;
    }
    return false;
}

function resizePanoCanvas() {
    const rect = container.getBoundingClientRect();
    panoCanvas.width = rect.width * devicePixelRatio;
    panoCanvas.height = (rect.width / 2) * devicePixelRatio;
    panoCanvas.style.width = rect.width + 'px';
    panoCanvas.style.height = (rect.width / 2) + 'px';
    panoCtx.scale(devicePixelRatio, devicePixelRatio);
}

let previewTimer = null;

function drawPano() {
    if (!panoImg) return;
    const w = panoCanvas.width / devicePixelRatio, h = panoCanvas.height / devicePixelRatio;
    panoCtx.clearRect(0, 0, w, h);
    panoCtx.drawImage(panoImg, 0, 0, w, h);

    // Draw forbidden regions as semi-transparent red overlays
    for (let i = 0; i < forbiddenRegions.length; i++) {
        const r = forbiddenRegions[i];
        const xMin = Math.min(r.x1, r.x2) * w;
        const xMax = Math.max(r.x1, r.x2) * w;
        const yMin = Math.min(r.y1, r.y2) * h;
        const yMax = Math.max(r.y1, r.y2) * h;
        const rw = xMax - xMin;
        const rh = yMax - yMin;

        // Fill
        panoCtx.fillStyle = (i === selectedRegionIdx) ? 'rgba(233,69,96,0.35)' : 'rgba(233,69,96,0.22)';
        panoCtx.fillRect(xMin, yMin, rw, rh);

        // Border
        panoCtx.strokeStyle = (i === selectedRegionIdx) ? '#e94560' : 'rgba(233,69,96,0.6)';
        panoCtx.lineWidth = (i === selectedRegionIdx) ? 2 : 1;
        panoCtx.setLineDash(i === selectedRegionIdx ? [] : [4, 4]);
        panoCtx.strokeRect(xMin, yMin, rw, rh);
        panoCtx.setLineDash([]);

        // Draw corner handles if selected
        if (i === selectedRegionIdx) {
            const hs = 6;
            panoCtx.fillStyle = '#e94560';
            // TL
            panoCtx.fillRect(xMin - hs/2, yMin - hs/2, hs, hs);
            // TR
            panoCtx.fillRect(xMax - hs/2, yMin - hs/2, hs, hs);
            // BL
            panoCtx.fillRect(xMin - hs/2, yMax - hs/2, hs, hs);
            // BR
            panoCtx.fillRect(xMax - hs/2, yMax - hs/2, hs, hs);

            // Source label
            panoCtx.fillStyle = '#fff';
            panoCtx.font = '10px sans-serif';
            panoCtx.fillText(r.source || 'manual', xMin + 3, yMin + 12);
        }
    }

    // Draw heading indicator
    panoCtx.strokeStyle = headingInForbidden(Math.round(heading)) ? '#ff8800' : '#e94560';
    panoCtx.lineWidth = headingInForbidden(Math.round(heading)) ? 3 : 2;
    const cx = w * (((heading % 360 + 360) % 360) / 360);
    panoCtx.beginPath(); panoCtx.moveTo(cx, 0); panoCtx.lineTo(cx, h); panoCtx.stroke();

    // If heading is in forbidden, show warning near the line
    if (headingInForbidden(Math.round(heading))) {
        panoCtx.fillStyle = '#ff8800';
        panoCtx.font = 'bold 12px sans-serif';
        panoCtx.fillText('⚠ BAD AREA', cx + 6, 16);
    }

    // Draw new region being created (if in mark-bad mode and dragging)
    if (markBadMode && regionDragMode === 'draw' && regionDragStart) {
        const r = regionDragStart;
        const xMin = Math.min(r.x1, r.x2) * w;
        const xMax = Math.max(r.x1, r.x2) * w;
        const yMin = Math.min(r.y1, r.y2) * h;
        const yMax = Math.max(r.y1, r.y2) * h;
        panoCtx.fillStyle = 'rgba(233,69,96,0.3)';
        panoCtx.fillRect(xMin, yMin, xMax - xMin, yMax - yMin);
        panoCtx.strokeStyle = '#e94560';
        panoCtx.lineWidth = 2;
        panoCtx.setLineDash([6, 3]);
        panoCtx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
        panoCtx.setLineDash([]);
    }

    schedulePreview();
}

function schedulePreview() {
    if (previewTimer) clearTimeout(previewTimer);
    previewTimer = setTimeout(fetchPreview, 150);
}

function fetchPreview() {
    if (!currentPano) return;
    const url = `/api/preview?id=${currentPano.id}&heading=${heading}&pitch=${pitch}&fov=${fovY}`;
    const img = new Image();
    img.onload = () => {
        const canvas = document.getElementById('preview-canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        document.getElementById('preview-info').textContent =
            `heading=${Math.round(heading)}° pitch=${Math.round(pitch)}° fov=${Math.round(fovY)}°`;
    };
    img.src = url;
}

function updateForbiddenInfo() {
    const el = document.getElementById('forbidden-info');
    const n = forbiddenRegions.length;
    if (n === 0) {
        el.textContent = 'No bad regions marked';
    } else {
        el.innerHTML = `<span style="color:#e94560">●</span> ${n} bad region${n > 1 ? 's' : ''} marked (drag to move/resize)`;
    }
}

function filterPanos() {
    const q = document.getElementById('search').value.toLowerCase();
    renderPanoList(panos.filter(p => p.id.toLowerCase().includes(q)));
}

function toggleMarkBadMode() {
    markBadMode = !markBadMode;
    const btn = document.getElementById('btn-mark-bad');
    if (markBadMode) {
        btn.classList.add('active-mode');
        btn.textContent = '✓ Drawing... (ESC to cancel)';
        panoCanvas.style.cursor = 'crosshair';
    } else {
        btn.classList.remove('active-mode');
        btn.textContent = 'Mark Bad Region';
        panoCanvas.style.cursor = 'grab';
    }
    regionDragMode = null;
    regionDragStart = null;
    drawPano();
}

function deleteSelectedRegion() {
    if (selectedRegionIdx < 0 || selectedRegionIdx >= forbiddenRegions.length) return;
    forbiddenRegions.splice(selectedRegionIdx, 1);
    selectedRegionIdx = -1;
    document.getElementById('btn-del-region').style.display = 'none';
    saveForbiddenRegions();
    drawPano();
}

async function saveCrop() {
    if (!currentPano) return;
    // Block save if heading is in forbidden region
    if (headingInForbidden(Math.round(heading))) {
        showToast('⚠ Cannot save: heading is in a bad area! Move to a clear area first.');
        return;
    }
    const resp = await fetch('/api/crop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ pano_id: currentPano.id, heading_deg: heading, pitch_deg: pitch, fov_y_deg: fovY })
    });
    const result = await resp.json();
    if (result.ok) {
        savedCrops[currentPano.id] = (savedCrops[currentPano.id] || 0) + 1;
        showToast('Saved crop');
        renderPanoList(panos);
    } else if (result.error) {
        showToast('⚠ ' + result.error);
    }
}

async function autoCrop() {
    if (!currentPano) return;
    showToast('Auto-cropping adaptive...');
    const resp = await fetch('/api/auto-crop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ pano_id: currentPano.id, target_crops: 3 })
    });
    const result = await resp.json();
    if (result.ok) {
        showToast(`Saved ${result.total} crops`);
        savedCrops[currentPano.id] = result.total;
        renderPanoList(panos);
    }
}

function showToast(msg) {
    const el = document.getElementById('toast');
    el.textContent = msg; el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 2500);
}

function resetView() {
    heading = getBestHeading();
    pitch = 0; fovY = 65;
    drawPano();
}

// ── Region interaction helpers ──
function canvasCoords(e) {
    const rect = panoCanvas.getBoundingClientRect();
    const w = rect.width, h = rect.height;
    return {
        nx: (e.clientX - rect.left) / w,  // normalized 0-1
        ny: (e.clientY - rect.top) / h,
        px: e.clientX - rect.left,
        py: e.clientY - rect.top,
        w, h
    };
}

function hitTestRegion(nx, ny) {
    // Returns index of region under cursor, or -1
    for (let i = forbiddenRegions.length - 1; i >= 0; i--) {
        const r = forbiddenRegions[i];
        const xMin = Math.min(r.x1, r.x2), xMax = Math.max(r.x1, r.x2);
        const yMin = Math.min(r.y1, r.y2), yMax = Math.max(r.y1, r.y2);
        if (nx >= xMin && nx <= xMax && ny >= yMin && ny <= yMax) return i;
    }
    return -1;
}

function hitTestHandle(nx, ny) {
    // Returns {idx, handle} if near a corner handle of selected region
    if (selectedRegionIdx < 0) return null;
    const r = forbiddenRegions[selectedRegionIdx];
    const xMin = Math.min(r.x1, r.x2), xMax = Math.max(r.x1, r.x2);
    const yMin = Math.min(r.y1, r.y2), yMax = Math.max(r.y1, r.y2);
    const threshold = 0.015; // ~1.5% of canvas

    const corners = [
        { handle: 'tl', x: xMin, y: yMin },
        { handle: 'tr', x: xMax, y: yMin },
        { handle: 'bl', x: xMin, y: yMax },
        { handle: 'br', x: xMax, y: yMax },
    ];
    for (const c of corners) {
        if (Math.abs(nx - c.x) < threshold && Math.abs(ny - c.y) < threshold) {
            return { idx: selectedRegionIdx, handle: c.handle };
        }
    }
    return null;
}

// ── Mouse event handlers ──
let tiltDrag = false, tiltStartY = 0, tiltStartPitch = 0;

panoCanvas.addEventListener('contextmenu', e => e.preventDefault());

panoCanvas.addEventListener('mousedown', e => {
    if (e.button === 2) {
        // Right-click tilt
        tiltDrag = true; tiltStartY = e.clientY; tiltStartPitch = pitch; e.preventDefault();
        return;
    }

    const { nx, ny } = canvasCoords(e);

    if (markBadMode) {
        // Start drawing a new bad region
        regionDragMode = 'draw';
        regionDragStart = { x1: nx, y1: ny, x2: nx, y2: ny };
        return;
    }

    // Check handle hit first
    const handleHit = hitTestHandle(nx, ny);
    if (handleHit) {
        regionDragMode = 'resize-' + handleHit.handle;
        regionDragStart = { mx: nx, my: ny, origRegion: { ...forbiddenRegions[handleHit.idx] } };
        return;
    }

    // Check region body hit (for moving)
    const regionIdx = hitTestRegion(nx, ny);
    if (regionIdx >= 0) {
        selectedRegionIdx = regionIdx;
        document.getElementById('btn-del-region').style.display = '';
        regionDragMode = 'move';
        regionDragStart = { mx: nx, my: ny, origRegion: { ...forbiddenRegions[regionIdx] } };
        drawPano();
        return;
    }

    // Deselect region
    if (selectedRegionIdx >= 0) {
        selectedRegionIdx = -1;
        document.getElementById('btn-del-region').style.display = 'none';
        drawPano();
    }

    // Pan the panorama
    dragging = true; dragStartX = e.clientX; dragStartHeading = heading;
    panoCanvas.style.cursor = 'grabbing';
});

document.addEventListener('mousemove', e => {
    if (dragging) {
        const dx = e.clientX - dragStartX;
        heading = dragStartHeading - dx * 0.3;
        drawPano();
        return;
    }

    if (tiltDrag) {
        pitch = Math.max(-60, Math.min(60, tiltStartPitch + (e.clientY - tiltStartY) * 0.3));
        drawPano();
        return;
    }

    if (regionDragMode && regionDragStart) {
        const { nx, ny } = canvasCoords(e);
        const r = forbiddenRegions[selectedRegionIdx];
        const orig = regionDragStart.origRegion;

        if (regionDragMode === 'draw') {
            regionDragStart.x2 = nx;
            regionDragStart.y2 = ny;
            drawPano();
            return;
        }

        const dx = nx - regionDragStart.mx;
        const dy = ny - regionDragStart.my;

        if (regionDragMode === 'move') {
            const rw = Math.abs(orig.x2 - orig.x1);
            const rh = Math.abs(orig.y2 - orig.y1);
            const newX1 = Math.max(0, Math.min(1, orig.x1 + dx));
            const newY1 = Math.max(0, Math.min(1, orig.y1 + dy));
            r.x1 = newX1; r.y1 = newY1;
            r.x2 = newX1 + (orig.x2 > orig.x1 ? rw : -rw);
            r.y2 = newY1 + (orig.y2 > orig.y1 ? rh : -rh);
        } else if (regionDragMode === 'resize-tl') {
            r.x1 = Math.max(0, Math.min(1, orig.x1 + dx));
            r.y1 = Math.max(0, Math.min(1, orig.y1 + dy));
        } else if (regionDragMode === 'resize-tr') {
            r.x2 = Math.max(0, Math.min(1, orig.x2 + dx));
            r.y1 = Math.max(0, Math.min(1, orig.y1 + dy));
        } else if (regionDragMode === 'resize-bl') {
            r.x1 = Math.max(0, Math.min(1, orig.x1 + dx));
            r.y2 = Math.max(0, Math.min(1, orig.y2 + dy));
        } else if (regionDragMode === 'resize-br') {
            r.x2 = Math.max(0, Math.min(1, orig.x2 + dx));
            r.y2 = Math.max(0, Math.min(1, orig.y2 + dy));
        }
        drawPano();
    }
});

document.addEventListener('mouseup', e => {
    if (e.button === 2) { tiltDrag = false; return; }

    if (regionDragMode === 'draw' && regionDragStart) {
        const { x1, y1, x2, y2 } = regionDragStart;
        const rw = Math.abs(x2 - x1);
        const rh = Math.abs(y2 - y1);
        if (rw > 0.01 && rh > 0.01) {
            // Normalize so x1 < x2, y1 < y2
            forbiddenRegions.push({
                x1: Math.min(x1, x2), y1: Math.min(y1, y2),
                x2: Math.max(x1, x2), y2: Math.max(y1, y2),
                source: 'manual'
            });
            saveForbiddenRegions();
            showToast('Bad region added');
        }
        regionDragMode = null;
        regionDragStart = null;
        drawPano();
        return;
    }

    if (regionDragMode && regionDragMode !== 'draw') {
        // Finished moving/resizing — save to server
        saveForbiddenRegions();
    }

    if (dragging) {
        dragging = false;
        panoCanvas.style.cursor = markBadMode ? 'crosshair' : 'grab';
    }
    regionDragMode = null;
    regionDragStart = null;
});

// ESC to cancel mark-bad mode
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        if (markBadMode) toggleMarkBadMode();
        if (regionDragMode === 'draw') {
            regionDragMode = null;
            regionDragStart = null;
            drawPano();
        }
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedRegionIdx >= 0 && document.activeElement.tagName !== 'INPUT') {
            deleteSelectedRegion();
        }
    }
});

// ── Mouse wheel to zoom ──
panoCanvas.addEventListener('wheel', e => {
    e.preventDefault();
    fovY = Math.max(20, Math.min(120, fovY + e.deltaY * -0.05));
    drawPano();
});

// Update cursor on hover to show handles
panoCanvas.addEventListener('mousemove', e => {
    if (dragging || regionDragMode || tiltDrag) return;
    const { nx, ny } = canvasCoords(e);
    const handleHit = hitTestHandle(nx, ny);
    if (handleHit) {
        const cursors = { tl: 'nwse-resize', tr: 'nesw-resize', bl: 'nesw-resize', br: 'nwse-resize' };
        panoCanvas.style.cursor = cursors[handleHit.handle];
    } else if (hitTestRegion(nx, ny) >= 0) {
        panoCanvas.style.cursor = 'move';
    } else {
        panoCanvas.style.cursor = markBadMode ? 'crosshair' : 'grab';
    }
});

updateForbiddenInfo();
init();
</script>
</body>
</html>"""


class PanoHandler(http.server.BaseHTTPRequestHandler):
    panos_dir = None
    state_path = None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/api/panos":
            panos = get_pano_list(self.panos_dir)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(panos).encode())
        elif self.path == "/api/state":
            state = self._load_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(state).encode())
        elif self.path.startswith("/api/forbidden/"):
            pano_id = self.path.split("/api/forbidden/", 1)[1]
            forbidden = self._load_forbidden()
            regions = forbidden.get(pano_id, [])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"regions": regions}).encode())
        elif self.path.startswith("/api/pano/"):
            pano_file = self.path.split("/")[-1]
            pano_path = Path(self.panos_dir) / pano_file
            if pano_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(pano_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        elif self.path.startswith("/api/preview"):
            from urllib.parse import urlparse, parse_qs

            qs = parse_qs(urlparse(self.path).query)
            pano_id = qs.get("id", [None])[0]
            h = float(qs.get("heading", [0])[0])
            p = float(qs.get("pitch", [0])[0])
            fv = float(qs.get("fov", [65])[0])
            if not pano_id:
                self.send_response(400)
                self.end_headers()
                return
            pano_path = Path(self.panos_dir) / f"{pano_id}.jpg"
            if not pano_path.exists():
                for ext in [".png", ".jpeg"]:
                    pp = Path(self.panos_dir) / f"{pano_id}{ext}"
                    if pp.exists():
                        pano_path = pp
                        break
            if not pano_path.exists():
                self.send_response(404)
                self.end_headers()
                return
            try:
                img = slice_perspective(
                    str(pano_path),
                    heading_deg=h,
                    pitch_deg=p,
                    fov_y_deg=fv,
                    out_w=360,
                    out_h=240,
                )
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(buf.getvalue())
            except Exception:
                self.send_response(500)
                self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/api/crop":
            res = self._save_crop(body)
        elif self.path == "/api/auto-crop":
            res = self._auto_crop(body)
        elif self.path.startswith("/api/forbidden/"):
            pano_id = self.path.split("/api/forbidden/", 1)[1]
            forbidden = self._load_forbidden()
            forbidden[pano_id] = body.get("regions", [])
            self._save_forbidden(forbidden)
            res = {"ok": True}
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(res).encode())

    def _load_state(self):
        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {}

    def _save_state(self, state):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _forbidden_path(self):
        return Path(os.path.dirname(self.state_path)) / "forbidden_regions.json"

    def _load_forbidden(self):
        p = self._forbidden_path()
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    def _save_forbidden(self, forbidden):
        os.makedirs(os.path.dirname(self._forbidden_path()), exist_ok=True)
        with open(self._forbidden_path(), "w") as f:
            json.dump(forbidden, f, indent=2)

    @staticmethod
    def detect_bad_regions(pano_path, step_deg=5.0, pitch_deg=0.0, fov_y_deg=65.0):
        """Detect bad regions by rendering perspective crops and evaluating quality.

        For each heading (at *step_deg* intervals), a perspective crop is rendered
        using slice_perspective and then scored with ``_perspective_crop_quality``.
        Consecutive bad headings are merged into forbidden regions.
        """
        try:
            Image.open(pano_path).verify()
        except Exception:
            return [
                {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "source": "corrupted_file"}
            ]

        headings = np.arange(0, 360, step_deg)
        bad_flags = []
        for h in headings:
            try:
                crop = slice_perspective(
                    str(pano_path),
                    heading_deg=float(h),
                    pitch_deg=pitch_deg,
                    fov_y_deg=fov_y_deg,
                    out_w=OUT_W,
                    out_h=OUT_H,
                )
                bad_flags.append(not _perspective_crop_quality(crop))
            except Exception:
                bad_flags.append(True)  # render failure = bad

        # Merge consecutive bad headings into regions
        regions = []
        in_run, start = False, 0
        for i in range(len(headings) + 1):
            is_bad = i < len(headings) and bad_flags[i]
            if is_bad and not in_run:
                in_run, start = True, i
            elif not is_bad and in_run:
                in_run = False
                h_start = float(headings[start])
                h_end = float(headings[i - 1]) + step_deg  # exclusive end
                # Convert heading range to normalized x coords (0-1)
                x1 = round(h_start / 360.0, 4)
                x2 = round(min(h_end, 360.0) / 360.0, 4)
                if (x2 - x1) >= 0.008:  # at least ~3 degrees wide
                    regions.append(
                        {
                            "x1": x1,
                            "y1": 0.0,
                            "x2": x2,
                            "y2": 1.0,
                            "source": "perspective_detect",
                        }
                    )
        return regions

    def _auto_crop(self, body):
        pano_id = body["pano_id"]
        target_crops = int(body.get("target_crops", 3))

        pano_path = Path(self.panos_dir) / f"{pano_id}.jpg"
        if not pano_path.exists():
            for ext in [".png", ".jpeg"]:
                p = Path(self.panos_dir) / f"{pano_id}{ext}"
                if p.exists():
                    pano_path = p
                    break

        if not pano_path.exists():
            return {"ok": False, "error": "Pano not found"}

        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        forbidden = self._load_forbidden().get(pano_id, [])

        def heading_in_forbidden(h_deg):
            for r in forbidden:
                if min(r["x1"], r["x2"]) <= (h_deg / 360.0) <= max(r["x1"], r["x2"]):
                    return True
            return False

        # Count existing valid crops
        existing = list(CROPS_DIR.glob(f"{pano_id}_*.png"))
        saved_count = len(existing)

        if saved_count >= target_crops:
            return {"ok": True, "total": saved_count}

        # Adaptive step fallback: try 90 deg -> 45 deg -> 30 deg -> 15 deg
        attempted_headings = set()
        for step_deg in [90.0, 45.0, 30.0, 15.0]:
            heading = 0.0
            while heading < 360.0:
                if heading not in attempted_headings:
                    attempted_headings.add(heading)
                    if not heading_in_forbidden(heading):
                        try:
                            img = slice_perspective(
                                str(pano_path),
                                heading_deg=heading,
                                pitch_deg=0.0,
                                fov_y_deg=65.0,
                                out_w=OUT_W,
                                out_h=OUT_H,
                            )
                        except Exception as e:
                            heading += step_deg
                            continue
                        # Ground / sky filter
                        if is_valid_sky_crop(img, min_sky_frac=0.15, min_relief=2.0):
                            saved_count += 1
                            fname = f"{pano_id}_h{heading:.0f}_p0_n{saved_count}.png"
                            img.save(CROPS_DIR / fname)
                            meta = {
                                "pano_id": pano_id,
                                "heading_deg": heading,
                                "pitch_deg": 0.0,
                                "fov_y_deg": 65.0,
                                "filename": fname,
                            }
                            with open(
                                CROPS_DIR
                                / f"{pano_id}_h{heading:.0f}_p0_n{saved_count}.json",
                                "w",
                            ) as f:
                                json.dump(meta, f, indent=2)

                            if saved_count >= target_crops:
                                state = self._load_state()
                                state[pano_id] = saved_count
                                self._save_state(state)
                                return {"ok": True, "total": saved_count}
                heading += step_deg

        state = self._load_state()
        state[pano_id] = saved_count
        self._save_state(state)
        return {"ok": True, "total": saved_count}

    def _save_crop(self, body):
        pano_id, heading_deg = body["pano_id"], float(body["heading_deg"])
        pano_path = Path(self.panos_dir) / f"{pano_id}.jpg"
        if not pano_path.exists():
            for ext in [".png", ".jpeg"]:
                p = Path(self.panos_dir) / f"{pano_id}{ext}"
                if p.exists():
                    pano_path = p
                    break

        # Block save if heading is in a forbidden region
        forbidden = self._load_forbidden().get(pano_id, [])
        h_norm = ((heading_deg % 360) + 360) % 360 / 360.0
        for r in forbidden:
            if min(r["x1"], r["x2"]) <= h_norm <= max(r["x1"], r["x2"]):
                return {"ok": False, "error": "Heading is in a bad area — move to a clear region first"}

        img = slice_perspective(
            str(pano_path),
            heading_deg=heading_deg,
            pitch_deg=0.0,
            fov_y_deg=65.0,
            out_w=OUT_W,
            out_h=OUT_H,
        )
        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        n = state.get(pano_id, 0) + 1
        state[pano_id] = n
        self._save_state(state)

        fname = f"{pano_id}_h{heading_deg:.0f}_p0_n{n}.png"
        img.save(CROPS_DIR / fname)
        meta = {
            "pano_id": pano_id,
            "heading_deg": heading_deg,
            "pitch_deg": 0.0,
            "fov_y_deg": 65.0,
            "filename": fname,
        }
        with open(CROPS_DIR / f"{pano_id}_h{heading_deg:.0f}_p0_n{n}.json", "w") as f:
            json.dump(meta, f, indent=2)
        return {"ok": True, "filename": fname}

    @classmethod
    def auto_crop_batch(cls, pano_id, panos_dir, state_path, target_crops=3):
        dummy_self = cls.__new__(cls)
        dummy_self.panos_dir = panos_dir
        dummy_self.state_path = state_path
        return dummy_self._auto_crop({"pano_id": pano_id, "target_crops": target_crops})

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Adaptive GSV Crop Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--panos-dir", type=str, default=str(ROOT / "data" / "street_view" / "panos")
    )
    parser.add_argument(
        "--auto-detect-all",
        action="store_true",
        help="Batch detect bad/cloudy zones in all panos",
    )
    parser.add_argument(
        "--auto-crop-all",
        action="store_true",
        help="Batch adaptive crop across all panos",
    )
    parser.add_argument(
        "--target-crops",
        type=int,
        default=3,
        help="Target good crops per pano (default: 3)",
    )
    args = parser.parse_args()

    PanoHandler.panos_dir = args.panos_dir
    PanoHandler.state_path = str(CROPS_DIR / "crop_state.json")

    panos = get_pano_list(args.panos_dir)

    if args.auto_detect_all:
        import tqdm

        forbidden_path = (
            Path(os.path.dirname(PanoHandler.state_path)) / "forbidden_regions.json"
        )
        forbidden = {}
        if forbidden_path.exists():
            with open(forbidden_path) as f:
                forbidden = json.load(f)

        print(f"Auto-detecting bad/cloudy areas across {len(panos)} panoramas...")
        for p in tqdm.tqdm(panos, desc="Detecting bad skylines"):
            pano_id = p["id"]
            pano_path = Path(p["path"])
            try:
                if pano_path.exists():
                    detected = PanoHandler.detect_bad_regions(pano_path)
                    manual = [
                        r
                        for r in forbidden.get(pano_id, [])
                        if r.get("source") != "auto_detect"
                    ]
                    forbidden[pano_id] = manual + detected
            except Exception:
                forbidden[pano_id] = []

        os.makedirs(os.path.dirname(forbidden_path), exist_ok=True)
        with open(forbidden_path, "w") as f:
            json.dump(forbidden, f, indent=2)

    if args.auto_crop_all:
        import tqdm

        print(
            f"\nAdaptive auto-cropping target={args.target_crops} good crops per pano..."
        )
        total_crops = 0
        for p in tqdm.tqdm(panos, desc="Adaptive auto-cropping"):
            res = PanoHandler.auto_crop_batch(
                p["id"],
                PanoHandler.panos_dir,
                PanoHandler.state_path,
                target_crops=args.target_crops,
            )
            if res.get("ok"):
                total_crops += res.get("total", 0)
        print(
            f"\nSuccessfully created/maintained {total_crops} good crops in {CROPS_DIR}!"
        )

    if args.auto_detect_all or args.auto_crop_all:
        return

    print(f"Dashboard running on: http://localhost:{args.port}")
    with socketserver.TCPServer(("", args.port), PanoHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
