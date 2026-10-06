#!/usr/bin/env python3
"""Smoke checks for the 4x Spark Step-5 server: text answer, tool call, vision, video. Prints PASS/FAIL/SKIP per check.

  python3 scripts/smoke.py [--key-file PATH] [--allow-skip] [http://<head>:8000]
API key: $STEP5_API_KEY, else --key-file, else $KEY_FILE, else ~/.step5-sparks/api_key (written by launch.sh).

No request carries max_tokens / max_completion_tokens: every answer stops naturally.
The video check builds a 4 s MP4 (red, then blue) in memory with OpenCV (pip install opencv-python-headless numpy;
the server image has it). Without OpenCV the check is SKIP, and any SKIP makes the run exit 1 (the skipped checks are
listed) unless --allow-skip is given. Exit 0 only when every check ran and passed."""
import argparse, base64, json, os, struct, sys, tempfile, urllib.request, zlib

ap = argparse.ArgumentParser(description="Step-5 server smoke checks (text, tools, image, video).")
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
ap.add_argument("--key-file", default=os.environ.get("KEY_FILE", "~/.step5-sparks/api_key"),
                help="API key file (default: $KEY_FILE or ~/.step5-sparks/api_key); ignored when STEP5_API_KEY is set")
ap.add_argument("--allow-skip", action="store_true", help="exit 0 even if a check was skipped (e.g. no OpenCV)")
A = ap.parse_args()
URL = A.url.rstrip("/")
if URL.endswith("/v1"):
    URL = URL[:-3]
KEY = os.environ.get("STEP5_API_KEY")
if not KEY:
    try:
        KEY = open(os.path.expanduser(A.key_file)).read().strip()
    except OSError as e:
        sys.exit(f"no API key: set STEP5_API_KEY or point --key-file / KEY_FILE at launch.sh's api_key ({e})")


def call(payload):
    assert "max_tokens" not in payload and "max_completion_tokens" not in payload
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


def model():
    r = urllib.request.Request(URL + "/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(r, timeout=30))["data"][0]["id"]


def png_halves():
    """128x128 PNG: left half red, right half blue."""
    w = h = 128
    rows = b"".join(b"\x00" + b"".join((b"\xff\x00\x00" if x < w // 2 else b"\x00\x00\xff") for x in range(w)) for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def mp4_red_then_blue():
    """4 s, 8 fps, 256x256 MP4: 2 s solid red, then 2 s solid blue. Returns bytes, or None without OpenCV."""
    try:
        import cv2, numpy as np
    except ImportError:
        return None
    w = h = 256; fps = 8
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.mp4")
        for fourcc in ("mp4v", "avc1"):
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if vw.isOpened():
                break
        else:
            return None
        for i in range(4 * fps):
            frame = np.zeros((h, w, 3), np.uint8)
            frame[:] = (0, 0, 255) if i < 2 * fps else (255, 0, 0)   # OpenCV frames are BGR
            vw.write(frame)
        vw.release()
        with open(path, "rb") as f:
            return f.read()


def ordered(txt, a, b):
    return a in txt and b in txt and txt.find(a) < txt.find(b)


M = model(); ok = True; skipped = []
r = call({"model": M, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}], "temperature": 0})
txt = r["choices"][0]["message"]["content"] or ""
print("text  ", "PASS" if "391" in txt else "FAIL", repr(txt[-120:]), r["usage"]); ok &= "391" in txt

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = call({"model": M, "messages": [{"role": "user", "content": "What's the weather in Warsaw right now?"}], "tools": tools,
          "temperature": 0})
tc = r["choices"][0]["message"].get("tool_calls") or []
good = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "warsaw" in tc[0]["function"]["arguments"].lower()
print("tools ", "PASS" if good else "FAIL", json.dumps(tc)[:200]); ok &= good

img = "data:image/png;base64," + base64.b64encode(png_halves()).decode()
r = call({"model": M, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": img}},
    {"type": "text", "text": "This image has two halves. What colour is the left half and what colour is the right half?"}]}]})
txt = (r["choices"][0]["message"]["content"] or "").lower()
good = ordered(txt, "red", "blue")
print("vision", "PASS" if good else "FAIL", repr(txt[-160:]), r["usage"]); ok &= good

clip = mp4_red_then_blue()
if clip is None:
    print("video  SKIP (needs OpenCV + numpy to build the test clip: pip install opencv-python-headless numpy, "
          "or run smoke.py inside the server image)")
    skipped.append("video")
else:
    vid = "data:video/mp4;base64," + base64.b64encode(clip).decode()
    r = call({"model": M, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": vid}},
        {"type": "text", "text": "This video shows one solid colour and then a different solid colour. "
                                 "Which colour comes first and which comes second?"}]}]})
    txt = (r["choices"][0]["message"]["content"] or "").lower()
    good = ordered(txt, "red", "blue")
    print("video ", "PASS" if good else "FAIL", repr(txt[-160:]), r["usage"]); ok &= good
if skipped:
    print(f"SKIPPED: {', '.join(skipped)} -- " + ("not counted as failure (--allow-skip)" if A.allow_skip
          else "the run is NOT a full pass; exit 1 (use --allow-skip to accept)"))
    if not A.allow_skip:
        ok = False
print("ALL PASS" if ok and not skipped else "PASS (with skips)" if ok else "FAIL")
sys.exit(0 if ok else 1)
