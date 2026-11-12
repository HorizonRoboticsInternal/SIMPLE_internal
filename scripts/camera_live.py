"""Live view of the cameras plugged into this workstation, in a browser (2026-11-09).

Finds every /dev/video* node that delivers frames, streams each one as MJPEG and serves a page with all of them plus
resolution, measured frame rate and frame age. No CDN, no extra packages beyond OpenCV (the SIMPLE .venv has it).

    .venv/bin/python scripts/camera_live.py                     # all working cameras, http://<this pc>:8090/
    .venv/bin/python scripts/camera_live.py --source /dev/video2 --width 1280 --height 720 --fps 30
    .venv/bin/python scripts/camera_live.py --source clip.mp4   # a video file instead of a camera (loops), for testing
"""
import argparse, glob, html, json, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2


def node_name(source):
    m = re.fullmatch(r'/dev/video(\d+)', str(source))
    p = Path(f'/sys/class/video4linux/video{m.group(1)}/name') if m else None
    return p.read_text().strip() if p and p.exists() else Path(str(source)).name


class Camera:
    def __init__(self, source, width, height, fps, quality):
        self.source, self.name, self.quality = source, node_name(source), quality
        self.file = not str(source).startswith('/dev/video')
        self.cap = cv2.VideoCapture(str(source)) if self.file else cv2.VideoCapture(str(source), cv2.CAP_V4L2)
        if not self.file:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))   # USB 2 cameras need MJPG for full rate
            if width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps: self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.jpeg, self.count, self.shape, self.last, self.rate, self.error = None, 0, None, 0., 0., ''
        self.new = threading.Condition()
        self.period = 1 / (self.cap.get(cv2.CAP_PROP_FPS) or 30) if self.file else 0

    def first_frame(self, timeout=3.):
        t0 = time.time()
        while time.time() - t0 < timeout:
            ok, img = self.cap.read()
            if ok and img is not None and img.size: return img
        return None

    def run(self):
        times = []
        while True:
            t0 = time.time(); ok, img = self.cap.read()
            if not ok or img is None:
                if self.file: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                self.error = 'no frame (unplugged?)'; time.sleep(.2); continue
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            now = time.time(); times = [t for t in times if now - t < 2] + [now]
            with self.new:
                self.jpeg, self.count, self.shape, self.last, self.error = buf.tobytes(), self.count + 1, img.shape, now, ''
                self.rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.
                self.new.notify_all()
            if self.period: time.sleep(max(0., self.period - (time.time() - t0)))

    def stats(self):
        h, w = self.shape[:2] if self.shape else (0, 0)
        return dict(source=str(self.source), name=self.name, width=w, height=h, fps=round(self.rate, 1), frames=self.count,
                    age_ms=round(1000 * (time.time() - self.last)) if self.last else None, error=self.error)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera Live View</title><style>
:root{--bg:#10161f;--card:#1a2330;--text:#e3eaf2;--dim:#9fb0c3;--ok:#4cc38a;--bad:#e5534b}
@media (prefers-color-scheme: light){:root{--bg:#f4f6f9;--card:#fff;--text:#18212c;--dim:#5a6878;--ok:#1a7f4b;--bad:#c62828}}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.4 system-ui,sans-serif}
main{max-width:1600px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}p{margin:0 0 14px;color:var(--dim)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%%,480px),1fr));gap:14px}
.cam{background:var(--card);border-radius:10px;padding:10px}.cam img{width:100%%;display:block;border-radius:6px;background:#000}
.row{display:flex;justify-content:space-between;gap:8px;margin-top:8px;flex-wrap:wrap}.dim{color:var(--dim)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%%;background:var(--bad);margin-right:6px}.dot.live{background:var(--ok)}
a{color:inherit}</style></head><body><main><h1>Camera live view</h1>
<p>%(host)s · %(n)d camera(s) · stats refresh every second · <a href="/stats.json">stats.json</a></p><div class="grid">%(tiles)s</div></main>
<script>
async function tick(){try{const s=await (await fetch('/stats.json',{cache:'no-store'})).json();
 s.forEach((c,i)=>{const live=c.age_ms!==null&&c.age_ms<1000;document.getElementById('dot'+i).className='dot'+(live?' live':'');
  document.getElementById('st'+i).textContent=c.error||(c.width+'x'+c.height+' · '+c.fps+' fps · frame '+c.frames+' · age '+c.age_ms+' ms');});}catch(e){}}
setInterval(tick,1000);tick();
</script></body></html>"""
TILE = """<div class="cam"><img src="/stream/%(i)d.mjpg" alt="%(name)s"><div class="row"><span><span id="dot%(i)d" class="dot"></span>
<b>%(name)s</b> <span class="dim">%(source)s</span></span><span id="st%(i)d" class="dim">waiting…</span></div>
<div class="row"><a class="dim" href="/snapshot/%(i)d.jpg" download>snapshot</a></div></div>"""


def handler(cams, host):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def send(self, body, kind):
            self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            m = re.fullmatch(r'/(stream|snapshot)/(\d+)\.(mjpg|jpg)', path)
            if path == '/':
                tiles = ''.join(TILE % dict(i=i, name=html.escape(c.name), source=html.escape(str(c.source))) for i, c in enumerate(cams))
                return self.send((PAGE % dict(host=html.escape(host), n=len(cams), tiles=tiles)).encode(), 'text/html; charset=utf-8')
            if path == '/stats.json': return self.send(json.dumps([c.stats() for c in cams]).encode(), 'application/json')
            if not m or int(m.group(2)) >= len(cams): return self.send_error(404)
            cam = cams[int(m.group(2))]
            if m.group(1) == 'snapshot':
                return self.send(cam.jpeg, 'image/jpeg') if cam.jpeg else self.send_error(503)
            self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store'); self.end_headers()
            seen = -1
            try:
                while True:
                    with cam.new:
                        cam.new.wait_for(lambda: cam.count != seen, timeout=2)
                        jpeg, seen = cam.jpeg, cam.count
                    if jpeg is None: continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpeg) + jpeg + b'\r\n')
            except (BrokenPipeError, ConnectionResetError): pass
    return H


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', action='append', help='camera node or video file (repeatable); default: every working /dev/video*')
    p.add_argument('--width', type=int); p.add_argument('--height', type=int); p.add_argument('--fps', type=int)
    p.add_argument('--quality', type=int, default=80, help='JPEG quality of the stream')
    p.add_argument('--host', default='0.0.0.0'); p.add_argument('--port', type=int, default=8090); a = p.parse_args()
    sources = a.source or sorted(glob.glob('/dev/video*'), key=lambda s: int(re.sub(r'\D', '', s)))
    cams = []
    for s in sources:
        cam = Camera(s, a.width, a.height, a.fps, a.quality)
        img = cam.first_frame() if cam.cap.isOpened() else None
        if img is None:
            print(f'{s} ({cam.name}): opens no picture, skipped'); cam.cap.release(); continue
        print(f'{s} ({cam.name}): {img.shape[1]}x{img.shape[0]} OK'); cams.append(cam)
    if not cams: raise SystemExit('no camera delivered a frame (ls /dev/video*; lsusb; journalctl -k | tail)')
    server = ThreadingHTTPServer((a.host, a.port), handler(cams, f'{a.host}:{a.port}'))   # bind first: a busy port fails cleanly
    for cam in cams: threading.Thread(target=cam.run, daemon=True).start()
    print(f'live view: http://{a.host if a.host != "0.0.0.0" else "localhost"}:{a.port}/  (Ctrl+C to stop)', flush=True)
    server.serve_forever()


if __name__ == '__main__': main()
