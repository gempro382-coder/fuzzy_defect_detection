"""
Build a self-contained offline snapshot (preview_dashboard.html) of the modern
dashboard with a real fused analysis embedded, so it renders without the server.
"""
import os, re, json, base64, glob, sys
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from inference_pipeline import InferencePipeline   # noqa: E402


def thumb(path, size=(80, 52)):
    img = cv2.resize(cv2.imread(path), size)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def main(sample_glob="../data/custom_samples/dent/*.png"):
    css = open(os.path.join(HERE, "static", "css", "style.css"), encoding='utf-8').read()
    js_gl = open(os.path.join(HERE, "static", "js", "nf-gl.js"), encoding='utf-8').read()
    js_in = open(os.path.join(HERE, "static", "js", "nf-intro.js"), encoding='utf-8').read()
    js_st = open(os.path.join(HERE, "static", "js", "nf-stage.js"), encoding='utf-8').read()
    js_app = open(os.path.join(HERE, "static", "js", "app.js"), encoding='utf-8').read()
    
    js = js_gl + "\n" + js_in + "\n" + js_st + "\n" + js_app

    html = open(os.path.join(HERE, "templates", "index.html"), encoding='utf-8').read()

    imgs = glob.glob(os.path.join(HERE, sample_glob))
    if not imgs:
        print(f"Warning: No images found for {sample_glob}")
        img = None
    else:
        img = imgs[0]
        
    if img:
        result = InferencePipeline().run(img)
    else:
        result = {"error": "no image"}

    # inline css, drop font link & external js
    html = re.sub(r'<link rel="preconnect"[^>]*>', "", html)
    html = re.sub(r'<link href="https://fonts[^>]*>', "", html)
    html = re.sub(r'<link rel="stylesheet"[^>]*style\.css[^>]*>',
                  f"<style>\n{css}\n</style>", html)
    html = re.sub(r'<script.*?</script>', "", html, flags=re.DOTALL)

    # static sample chips
    chips = []
    for cls in ["good", "scratch", "crack", "dent", "corrosion"]:
        ps = glob.glob(os.path.join(ROOT, "data", "custom_samples", cls, "*.png"))[:2]
        for p in ps:
            chips.append(f'<div class="sample" title="{cls}">'
                         f'<img src="{thumb(p)}"><span>{cls}</span></div>')
    html = html.replace('<div class="samples" id="samples"><p class="muted sm">Loading…</p></div>',
                        '<div class="samples" id="samples">' + "".join(chips) + "</div>")

    # neutralise live fetches; expose render; auto-run embedded result.
    # Redirect every fetch() to a rejected promise so the offline page never
    # tries to hit the server (its .catch handlers degrade gracefully; the
    # samples list is pre-filled with static chips above).
    # app.js closes with a top-level `})();` at column 0; nested IIFEs inside it
    # are indented, so anchor on the line start to patch only the outer one.
    js_mod, n = re.subn(r"(?m)^\}\)\(\);", "  window.__render = render;\n})();", js)
    if n != 1:
        raise RuntimeError(f"expected 1 top-level IIFE tail to patch, found {n}")
    js_mod = "const fetch = () => Promise.reject(new Error('offline preview'));\n" + js_mod

    boot = (f"<script>{js_mod}</script>\n<script>\n"
            f"document.addEventListener('DOMContentLoaded',function(){{\n"
            f"  var rep={json.dumps(result)};\n"
            f"  document.getElementById('emptyState').hidden=true;\n"
            f"  document.getElementById('progressPanel').hidden=true;\n"
            f"  setTimeout(function(){{ if(window.__render) window.__render(rep); }},60);\n"
            f"}});\n</script>\n")
    html = html.replace("</body>", boot + "</body>")

    out = os.path.join(ROOT, "preview_dashboard.html")
    open(out, "w", encoding='utf-8').write(html)
    if img:
        print(f"Wrote {out} ({len(html)//1024} KB) — verdict "
              f"{result['final']['decision']}/{result['final']['defect_type']}")
    else:
        print(f"Wrote {out} ({len(html)//1024} KB) without image processing")

if __name__ == "__main__":
    main()
