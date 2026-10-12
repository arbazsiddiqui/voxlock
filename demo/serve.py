"""Serve the demo locally with the headers that let onnxruntime-web use threads: python demo/serve.py [port]"""
import sys, http.server, functools
from pathlib import Path
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
print(f"http://127.0.0.1:{port}")
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory=str(Path(__file__).parent))).serve_forever()
