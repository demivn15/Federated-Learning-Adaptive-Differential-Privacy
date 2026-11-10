"""Local preview of actual standalone smoke journals, not a NEBULA deployment."""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from jinja2 import ChoiceLoader, DictLoader, Environment, FileSystemLoader, select_autoescape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "nebula"))
from nebula.frontend.research_view import research_payload


def serve(output, port):
    frontend = HERE / "nebula" / "nebula" / "frontend"
    env = Environment(loader=ChoiceLoader([DictLoader({"layout.html": '<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Project 2 · standalone test preview</title><style>body{margin:0;background:#091421}.btn{display:inline-block;padding:8px 12px;border:1px solid #91a1b6;border-radius:5px;color:inherit;text-decoration:none;cursor:pointer}button{color:#edf2f7!important;background:#21354f}</style></head><body>{% block body %}{% endblock %}</body></html>'}), FileSystemLoader(frontend / "templates")]), autoescape=select_autoescape())
    env.globals["url_for"] = lambda name, path: "/static/" + path.lstrip("/")
    names = tuple(p.name for p in sorted((output / "config").iterdir()) if (p / "research_manifest.json").is_file())
    if not names:
        raise ValueError("No measured smoke scenarios found in the selected output")
    def metadata(name):
        return {"name": name, "title": name + " · standalone smoke", "status": "completed"}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self.send_response(302)
                preferred = "smoke_adaptive" if "smoke_adaptive" in names else names[-1]
                self.send_header("Location", f"/platform/dashboard/{preferred}/research")
                self.end_headers()
                return
            if path == "/platform/api/dashboard":
                self.reply(json.dumps([metadata(n) for n in names]).encode(), "application/json")
                return
            if path.startswith("/static/"):
                file = (frontend / "static" / path.removeprefix("/static/")).resolve()
                if not file.is_relative_to((frontend / "static").resolve()) or not file.is_file():
                    self.send_error(404)
                    return
                self.reply(file.read_bytes(), "text/css" if file.suffix == ".css" else "text/javascript")
                return
            tokens = path.strip("/").split("/")
            name = tokens[-2] if len(tokens) > 1 else ""
            if name not in names or not path.endswith("/research"):
                self.send_error(404)
                return
            if path.startswith("/platform/api/"):
                payload = research_payload(output / "logs" / name, output / "config" / name, metadata(name))
                self.reply(json.dumps(payload).encode(), "application/json")
            else:
                html = env.get_template("research.html").render(scenario_name=name)
                self.reply(html.encode(), "text/html; charset=utf-8")
        def reply(self, data, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    print(f"Standalone smoke preview: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=6061)
    args = parser.parse_args()
    serve(args.output, args.port)
