"""Local, read-only teaching demo. Run: python -m web.thesis_demo --port 5011.

Uses only Python's standard library at runtime. The exporter authenticates and
packages evidence separately; HTTP never runs models or experiments.
"""
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import mimetypes
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/demos/geoi_bq_20261111_v1'


def verify_bundle(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema')!='geoi-bq-demo-manifest-v1':
        raise ValueError('Unsupported demo manifest')
    for name,digest in manifest['files_sha256'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Demo checksum mismatch: '+name)
    return manifest


class Handler(BaseHTTPRequestHandler):
    def __init__(self,*args,directory,manifest,**kwargs):
        self.directory=Path(directory).resolve();self.manifest=manifest
        super().__init__(*args,**kwargs)

    def do_GET(self): self.send_content(False)
    def do_HEAD(self): self.send_content(True)

    def send_content(self,head):
        name=unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
        if name.endswith('.json') and name!='manifest.json': name+='.gz'
        if name not in self.manifest['files_sha256'] and name!='manifest.json':
            self.send_error(404);return
        path=(self.directory/name).resolve()
        if not path.is_relative_to(self.directory): self.send_error(404);return
        raw=path.read_bytes()
        # Decompress server-side so fetch works in every browser and without JS libraries.
        kind=path.name[:-3] if name.endswith('.gz') else path.name
        if name.endswith('.gz'): raw=gzip.decompress(raw)
        self.send_response(200)
        self.send_header('Content-Type',mimetypes.guess_type(kind)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-cache')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'self'; frame-ancestors 'self'")
        self.end_headers()
        if not head: self.wfile.write(raw)


def create_server(directory=DEFAULT,port=5011):
    manifest=verify_bundle(directory)
    return ThreadingHTTPServer(('127.0.0.1',port),partial(Handler,directory=directory,manifest=manifest))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=5011)
    p.add_argument('--directory',type=Path,default=DEFAULT);args=p.parse_args()
    try: server=create_server(args.directory,args.port)
    except FileNotFoundError: p.error('Build evidence first: python -m experiments.export_thesis_demo_20261111')
    print(f'GeoI-BQ demo: http://127.0.0.1:{server.server_port} · read-only frozen replay',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=='__main__': main()
