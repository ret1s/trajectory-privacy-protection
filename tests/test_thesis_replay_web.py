"""HTTP isolation and fidelity of the frozen teaching replay, not new model tests."""
import gzip
import hashlib
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from web.thesis_demo import create_server, verify_bundle, DEFAULT


@pytest.fixture
def bundle(tmp_path):
    (tmp_path/'data').mkdir()
    files={'index.html':b'<h1>Demo</h1>', 'logic.mjs':b'export const value=1;',
           'data/index.json.gz':gzip.compress(b'{"replay_only":true}',mtime=0)}
    for name,content in files.items(): (tmp_path/name).write_bytes(content)
    manifest={'schema':'geoi-bq-demo-manifest-v1','files_sha256':{
        name:hashlib.sha256(content).hexdigest() for name,content in files.items()}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'not-allowlisted.txt').write_text('must not be served')
    return tmp_path


@pytest.fixture
def endpoint(bundle):
    server=create_server(bundle,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown();thread.join();server.server_close()


def test_json_decompression_head_and_module_mime(endpoint):
    with urlopen(endpoint+'/data/index.json') as response:
        assert json.load(response)=={'replay_only':True}
        assert response.headers.get_content_type()=='application/json'
        assert response.headers['X-Content-Type-Options']=='nosniff'
        assert "script-src 'self'" in response.headers['Content-Security-Policy']
    with urlopen(Request(endpoint+'/data/index.json',method='HEAD')) as response:
        assert response.read()==b'' and int(response.headers['Content-Length'])>0
    with urlopen(endpoint+'/logic.mjs') as response:
        assert response.headers.get_content_type() in ('text/javascript','application/javascript')


@pytest.mark.parametrize('path',['/not-allowlisted.txt','/data/','/../README.md',
                                '/%2e%2e/README.md','/data/%2e%2e/not-allowlisted.txt'])
def test_only_manifest_files_served(endpoint,path):
    with pytest.raises(HTTPError) as error: urlopen(endpoint+path)
    assert error.value.code==404


def test_http_cannot_launch_experiment(endpoint):
    with pytest.raises(HTTPError) as error:
        urlopen(Request(endpoint+'/',data=b'{}',method='POST'))
    assert error.value.code==501


def test_tampered_or_escaping_bundle_rejected(bundle):
    (bundle/'index.html').write_text('changed')
    with pytest.raises(ValueError,match='checksum'): verify_bundle(bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    manifest['files_sha256']={'../outside':'a'*64}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='checksum'): verify_bundle(bundle)


def test_complete_demo_cohort_and_saved_wire():
    """Inspect every retained draw; exporting never resamples proposed outputs."""
    from experiments.export_thesis_demo_20261111 import read, check, PRIMARY, ORIGINAL
    check(DEFAULT)
    index=read(DEFAULT/'data/index.json.gz')
    assert len({s['family'] for s in index['sessions']})==20
    assert len(index['sessions'])==40
    primary={};old={}
    for session in index['sessions']:
        family=session['family']
        if family not in primary:
            primary[family]={t['session_id']:t for t in read(PRIMARY/'families'/f'{family}.json.gz')['trips']}
            old[family]={t['session_id']:t for t in read(ORIGINAL/'families'/f'{family}.json.gz')['trips']}
        assert len(session['draws'])==3
        for draw in session['draws']:
            run=read(DEFAULT/'data/runs'/f'{draw}.json.gz')
            assert run['ledger']==old[family][draw]['GeoI_ledger']
            assert max(x['spent_units'] for x in run['ledger'])<=23
            assert len(run['certificates'])==31
            for method,stream in run['streams'].items():
                expected=primary[family][draw]['events'][method]
                assert [(e['t'],e['coordinates']) for e in stream]==[
                    (e['timestamp_s'],e['coordinates']) for e in expected]
            assert run['utility']==primary[family][draw]['utility']
    results=read(DEFAULT/'data/results.json.gz');source=read(PRIMARY/'results.json')
    assert results['cost']==source['cost'] and results['comparisons']==source['comparisons']
