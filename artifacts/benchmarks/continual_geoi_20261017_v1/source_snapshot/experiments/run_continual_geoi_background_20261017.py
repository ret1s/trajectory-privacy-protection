"""Detached one-shot worker; explicit flag enables the authorized Git workflow.

Progress/exit status are outside the repository; no session/RNG secrets are
written to stdout. The study freezes itself before generating private draws.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT='artifacts/benchmarks/continual_geoi_20261017_v1'
WORK=Path('/private/tmp/trajectory-continual-geoi-20261017-v1')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit-and-push',action='store_true')
    args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True)
    status=WORK/'worker_status.json'
    def update(state,**fields):
        temporary=status.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state,pid=os.getpid(),artifact=ARTIFACT,**fields),indent=2)+'\n')
        temporary.replace(status)
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONUNBUFFERED='1')
    try:
        update('running')
        subprocess.run([sys.executable,'-m','experiments.continual_geoi_study_20261017','run'],cwd=ROOT,env=environment,check=True)
        update('verified')
        if args.commit_and_push:
            # Never sweep unrelated user edits/staging into this commit.
            subprocess.run(['git','add','--',ARTIFACT],cwd=ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','Record continual Geo-I pilot results','--',ARTIFACT],cwd=ROOT,check=True)
            update('committed',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
            subprocess.run(['git','push'],cwd=ROOT,check=True)
        update('complete',committed_and_pushed=args.commit_and_push)
    except Exception as exc:
        update('failed',error=str(exc))
        raise


if __name__=='__main__':main()
