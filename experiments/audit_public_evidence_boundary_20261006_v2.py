"""Add exact private-key SHA256 digest checks without changing the first audit.

No key bytes, actual key digests or private key filenames are returned/printed.
The original scanner/source/report remain immutable. Git-visible NEW files
and gzip expansion are checked against both material and digest variants.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments import audit_public_evidence_boundary_20261006 as original

ROOT=original.ROOT


def private_patterns(workdirs):
    material,counts=original.private_patterns(workdirs)
    digest_patterns=set()
    # Among original raw/hex/base64 patterns, only raw key bytes have length32.
    for key in material:
        if len(key)!=32:continue
        digest=hashlib.sha256(key).digest()
        digest_patterns.update((digest,digest.hex().encode(),digest.hex().upper().encode()))
        for encoder in (base64.b64encode,base64.urlsafe_b64encode):
            encoded=encoder(digest);digest_patterns.update((encoded,encoded.rstrip(b'=')))
    counts['sha256_digest_encoding_variant_count']=len(digest_patterns)
    combined=set(material)|digest_patterns
    counts['combined_key_material_and_digest_variant_count']=len(combined)
    return tuple(combined),counts


def audit(root=ROOT,workdirs=None):
    root=Path(root);names=original.git_new_files(root)
    patterns,counts=private_patterns(original.discover_private_workdirs() if workdirs is None else workdirs)
    value=original.audit_files(root,names,patterns)
    value['gates']['no_private_key_or_SHA256_digest_bytes']=value['gates'].pop('no_private_key_bytes_raw_hex_or_base64')
    value['private_key_or_sha256_digest_matching_file_count']=value.pop('private_key_matching_file_count')
    value['private_key_or_sha256_digest_matching_gzip_expanded_file_count']=value.pop('private_key_matching_gzip_expanded_file_count')
    complete=bool(counts['distinct_32byte_private_key_count']) and counts['private_non32byte_key_file_count']==0
    value['gates']['private32byte_reference_set_complete']=complete
    if not complete:value['status']='fail'
    after=original.git_new_files(root)
    value.update(schema='qplanner-sanitized-public-boundary-audit-v2',
        completed_utc=datetime.now(timezone.utc).isoformat(),private_reference_counts=counts,
        new_visible_files_created_during_scan=len(set(after)-set(names)),
        inventory_scope='Git NEW staged additions and untracked files excluding ignored; existing tracked modifications excluded',
        temporary_private_key_bytes_or_digests_exported=False,
        scanner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_scanner_source_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        limitations='exact32byte/raw/hex/base64 key material and SHA256digest variants plus gzip expansion; not proof against encrypted/fragmented/arbitrary encodings or unrelated secrets')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--private-workdir',type=Path,action='append')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=audit(args.root,args.private_workdir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(value,indent=2,allow_nan=False))
    return 0 if value['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
