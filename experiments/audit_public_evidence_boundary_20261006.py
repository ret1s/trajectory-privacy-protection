"""Sanitized read-only secret-boundary check of Git-visible NEW files only.

Private key bytes, digests and private key filenames are never returned or
printed. Ignored files and existing tracked modifications are outside this
audit. Non-secret local storage paths in reproduction provenance are counted
separately from exported secret/state material. No evidence file is modified.
"""
import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
GIT_FILE_LIMIT=100*2**20
PRIVATE_SUFFIXES=('.key','.sqlite','.sqlite3','.db','.db-wal','.sqlite-wal','.sqlite-shm')
PRIVATE_COMPONENTS={'private_state','paired_development_checkpoint','private_rng'}
PRIVATE_PATH_REFERENCE=re.compile(rb'(?:/private/tmp|/tmp)/[^\s"\x27<>`]*?(?:private_state|paired_development_checkpoint|private_rng)')


def git_new_files(root=ROOT):
    root=Path(root)
    others=subprocess.run(['git','ls-files','--others','--exclude-standard','-z'],cwd=root,
        check=True,capture_output=True).stdout
    added=subprocess.run(['git','diff','--cached','--name-only','--diff-filter=A','-z'],cwd=root,
        check=True,capture_output=True).stdout
    return sorted(set(p.decode() for p in (others+added).split(b'\0') if p))


def discover_private_workdirs(tmp=Path('/private/tmp')):
    """Only qplanner work roots; do not inspect unrelated ignored private state."""
    roots=[p for p in Path(tmp).iterdir() if p.is_dir() and not p.is_symlink() and 'qplanner' in p.name]
    native=Path(tmp)/'trajectory-native-future-evaluation-v1'
    if native.is_dir():
        roots.extend(p for p in native.iterdir() if p.is_dir() and not p.is_symlink() and 'qplanner' in p.name)
    return sorted(set(roots))


def private_patterns(workdirs):
    keys=set(); files=0; invalid=0; ignored_symlink=0
    for work in workdirs:
        for path in Path(work).rglob('*.key'):
            if path.is_symlink(): ignored_symlink+=1;continue
            if not path.is_file(): continue
            files+=1; value=path.read_bytes()
            if len(value)!=32: invalid+=1;continue
            keys.add(value)
    patterns=set()
    for key in keys:
        patterns.update((key,key.hex().encode(),key.hex().upper().encode()))
        for encoder in (base64.b64encode,base64.urlsafe_b64encode):
            value=encoder(key);patterns.update((value,value.rstrip(b'=')))
    counts=dict(private_workdir_count=len(workdirs),private_key_file_count=files,
        distinct_32byte_private_key_count=len(keys),private_non32byte_key_file_count=invalid,
        private_symlink_key_files_not_read=ignored_symlink,encoding_variant_count=len(patterns))
    return tuple(patterns),counts


def audit_files(root,names,patterns):
    root=Path(root); count=0; total=0; decoded=0; expanded=0; matches=0; decoded_matches=0
    private_files=0; private_locations=0; sqlite_magic=0; oversized=0; errors=0; changing=0
    references=0; files=[]
    for name in names:
        path=root/name
        if path.is_symlink() or not path.is_file(): errors+=1;continue
        before=path.stat();content=path.read_bytes();after=path.stat()
        count+=1; total+=len(content)
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): changing+=1
        private_name=path.name.lower().endswith(PRIVATE_SUFFIXES)
        private_location=bool(PRIVATE_COMPONENTS.intersection(Path(name).parts))
        private_files+=private_name;private_locations+=private_location
        sqlite_magic+=content.startswith(b'SQLite format 3\0')
        oversized+=len(content)>GIT_FILE_LIMIT
        matches+=any(pattern in content for pattern in patterns)
        expanded_content=None
        if path.suffix=='.gz':
            try:
                expanded_content=gzip.decompress(content);decoded+=1;expanded+=len(expanded_content)
                decoded_matches+=any(pattern in expanded_content for pattern in patterns)
            except (OSError,EOFError): errors+=1
        references+=bool(PRIVATE_PATH_REFERENCE.search(content) or (expanded_content is not None
            and PRIVATE_PATH_REFERENCE.search(expanded_content)))
        files.append(dict(path='[private storage file redacted]' if private_name or private_location else name,
            bytes=len(content),MiB=len(content)/2**20))
    gates=dict(no_actual_key_or_database_files=private_files==0 and sqlite_magic==0,
        no_actual_private_state_paths=private_locations==0,
        no_private_key_bytes_raw_hex_or_base64=matches==0 and decoded_matches==0,
        all_selected_files_regular_and_decodable=errors==0,
        selected_files_stable_during_scan=changing==0,
        git_each_file_within_100MiB=oversized==0)
    return dict(status='pass' if all(gates.values()) else 'fail',gates=gates,
        selected_inventory_file_count=len(names),scanned_regular_file_count=count,
        selected_file_bytes=total,gzip_file_count=decoded,gzip_uncompressed_bytes=expanded,
        actual_private_key_database_filename_count=private_files,actual_private_storage_path_count=private_locations,
        sqlite_magic_header_count=sqlite_magic,private_key_matching_file_count=matches,
        private_key_matching_gzip_expanded_file_count=decoded_matches,
        non_secret_private_storage_path_reference_file_count=references,
        path_reference_scope='allowed non-secret reproduction/provenance text; not actual key/state files',
        decoding_or_nonregular_errors=errors,files_changed_during_scan=changing,
        files_over_git_100MiB_limit=oversized,largest_new_files=sorted(files,key=lambda row:row['bytes'],reverse=True)[:10])


def audit(root=ROOT,workdirs=None):
    root=Path(root);names=git_new_files(root)
    patterns,key_counts=private_patterns(discover_private_workdirs() if workdirs is None else workdirs)
    result=audit_files(root,names,patterns)
    if key_counts['private_non32byte_key_file_count'] or not key_counts['distinct_32byte_private_key_count']:
        result['gates']['private32byte_reference_set_complete']=False;result['status']='fail'
    else:result['gates']['private32byte_reference_set_complete']=True
    after=git_new_files(root)
    result.update(schema='qplanner-sanitized-public-boundary-audit-v1',
        completed_utc=datetime.now(timezone.utc).isoformat(),private_reference_counts=key_counts,
        new_visible_files_created_during_scan=len(set(after)-set(names)),
        inventory_scope='Git NEW staged additions and untracked files excluding ignored; existing tracked modifications excluded',
        temporary_private_key_bytes_or_digests_exported=False,
        scanner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations='exact32byte/raw/hex/base64 matching plus gzip expansion; not proof against encrypted, fragmented, arbitrary encodings or unrelated secrets; rerun after final generation')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--private-workdir',type=Path,action='append')
    parser.add_argument('--output',type=Path,required=True,help='NEW sanitized JSON report; never overwrite')
    args=parser.parse_args();value=audit(args.root,args.private_workdir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(value,indent=2,allow_nan=False))
    return 0 if value['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
