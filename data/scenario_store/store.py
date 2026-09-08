"""Atomic immutable dataset revisions, not a protection/attacker data endpoint.

The SQL layer enforces relational integrity. Import also checks ordering, split
isolation and declared counts. Physical SUMO/scenario gates remain the job of
the independent native verifiers. Never expose this database to the LSP.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sqlite3


class StoreError(ValueError):
    """A dataset or revision does not satisfy the store contract."""


TABLES = ('families', 'sessions', 'points', 'cases', 'records',
          'record_sessions', 'observations', 'source_hashes')
POINT_FIELDS = ('time_s', 'lat', 'lon', 'speed_m_s', 'lane_id', 'edge_id',
                'lane_pos_m', 'angle_deg')
TOP_FIELDS = {'families', 'traces', 'records', 'catalogue', 'source_sha256'}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def content_hash(value):
    return sha256(json_text(value).encode()).hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise StoreError(f'duplicate JSON key: {key}')
        value[key] = item
    return value


def read_bundle(path):
    raw = Path(path).read_bytes()
    bundle = json.loads(raw, object_pairs_hook=_unique_object)
    # Also rejects NaN/Infinity, including overflow such as 1e999.
    json_text(bundle)
    return bundle, sha256(raw).hexdigest()


def _rest(value, excluded):
    return {k: v for k, v in value.items() if k not in excluded}


@lru_cache(maxsize=1)
def _schema():
    script = Path(__file__).with_name('schema.sql').read_text()
    for table in TABLES:
        script += f"""
CREATE TRIGGER {table}_insert BEFORE INSERT ON {table} BEGIN
  SELECT CASE WHEN (SELECT state FROM releases WHERE release_id=NEW.release_id) IS NOT 'staging'
    THEN RAISE(ABORT,'insert into staging release only') END;
END;
CREATE TRIGGER {table}_update BEFORE UPDATE ON {table} BEGIN
  SELECT RAISE(ABORT,'rows are immutable; create a revision');
END;
CREATE TRIGGER {table}_delete BEFORE DELETE ON {table} BEGIN
  SELECT RAISE(ABORT,'rows are immutable; create a revision');
END;
"""
    return script


def _schema_digest(connection):
    return content_hash([tuple(row) for row in connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name")])


@lru_cache(maxsize=1)
def _expected_schema_digest():
    connection = sqlite3.connect(':memory:')
    try:
        connection.executescript(_schema())
        return _schema_digest(connection)
    finally:
        connection.close()


def validate_bundle(bundle):
    """Format, indexing, identity partitioning and counts, not physical gates."""
    if bundle.get('schema') not in {'urban-scenario-suite-v1', 'urban-scenario-suite-v2'}:
        raise StoreError('unsupported dataset schema; add an explicit adapter')
    traces, records = bundle['traces'], bundle['records']
    sessions, entities = {}, {}
    family_ids = set()
    for family in bundle['families']:
        fid = family['family_id']
        if fid in family_ids:
            raise StoreError('duplicate family')
        family_ids.add(fid)
        for session in family['sessions']:
            sid = session['session_id']
            if sid in sessions:
                raise StoreError('duplicate session')
            sessions[sid] = (fid, family['split'])
            trace = traces[sid]
            if not trace:
                raise StoreError('empty trace')
            if any(set(p) != set(POINT_FIELDS) for p in trace):
                raise StoreError('unexpected FCD fields; add a schema migration')
            if any(b['time_s'] <= a['time_s'] for a, b in zip(trace, trace[1:])):
                raise StoreError('FCD times must be strictly increasing')
            for kind in ('person_id', 'device_id', 'physical_vehicle_id'):
                key = (kind, session[kind])
                previous = entities.setdefault(key, [])
                if any(other[0] != fid for other in previous):
                    raise StoreError('related identity crosses families/splits')
                interval = (trace[0]['time_s'], trace[-1]['time_s'])
                if any(max(interval[0], x[1]) <= min(interval[1], x[2]) for x in previous):
                    raise StoreError('same identity has overlapping sessions')
                previous.append((fid, *interval))
    if set(sessions) != set(traces):
        raise StoreError('trace/session key mismatch')
    catalogue = {c['case_id']: c for c in bundle['catalogue']}
    if len(catalogue) != len(bundle['catalogue']):
        raise StoreError('duplicate catalogue case')
    ids = set()
    for record in records:
        if record['record_id'] in ids:
            raise StoreError('duplicate record')
        ids.add(record['record_id'])
        if record['scenario'] != catalogue[record['case_id']]['scenario']:
            raise StoreError('record/case scenario mismatch')
        slots, indices = record['session_ids'], record['observed_indices']
        if not slots or len(slots) != len(indices) or len(set(slots)) != len(slots):
            raise StoreError('invalid record slots')
        for sid, allowed in zip(slots, indices):
            if sessions[sid] != (record['family_id'], record['split']):
                raise StoreError('record/session family or split mismatch')
            if not allowed or any(type(i) is not int or i < 0 or i >= len(traces[sid]) for i in allowed):
                raise StoreError('observation index outside trace')
            if any(b <= a for a, b in zip(allowed, allowed[1:])):
                raise StoreError('observation indices must be strictly increasing')
        clock = record['observation_policy']['clock']
        if clock not in {'common_pair_epoch', 'relative_to_first_allowed_sample_per_session'}:
            raise StoreError('unsupported observation clock')
        labels = record['labels']
        target_slot = labels.get('target_slot', 0)
        if type(target_slot) is not int or not 0 <= target_slot < len(slots):
            raise StoreError('invalid target slot')
        # Generic index integrity only; what is hidden/eligible is scenario-specific.
        for name in ('target_index', 'future_index'):
            if name in labels:
                i = labels[name]
                if type(i) is not int or not 0 <= i < len(traces[slots[target_slot]]):
                    raise StoreError(f'{name} outside target trace')
    by_case, by_scenario = Counter(r['case_id'] for r in records), Counter(r['scenario'] for r in records)
    expected = {'families': len(family_ids), 'sessions': len(sessions),
                'raw_fcd_samples': sum(map(len, traces.values())),
                'scenario_records': len(records), 'generated_subcases': len(by_case),
                'declared_subcases': len(catalogue), 'generated_scenarios': len(by_scenario),
                'by_case': dict(by_case), 'by_scenario': dict(by_scenario)}
    if bundle['summary'] != expected:
        raise StoreError('declared summary disagrees with rows')
    for cid, case in catalogue.items():
        if case['generated_records'] != by_case[cid]:
            raise StoreError('catalogue count disagrees with records')
    json_text(bundle)


class ScenarioStore:
    """Readers pin a sealed release; the only writer operation is import_revision.

    Read-only connections by default. BEGIN IMMEDIATE serializes writers; callers
    supply expected_parent to reject a stale agent's import after another commit.
    """

    @classmethod
    def create(cls, path):
        if sqlite3.sqlite_version_info < (3, 37, 0):
            raise StoreError('SQLite >=3.37 with JSON functions is required')
        _expected_schema_digest()  # Check runtime/schema before creating any file.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally replacing a user database.
        with path.open('xb'):
            pass
        connection = sqlite3.connect(path)
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            connection.executescript('BEGIN IMMEDIATE;\n' + _schema() + '\nCOMMIT;')
        finally:
            connection.close()

    def __init__(self, path, *, writable=False):
        self.path = Path(path).resolve()
        uri = self.path.as_uri() + ('?mode=rw' if writable else '?mode=ro')
        self.connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.writable = writable
        try:
            self.connection.execute('PRAGMA foreign_keys=ON')
            self.connection.execute('PRAGMA recursive_triggers=ON')
            self.connection.execute('PRAGMA busy_timeout=10000')
            if self.connection.execute('PRAGMA foreign_keys').fetchone()[0] != 1:
                raise StoreError('foreign keys unavailable')
            if self.connection.execute('PRAGMA user_version').fetchone()[0] != 1:
                raise StoreError('unsupported database schema version')
            if _schema_digest(self.connection) != _expected_schema_digest():
                raise StoreError('database schema/guards differ from version 1')
        except Exception:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    @contextmanager
    def _snapshot(self):
        """Multi-query reads see one SQLite snapshot, including while writers commit."""
        own = not self.connection.in_transaction
        if own:
            self.connection.execute('BEGIN')
        try:
            yield
        finally:
            if own:
                self.connection.rollback()

    def head(self):
        row = self.connection.execute('SELECT release_id FROM update_log ORDER BY event_id DESC LIMIT 1').fetchone()
        return row[0] if row else None

    def _release(self, release_id, *, staging=False):
        row = self.connection.execute('SELECT * FROM releases WHERE release_id=?', (release_id,)).fetchone()
        if row is None or (not staging and row['state'] != 'sealed'):
            raise StoreError(f'unknown sealed release: {release_id}')
        return row

    def releases(self):
        return [dict(r) for r in self.connection.execute(
            'SELECT event_id,release_id,parent_id,created_at,actor,reason,source_sha256,content_sha256,counts_json FROM update_log ORDER BY event_id')]

    def import_revision(self, source, release_id, *, expected_parent, actor, reason):
        if not self.writable:
            raise StoreError('open a writable store to import')
        if not release_id or release_id.strip() != release_id or not actor.strip() or not reason.strip():
            raise StoreError('release id, actor and reason are required')
        bundle, source_hash = read_bundle(source)
        validate_bundle(bundle)
        digest = content_hash(bundle)
        con = self.connection
        con.execute('BEGIN IMMEDIATE')
        try:
            existing = con.execute('SELECT * FROM releases WHERE release_id=?', (release_id,)).fetchone()
            if existing:
                if existing['state'] != 'sealed' or existing['source_sha256'] != source_hash or existing['content_sha256'] != digest:
                    raise StoreError('release name already identifies different content')
                if existing['parent_id'] != expected_parent:
                    raise StoreError('idempotent retry has different parent')
                if content_hash(self.export_bundle(release_id)) != digest:
                    raise StoreError('existing release failed content verification')
                con.rollback()
                return {'release_id': release_id, 'status': 'already_present', 'content_sha256': digest}
            if self.head() != expected_parent:
                raise StoreError('stale expected_parent; inspect the update log and retry explicitly')
            if bundle.get('base_dataset_sha256'):
                # v2 declares v1 as its generator ancestor, not necessarily its immediate revision parent.
                if not con.execute('SELECT 1 FROM releases WHERE source_sha256=?', (bundle['base_dataset_sha256'],)).fetchone():
                    raise StoreError('declared base dataset has not been imported')
            con.execute('INSERT INTO releases VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (release_id, expected_parent, source_hash, digest, str(source),
                         datetime.now(timezone.utc).isoformat(), actor, reason,
                         json_text(_rest(bundle, TOP_FIELDS)), 'staging'))
            for ordinal, family in enumerate(bundle['families']):
                fid = family['family_id']
                con.execute('INSERT INTO families VALUES(?,?,?,?,?)',
                            (release_id, fid, ordinal, family['split'],
                             json_text(_rest(family, {'family_id', 'split', 'sessions'}))))
                for order, session in enumerate(family['sessions']):
                    keys = {'session_id', 'person_id', 'device_id', 'physical_vehicle_id'}
                    con.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)',
                                (release_id, session['session_id'], fid, order,
                                 session['person_id'], session['device_id'], session['physical_vehicle_id'],
                                 json_text(_rest(session, keys))))
            for sid, trace in bundle['traces'].items():
                con.executemany('INSERT INTO points VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                ((release_id, sid, i, *(p[k] for k in POINT_FIELDS)) for i, p in enumerate(trace)))
            for ordinal, case in enumerate(bundle['catalogue']):
                con.execute('INSERT INTO cases VALUES(?,?,?,?,?)',
                            (release_id, case['case_id'], case['scenario'], ordinal,
                             json_text(_rest(case, {'case_id', 'scenario'}))))
            for ordinal, record in enumerate(bundle['records']):
                keys = {'record_id', 'case_id', 'scenario', 'family_id', 'split', 'session_ids', 'observed_indices'}
                rid, fid = record['record_id'], record['family_id']
                con.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',
                            (release_id, rid, ordinal, record['case_id'], record['scenario'], fid, record['split'],
                             json_text(_rest(record, keys))))
                for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                    con.execute('INSERT INTO record_sessions VALUES(?,?,?,?,?)', (release_id, rid, slot, sid, fid))
                    con.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?)',
                                    ((release_id, rid, slot, i, sid, index) for i, index in enumerate(indices)))
            con.executemany('INSERT INTO source_hashes VALUES(?,?,?)',
                            ((release_id, path, digest) for path, digest in bundle['source_sha256'].items()))
            rebuilt = self.export_bundle(release_id, _staging=True)
            if content_hash(rebuilt) != digest:
                raise StoreError('lossless import/export check failed')
            if con.execute('PRAGMA foreign_key_check').fetchall():
                raise StoreError('foreign key check failed')
            con.execute("UPDATE releases SET state='sealed' WHERE release_id=?", (release_id,))
            con.commit()
        except Exception:
            con.rollback()
            raise
        return {'release_id': release_id, 'status': 'imported', 'content_sha256': digest}

    def export_bundle(self, release_id, *, _staging=False):
        """Lossless semantic JSON snapshot; original byte hash remains separately pinned."""
        with self._snapshot():
            release = self._release(release_id, staging=_staging)
            result = json.loads(release['metadata_json'])
            result['families'], result['traces'], result['records'], result['catalogue'] = [], {}, [], []
            for row in self.connection.execute('SELECT * FROM families WHERE release_id=? ORDER BY ordinal', (release_id,)):
                family = dict(json.loads(row['metadata_json']), family_id=row['family_id'], split=row['split'], sessions=[])
                for s in self.connection.execute('SELECT * FROM sessions WHERE release_id=? AND family_id=? ORDER BY ordinal', (release_id, row['family_id'])):
                    family['sessions'].append(dict(json.loads(s['metadata_json']), **{k: s[k] for k in ('session_id', 'person_id', 'device_id', 'physical_vehicle_id')}))
                    result['traces'][s['session_id']] = []
                result['families'].append(family)
            for p in self.connection.execute('SELECT * FROM points WHERE release_id=? ORDER BY session_id,point_index', (release_id,)):
                result['traces'][p['session_id']].append({k: p[k] for k in POINT_FIELDS})
            for c in self.connection.execute('SELECT * FROM cases WHERE release_id=? ORDER BY ordinal', (release_id,)):
                result['catalogue'].append(dict(json.loads(c['metadata_json']), case_id=c['case_id'], scenario=c['scenario']))
            result['records'] = [self._record(release_id, r[0]) for r in self.connection.execute(
                'SELECT record_id FROM records WHERE release_id=? ORDER BY ordinal', (release_id,))]
            result['source_sha256'] = dict(self.connection.execute('SELECT path,sha256 FROM source_hashes WHERE release_id=?', (release_id,)))
            return result

    def _record(self, release_id, record_id):
        row = self.connection.execute('SELECT * FROM records WHERE release_id=? AND record_id=?', (release_id, record_id)).fetchone()
        if row is None:
            raise StoreError(f'unknown record: {record_id}')
        record = dict(json.loads(row['metadata_json']), **{k: row[k] for k in ('record_id', 'case_id', 'scenario', 'family_id', 'split')})
        record['session_ids'], record['observed_indices'] = [], []
        for s in self.connection.execute('SELECT * FROM record_sessions WHERE release_id=? AND record_id=? ORDER BY slot', (release_id, record_id)):
            record['session_ids'].append(s['session_id'])
            record['observed_indices'].append([p[0] for p in self.connection.execute(
                'SELECT point_index FROM observations WHERE release_id=? AND record_id=? AND slot=? ORDER BY ordinal',
                (release_id, record_id, s['slot']))])
        return record

    def device_view(self, release_id, record_id, *, slot=0):
        """Bounded PRIVATE device input, not attacker output. Query only allowed rows.

        No full traces, future labels, global clock, lanes or identities returned.
        S8 uses a shared epoch; S6.C caller selects its explicit target slot.
        """
        with self._snapshot():
            self._release(release_id)
            record = self._record(release_id, record_id)
            if type(slot) is not int or not 0 <= slot < len(record['session_ids']):
                raise StoreError('invalid observation slot')
            rows = list(self.connection.execute('''SELECT o.slot,o.ordinal,p.time_s,p.lat,p.lon
                FROM observations o JOIN points p USING(release_id,session_id,point_index)
                WHERE o.release_id=? AND o.record_id=? ORDER BY o.slot,o.ordinal''', (release_id, record_id)))
            chosen = [p for p in rows if p['slot'] == slot]
            epoch = (min(p['time_s'] for p in rows if p['ordinal'] == 0)
                     if record['observation_policy']['clock'] == 'common_pair_epoch' else chosen[0]['time_s'])
            queries = record['labels'].get('true_queries', [])
            return [{'time_s': p['time_s']-epoch, 'lat': p['lat'], 'lon': p['lon'],
                     'query_category': queries[i] if i < len(queries) else 'cafe'}
                    for i, p in enumerate(chosen)]

    def coverage(self, release_id):
        self._release(release_id)
        return [dict(row) for row in self.connection.execute(
            'SELECT case_id,split,records,families FROM case_coverage WHERE release_id=? ORDER BY case_id,split', (release_id,))]

    def verify(self):
        """Reconstruct every revision and recheck its hash, counts and audit lineage."""
        with self._snapshot():
            con = self.connection
            if _schema_digest(con) != _expected_schema_digest():
                raise StoreError('database schema/guards changed')
            if [r[0] for r in con.execute('PRAGMA integrity_check')] != ['ok'] or con.execute('PRAGMA foreign_key_check').fetchall():
                raise StoreError('SQLite integrity check failed')
            logs = self.releases()
            if con.execute('SELECT count(*) FROM releases').fetchone()[0] != len(logs):
                raise StoreError('unsealed or unlogged release')
            previous, checks = None, []
            for event in logs:
                release = self._release(event['release_id'])
                if event['parent_id'] != previous:
                    raise StoreError('broken revision chain')
                for key in ('parent_id', 'created_at', 'actor', 'reason', 'source_sha256', 'content_sha256'):
                    if event[key] != release[key]:
                        raise StoreError('log disagrees with release')
                bundle = self.export_bundle(event['release_id'])
                validate_bundle(bundle)
                if content_hash(bundle) != release['content_sha256']:
                    raise StoreError('content hash mismatch')
                for table, group, ordinal in (
                    ('families', 'release_id', 'ordinal'), ('sessions', 'family_id', 'ordinal'),
                    ('points', 'session_id', 'point_index'), ('cases', 'release_id', 'ordinal'),
                    ('records', 'release_id', 'ordinal'), ('record_sessions', 'record_id', 'slot'),
                    ('observations', 'record_id,slot', 'ordinal')):
                    if con.execute(f'''SELECT {group} FROM {table} WHERE release_id=?
                        GROUP BY {group} HAVING min({ordinal})!=0 OR max({ordinal})+1!=count(*)''',
                                   (event['release_id'],)).fetchall():
                        raise StoreError(f'non-contiguous ordinal in {table}')
                counts = {t: con.execute(f'SELECT count(*) FROM {t} WHERE release_id=?', (event['release_id'],)).fetchone()[0]
                          for t in ('families', 'sessions', 'points', 'records', 'observations')}
                if counts != json.loads(event['counts_json']):
                    raise StoreError('log counts disagree with data')
                checks.append({'release_id': event['release_id'], 'content_sha256': release['content_sha256'], **counts})
                previous = event['release_id']
            return {'status': 'passed', 'schema_version': 1, 'head': previous, 'releases': checks,
                    'scope': 'storage integrity, not native SUMO or attack/protection validity'}
