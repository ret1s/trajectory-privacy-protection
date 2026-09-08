-- Local evaluator-only scenario registry. Migrations create a new schema version.
CREATE TABLE releases (
    release_id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES releases(release_id),
    source_sha256 TEXT NOT NULL UNIQUE CHECK(length(source_sha256)=64),
    content_sha256 TEXT NOT NULL UNIQUE CHECK(length(content_sha256)=64),
    source_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    actor TEXT NOT NULL CHECK(length(trim(actor))>0),
    reason TEXT NOT NULL CHECK(length(trim(reason))>0),
    metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
    state TEXT NOT NULL CHECK(state IN ('staging','sealed'))
) STRICT;

CREATE TABLE families (
    release_id TEXT NOT NULL REFERENCES releases(release_id),
    family_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal>=0),
    split TEXT NOT NULL CHECK(split IN ('development_train','development_validation','development_test','confirmation')),
    metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
    PRIMARY KEY(release_id,family_id), UNIQUE(release_id,ordinal),
    UNIQUE(release_id,family_id,split)
) STRICT;

CREATE TABLE sessions (
    release_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    family_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal>=0),
    person_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    physical_vehicle_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
    PRIMARY KEY(release_id,session_id), UNIQUE(release_id,family_id,ordinal),
    UNIQUE(release_id,session_id,family_id),
    FOREIGN KEY(release_id,family_id) REFERENCES families(release_id,family_id)
) STRICT;
CREATE INDEX session_person ON sessions(release_id,person_id);
CREATE INDEX session_device ON sessions(release_id,device_id);
CREATE INDEX session_vehicle ON sessions(release_id,physical_vehicle_id);

-- ANY + explicit numeric checks preserve JSON integer/float types losslessly.
CREATE TABLE points (
    release_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    point_index INTEGER NOT NULL CHECK(point_index>=0),
    time_s ANY NOT NULL CHECK(typeof(time_s) IN ('integer','real') AND time_s>=0 AND time_s<1e12),
    lat ANY NOT NULL CHECK(typeof(lat) IN ('integer','real') AND lat BETWEEN -90 AND 90),
    lon ANY NOT NULL CHECK(typeof(lon) IN ('integer','real') AND lon BETWEEN -180 AND 180),
    speed_m_s ANY NOT NULL CHECK(typeof(speed_m_s) IN ('integer','real') AND speed_m_s>=0 AND speed_m_s<1e6),
    lane_id TEXT NOT NULL CHECK(length(lane_id)>0),
    edge_id TEXT NOT NULL CHECK(length(edge_id)>0),
    lane_pos_m ANY NOT NULL CHECK(typeof(lane_pos_m) IN ('integer','real') AND lane_pos_m>=0 AND lane_pos_m<1e9),
    angle_deg ANY NOT NULL CHECK(typeof(angle_deg) IN ('integer','real') AND angle_deg BETWEEN 0 AND 360),
    PRIMARY KEY(release_id,session_id,point_index),
    UNIQUE(release_id,session_id,time_s),
    FOREIGN KEY(release_id,session_id) REFERENCES sessions(release_id,session_id)
) STRICT;

CREATE TABLE cases (
    release_id TEXT NOT NULL REFERENCES releases(release_id),
    case_id TEXT NOT NULL,
    scenario TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal>=0),
    metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
    PRIMARY KEY(release_id,case_id), UNIQUE(release_id,ordinal),
    UNIQUE(release_id,case_id,scenario)
) STRICT;

CREATE TABLE records (
    release_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal>=0),
    case_id TEXT NOT NULL,
    scenario TEXT NOT NULL,
    family_id TEXT NOT NULL,
    split TEXT NOT NULL,
    metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
    PRIMARY KEY(release_id,record_id), UNIQUE(release_id,ordinal),
    UNIQUE(release_id,record_id,family_id),
    FOREIGN KEY(release_id,case_id,scenario) REFERENCES cases(release_id,case_id,scenario),
    FOREIGN KEY(release_id,family_id,split) REFERENCES families(release_id,family_id,split)
) STRICT;
CREATE INDEX records_by_case_split ON records(release_id,case_id,split);

CREATE TABLE record_sessions (
    release_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    slot INTEGER NOT NULL CHECK(slot>=0),
    session_id TEXT NOT NULL,
    family_id TEXT NOT NULL,
    PRIMARY KEY(release_id,record_id,slot),
    UNIQUE(release_id,record_id,session_id),
    UNIQUE(release_id,record_id,slot,session_id),
    FOREIGN KEY(release_id,record_id,family_id) REFERENCES records(release_id,record_id,family_id),
    FOREIGN KEY(release_id,session_id,family_id) REFERENCES sessions(release_id,session_id,family_id)
) STRICT;

CREATE TABLE observations (
    release_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    slot INTEGER NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal>=0),
    session_id TEXT NOT NULL,
    point_index INTEGER NOT NULL,
    PRIMARY KEY(release_id,record_id,slot,ordinal),
    UNIQUE(release_id,record_id,slot,point_index),
    FOREIGN KEY(release_id,record_id,slot,session_id) REFERENCES record_sessions(release_id,record_id,slot,session_id),
    FOREIGN KEY(release_id,session_id,point_index) REFERENCES points(release_id,session_id,point_index)
) STRICT;

CREATE TABLE source_hashes (
    release_id TEXT NOT NULL REFERENCES releases(release_id),
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK(length(sha256)=64),
    PRIMARY KEY(release_id,path)
) STRICT;

CREATE TABLE update_log (
    event_id INTEGER PRIMARY KEY,
    release_id TEXT NOT NULL UNIQUE REFERENCES releases(release_id),
    parent_id TEXT REFERENCES releases(release_id),
    event_type TEXT NOT NULL CHECK(event_type='release_sealed'),
    created_at TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    counts_json TEXT NOT NULL CHECK(json_valid(counts_json))
) STRICT;

CREATE TRIGGER release_insert BEFORE INSERT ON releases BEGIN
    SELECT CASE WHEN NEW.state!='staging' THEN RAISE(ABORT,'insert staging release only') END;
    SELECT CASE WHEN NEW.parent_id IS NOT (SELECT release_id FROM update_log ORDER BY event_id DESC LIMIT 1)
        THEN RAISE(ABORT,'parent must be current head') END;
END;
CREATE TRIGGER release_update BEFORE UPDATE ON releases BEGIN
    SELECT CASE WHEN OLD.state!='staging' OR NEW.state!='sealed'
        OR NEW.release_id IS NOT OLD.release_id OR NEW.parent_id IS NOT OLD.parent_id
        OR NEW.source_sha256 IS NOT OLD.source_sha256 OR NEW.content_sha256 IS NOT OLD.content_sha256
        OR NEW.source_path IS NOT OLD.source_path OR NEW.created_at IS NOT OLD.created_at
        OR NEW.actor IS NOT OLD.actor OR NEW.reason IS NOT OLD.reason
        OR NEW.metadata_json IS NOT OLD.metadata_json
        THEN RAISE(ABORT,'release is immutable; create a revision') END;
END;
CREATE TRIGGER release_delete BEFORE DELETE ON releases BEGIN
    SELECT RAISE(ABORT,'release deletion forbidden');
END;
CREATE TRIGGER release_sealed AFTER UPDATE OF state ON releases WHEN NEW.state='sealed' BEGIN
    INSERT INTO update_log(release_id,parent_id,event_type,created_at,actor,reason,source_sha256,content_sha256,counts_json)
    VALUES(NEW.release_id,NEW.parent_id,'release_sealed',NEW.created_at,NEW.actor,NEW.reason,NEW.source_sha256,NEW.content_sha256,
        json_object('families',(SELECT count(*) FROM families WHERE release_id=NEW.release_id),
                    'sessions',(SELECT count(*) FROM sessions WHERE release_id=NEW.release_id),
                    'points',(SELECT count(*) FROM points WHERE release_id=NEW.release_id),
                    'records',(SELECT count(*) FROM records WHERE release_id=NEW.release_id),
                    'observations',(SELECT count(*) FROM observations WHERE release_id=NEW.release_id)));
END;
CREATE TRIGGER log_update BEFORE UPDATE ON update_log BEGIN
    SELECT RAISE(ABORT,'update log is append-only');
END;
CREATE TRIGGER log_delete BEFORE DELETE ON update_log BEGIN
    SELECT RAISE(ABORT,'update log is append-only');
END;

CREATE VIEW case_coverage AS
SELECT c.release_id,c.case_id,s.split,count(r.record_id) AS records,
       count(DISTINCT r.family_id) AS families
FROM cases c
JOIN (SELECT DISTINCT release_id,split FROM families) s ON s.release_id=c.release_id
LEFT JOIN records r ON r.release_id=c.release_id AND r.case_id=c.case_id AND r.split=s.split
GROUP BY c.release_id,c.case_id,s.split;
PRAGMA user_version=1;
