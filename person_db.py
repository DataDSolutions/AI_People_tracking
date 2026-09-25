import time
import sqlite3
import random
import string
import os
import threading
import numpy as np
from collections import defaultdict

SAME_CAM_RECENT_BOOST = float(os.environ.get("SAME_CAM_RECENT_BOOST", "0.05"))

CONTAM_MARGIN = float(os.environ.get("REID_CONTAM_MARGIN", "0.05"))

# Identities unseen for longer than this are NOT candidates for matching (0 =
# no limit). Keeps the DB match pool consistent with the MCMTT gallery's own
# retention: ReID embeddings are clothing-dominated, so week-old identities
# mostly add false candidates (and scan cost), not true re-identifications.
MATCH_MAX_AGE_SEC = float(os.environ.get("MATCH_MAX_AGE_SEC", "259200"))

DEFAULT_OBJECT_CLASS = "person"

DEFAULT_ABSENCE_SECONDS = 900.0


def _l2(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _ts_str(ts):
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return ""


def is_valid_uid(uid: str) -> bool:
    """Validate a 9-char Crockford UID by checking the
    computed check digit against the stored 9th character.
    Accepts UIDs with or without display separators (hyphens).
    Returns False for any 8-char legacy UID."""
    uid = uid.replace("-", "")
    if len(uid) != 9:
        return False
    _ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"
    expected = _ALPHABET[
        sum(ord(c) * (i + 1) for i, c in enumerate(uid[:8]))
        % len(_ALPHABET)
    ]
    return uid[8] == expected


class PersonDatabase:
    def __init__(self, db_path="reid_persons.db",
                 max_embeddings_per_person=60,
                 feature_size=256,
                 commit_interval=5.0,
                 seed_target=25,
                 overlap_groups=None):
        self.db_path = db_path
        self.max_emb = max_embeddings_per_person
        self.feat_sz = feature_size
        self.commit_interval = commit_interval
        self.seed_target = seed_target

        self.overlap_groups = [set(g) for g in (overlap_groups or [])]

        self._emb_cache = defaultdict(list)
        self._mat_cache = {}
        self._meta_cache = {}

        self._fixture_protos = []
        self._fixture_reject_thr = float(
            os.environ.get("FIXTURE_REJECT_THR", "0.83"))
        self._fixture_merge_thr = 0.94
        self._fixture_ttl = float(
            os.environ.get("FIXTURE_PROTO_TTL", "600"))
        self._fixture_max = 256

        self._dirty = False
        self._last_commit = time.time()

        # Hot-path write throttling: in-memory state is still updated every
        # call, but touch() only writes to SQLite once per flush interval per
        # (pid, cam). Set DB_TOUCH_FLUSH_SEC=0 to restore per-call writes.
        self._lock = threading.RLock()
        self._touch_flush_every = float(os.environ.get("DB_TOUCH_FLUSH_SEC", "1.0"))
        self._touch_flush_ts = {}    # (pid, cam) -> last DB write time
        self._touch_pending = {}     # (pid, cam) -> frames since last DB write

        # Cannot-link registry: pairs of UIDs observed as two DIFFERENT people at
        # the same instant (simultaneously visible, distinct boxes). Such a pair
        # is never merged. {(a, b) with a < b: first_ts}
        self._cannot = {}
        self._cannot_by_uid = defaultdict(set)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        # WAL + NORMAL: consistent on power loss; only the last few seconds of
        # deferred (non-identity-defining) writes can be lost.
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_tables()
        self._load_into_memory()

        self._open_incidents = {}
        self._load_open_incidents()

    def _create_tables(self):
        c = self._conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS uid_ledger (
                person_id   TEXT PRIMARY KEY,
                minted_at   REAL,
                minted_str  TEXT,
                retired_at  REAL,      -- NULL while live
                retired_why TEXT       -- 'merged'|'pruned'|NULL
            )""")

        c.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id        TEXT PRIMARY KEY,  -- global UID
                object_class     TEXT,               -- identity partition
                                                       -- (e.g. "person"); never
                                                       -- mixed across classes
                created_at       REAL,               -- Unix ts, first seen
                last_seen        REAL,               -- Unix ts, most recent
                sighting_count   INTEGER,            -- frames tracked (not visits)
                first_seen_str   TEXT,               -- readable first-seen datetime
                last_seen_str    TEXT,               -- readable last-seen datetime
                last_camera      TEXT,               -- camera of most recent sighting
                num_embeddings   INTEGER             -- vectors stored for this person
            )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  TEXT,
                object_class TEXT,                   -- redundant w/ persons.object_class,
                                                       -- kept per-row so a mismatch is
                                                       -- detectable/auditable
                vec        BLOB,
                created_at REAL,
                created_str TEXT,                    -- readable datetime
                camera     TEXT,                     -- camera this vec came from
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_emb_person ON embeddings(person_id)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                person_id   TEXT,
                camera      TEXT,
                first_seen  REAL,
                last_seen   REAL,
                first_str   TEXT,
                last_str    TEXT,
                count       INTEGER,
                PRIMARY KEY (person_id, camera)
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sight_person ON sightings(person_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sight_camera ON sightings(camera)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id         TEXT,
                object_class      TEXT,
                camera_id         TEXT,
                entry_time        REAL,
                last_detected_time REAL,
                exit_time         REAL,             -- NULL while open
                dwell_seconds     REAL,             -- NULL while open
                detection_count   INTEGER,
                movement_status   TEXT,             -- 'stationary'|'moving'|'unknown'
                status            TEXT,             -- DETECTED|ACTIVE|POSSIBLY_EXITED|CLOSED
                created_at        REAL,
                updated_at        REAL,
                FOREIGN KEY(person_id) REFERENCES persons(person_id)
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_inc_person ON incidents(person_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_inc_status ON incidents(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_inc_camera ON incidents(camera_id)")

        self._conn.commit()
        self._migrate_add_columns()

    def _migrate_add_columns(self):
        c = self._conn.cursor()
        existing = {row[1] for row in c.execute("PRAGMA table_info(persons)")}
        for col, decl in (("first_seen_str", "TEXT"), ("last_seen_str", "TEXT"),
                          ("last_camera", "TEXT"), ("num_embeddings", "INTEGER"),
                          ("object_class", "TEXT")):
            if col not in existing:
                c.execute(f"ALTER TABLE persons ADD COLUMN {col} {decl}")

        eexisting = {row[1] for row in c.execute("PRAGMA table_info(embeddings)")}
        for col, decl in (("created_str", "TEXT"), ("camera", "TEXT"),
                          ("object_class", "TEXT")):
            if col not in eexisting:
                c.execute(f"ALTER TABLE embeddings ADD COLUMN {col} {decl}")

        c.execute("UPDATE persons SET object_class=? WHERE object_class IS NULL",
                   (DEFAULT_OBJECT_CLASS,))
        c.execute("UPDATE embeddings SET object_class=? WHERE object_class IS NULL",
                   (DEFAULT_OBJECT_CLASS,))

        c.execute("CREATE INDEX IF NOT EXISTS idx_emb_class ON embeddings(object_class)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS cannot_link (
                a  TEXT NOT NULL,
                b  TEXT NOT NULL,
                ts REAL,
                PRIMARY KEY (a, b)
            )""")

        for pid, ca, ls in c.execute(
                "SELECT person_id, created_at, last_seen FROM persons").fetchall():
            c.execute("UPDATE persons SET first_seen_str=?, last_seen_str=? "
                      "WHERE person_id=?",
                      (_ts_str(ca), _ts_str(ls), pid))
        self._conn.commit()

    def _load_into_memory(self):
        c = self._conn.cursor()

        c.execute(
            "INSERT OR IGNORE INTO uid_ledger(person_id, minted_at, minted_str) "
            "SELECT person_id, created_at, first_seen_str FROM persons")
        self._uid_ever = {r[0] for r in c.execute(
            "SELECT person_id FROM uid_ledger")}

        for pid, last_seen, count, last_cam, created_at, first_str, obj_class in c.execute(
                "SELECT person_id, last_seen, sighting_count, last_camera, "
                "created_at, first_seen_str, object_class FROM persons"):
            self._meta_cache[pid] = {
                "last_seen": last_seen, "count": count,
                "last_cam": last_cam,
                "cams": set(),
                "first_seen": created_at,
                "first_seen_str": first_str or _ts_str(created_at),
                "class": obj_class or DEFAULT_OBJECT_CLASS,
            }

        loaded = 0
        for pid, blob, camera in c.execute(
                "SELECT person_id, vec, camera FROM embeddings"):
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] == self.feat_sz:
                self._emb_cache[pid].append(_l2(vec))
                loaded += 1
                if pid in self._meta_cache and camera:
                    self._meta_cache[pid]["cams"].add(camera)

        for pid in self._emb_cache:
            if len(self._emb_cache[pid]) > self.max_emb:
                self._emb_cache[pid] = self._emb_cache[pid][-self.max_emb:]
                self._mat_cache.pop(pid, None)

        try:
            _n_cl = 0
            for a, b, ts in c.execute("SELECT a, b, ts FROM cannot_link"):
                if a in self._meta_cache and b in self._meta_cache:
                    self._cannot[(a, b) if a < b else (b, a)] = ts
                    self._cannot_by_uid[a].add(b)
                    self._cannot_by_uid[b].add(a)
                    _n_cl += 1
            if _n_cl:
                print(f"[DB] Loaded {_n_cl} cannot-link pair(s)")
        except Exception as e:
            print(f"[DB] cannot_link load failed: {e}")

        print(f"[DB] Loaded {len(self._meta_cache)} persons, "
              f"{loaded} embeddings from {self.db_path}")

    def _load_open_incidents(self):
        rows = self._conn.execute(
            "SELECT incident_id, person_id, camera_id FROM incidents "
            "WHERE status IN ('DETECTED','ACTIVE','POSSIBLY_EXITED')").fetchall()
        for incident_id, pid, cam in rows:
            self._open_incidents[(pid, cam)] = incident_id
        if rows:
            print(f"[DB] Resumed {len(rows)} open incident(s) from {self.db_path}")

    def _new_id(self) -> str:
        # Crockford Base32 — removes ambiguous characters:
        # 0/O, 1/I/L, U. DO NOT MODIFY THIS ALPHABET.
        _ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"
        _LENGTH = 8
        while True:
            base = "".join(random.choices(_ALPHABET, k=_LENGTH))
            check = _ALPHABET[
                sum(ord(c) * (i + 1) for i, c in enumerate(base))
                % len(_ALPHABET)
            ]
            pid = base + check
            if pid in self._meta_cache or pid in self._uid_ever:
                continue
            now = time.time()
            try:
                self._conn.execute(
                    "INSERT INTO uid_ledger(person_id, minted_at, "
                    "minted_str) VALUES (?,?,?)",
                    (pid, now, _ts_str(now)))
            except sqlite3.IntegrityError:
                self._uid_ever.add(pid)
                continue
            self._uid_ever.add(pid)
            return pid

    def _retire_uid(self, pid, why):
        """Mark an id as no longer live. It stays in the ledger forever so it
        can never be reissued; this only records why and when it went away."""
        try:
            self._conn.execute(
                "UPDATE uid_ledger SET retired_at=?, retired_why=? "
                "WHERE person_id=? AND retired_at IS NULL",
                (time.time(), why, pid))
        except Exception:
            pass

    @staticmethod
    def _cosine(a, b):
        a = _l2(a)
        b = _l2(b)
        return float(np.dot(a, b))

    def _cams_overlap(self, cam_a, cam_b):
        """True if two cameras share a physical space (same overlap group), so
        the same person may legitimately appear on both at the same instant."""
        if cam_a == cam_b:
            return True
        if not cam_a or not cam_b:
            return False
        for g in self.overlap_groups:
            if cam_a in g and cam_b in g:
                return True
        return False

    def _is_effectively_same_cam(self, cam, cams, last_cam):
        """A candidate is 'effectively same-camera' (NOT a cross-camera match)
        if THIS camera, or any overlapping camera, is among the cameras it was
        seen on. Overlapping cameras view the same room, so a person there is
        not physically 'somewhere else'."""
        if not cam:
            return False
        if cam in cams or self._cams_overlap(cam, last_cam):
            return True
        for c in cams:
            if self._cams_overlap(cam, c):
                return True
        return False

    def class_of(self, pid):
        """The identity_class this global UID belongs to, or None if unknown."""
        m = self._meta_cache.get(pid)
        return m.get("class") if m else None

    def match(self, vec, cam=None, object_class=DEFAULT_OBJECT_CLASS,
              threshold=0.55, soft_threshold=0.48,
              cross_camera_threshold=0.45,
              soft_window=120.0, margin=0.04, topk=3,
              exclude=None,
              posture_threshold=None, posture_window=12.0,
              cross_camera_floor=None, cross_margin=0.12,
              require_absent_seconds=None, max_age=None):

        vec = _l2(vec)
        now = time.time()
        exclude = exclude or set()
        _age_cut = MATCH_MAX_AGE_SEC if max_age is None else float(max_age)

        scored = []
        for pid, embs in self._emb_cache.items():
            if not embs or pid in exclude:
                continue
            meta = self._meta_cache.get(pid, {})
            if meta.get("class", DEFAULT_OBJECT_CLASS) != object_class:
                continue
            if _age_cut and (now - meta.get("last_seen", 0)) > _age_cut:
                continue
            if require_absent_seconds is not None:
                if (now - meta.get("last_seen", 0)) < require_absent_seconds:
                    continue

            mat = self._mat_cache.get(pid)
            if mat is None or mat.shape[0] != len(embs):
                mat = np.asarray(embs, dtype=np.float32)
                self._mat_cache[pid] = mat
            sims = mat @ vec
            k = min(topk, sims.shape[0])
            top_mean = float(np.mean(np.sort(sims)[-k:]))
            recent = (now - meta.get("last_seen", 0)) < soft_window
            cams = meta.get("cams") or set()
            last_cam = meta.get("last_cam")
            eff_same = self._is_effectively_same_cam(cam, cams, last_cam)
            is_cross = bool(cam) and not eff_same
            if recent and not is_cross:
                score = min(top_mean + SAME_CAM_RECENT_BOOST, 1.0)
            else:
                score = top_mean
            _very_recent_here = (bool(cam)
                                  and (now - meta.get("last_seen", 0)) < posture_window
                                  and eff_same)
            scored.append((score, pid, recent, is_cross, _very_recent_here))

        if not scored:
            return None, 0.0

        scored.sort(reverse=True)
        best_score, best_pid, best_recent, best_cross, best_very_recent = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        bar = cross_camera_threshold if best_cross else threshold

        if (posture_threshold is not None
                and best_very_recent
                and (best_score - second_score) >= margin):
            if best_score >= posture_threshold:
                return best_pid, best_score

        confident = best_score >= (bar + 0.10)
        if confident:
            return best_pid, best_score

        if cross_camera_floor is not None:
            MIN_REAL_RUNNERUP = min(soft_threshold, cross_camera_floor - 0.10)
            has_real_runnerup = (len(scored) >= 2 and second_score >= MIN_REAL_RUNNERUP)
            if (best_cross
                    and has_real_runnerup
                    and best_score >= cross_camera_floor
                    and (best_score - second_score) >= cross_margin):
                return best_pid, best_score

        second_qualifies = second_score >= bar
        if (best_score - second_score) < margin and second_qualifies:
            return None, best_score

        soft_ok = (best_score >= soft_threshold and best_recent
                   and not best_cross)
        if best_score >= bar or soft_ok:
            return best_pid, best_score

        return None, best_score

    def top_candidates(self, vec, cam=None, object_class=DEFAULT_OBJECT_CLASS,
                       k=3, max_age=None, topk=3):
        """Diagnostics only (never affects a decision): the k best-scoring
        candidate identities for `vec`, as (raw_top_mean_similarity, pid,
        is_cross_camera, age_seconds). Uses the same scoring and the same
        age-limited pool as match(), but without boost/exclusion, so it shows
        WHY a new UID was minted."""
        vec = _l2(vec)
        now = time.time()
        cut = MATCH_MAX_AGE_SEC if max_age is None else float(max_age)
        out = []
        for pid, embs in self._emb_cache.items():
            if not embs:
                continue
            meta = self._meta_cache.get(pid, {})
            if meta.get("class", DEFAULT_OBJECT_CLASS) != object_class:
                continue
            age = now - meta.get("last_seen", 0)
            if cut and age > cut:
                continue
            mat = self._mat_cache.get(pid)
            if mat is None or mat.shape[0] != len(embs):
                mat = np.asarray(embs, dtype=np.float32)
                self._mat_cache[pid] = mat
            sims = mat @ vec
            kk = min(topk, sims.shape[0])
            top_mean = float(np.mean(np.sort(sims)[-kk:]))
            cams = meta.get("cams") or set()
            is_cross = bool(cam) and not self._is_effectively_same_cam(
                cam, cams, meta.get("last_cam"))
            out.append((top_mean, pid, is_cross, age))
        out.sort(reverse=True)
        return out[:k]

    def restore_candidates(self, vec, object_class=DEFAULT_OBJECT_CLASS,
                            cam=None, min_absence_seconds=DEFAULT_ABSENCE_SECONDS,
                            threshold=0.55, cross_camera_threshold=0.45,
                            **kwargs):
        """15-MINUTE ABSENCE RULE. Look for a Global UID of this object_class
        that has NOT been seen on ANY camera for at least min_absence_seconds,
        and try to re-identify `vec` against it.

        This is deliberately just `match()` scoped with require_absent_seconds:
        one scoring codepath means the confidence bars, contamination
        resistance, and margin logic are IDENTICAL to live matching -- the
        absence rule never gets a looser bar than a fresh commit would. Per
        "never force a match", if nothing clears the same bar an ordinary
        match would need, this returns (None, best_score) and the caller
        should keep the object on its temporary UID rather than guess.

        Returns (person_id_or_None, score).
        """
        return self.match(
            vec, cam=cam, object_class=object_class,
            threshold=threshold, cross_camera_threshold=cross_camera_threshold,
            require_absent_seconds=min_absence_seconds, **kwargs)

    def _record_sighting(self, pid, cam, n=1):
        if not cam:
            return
        now = time.time()
        nowstr = _ts_str(now)
        cur = self._conn.execute(
            "SELECT count FROM sightings WHERE person_id=? AND camera=?",
            (pid, cam)).fetchone()
        if cur is None:
            self._conn.execute(
                "INSERT INTO sightings(person_id, camera, first_seen, last_seen, "
                "first_str, last_str, count) VALUES (?,?,?,?,?,?,?)",
                (pid, cam, now, now, nowstr, nowstr, n))
            print(f"[SIGHTING] {pid} first seen on {cam}")
        else:
            self._conn.execute(
                "UPDATE sightings SET last_seen=?, last_str=?, count=count+? "
                "WHERE person_id=? AND camera=?", (now, nowstr, n, pid, cam))

    def create_person(self, vec, cam=None, object_class=DEFAULT_OBJECT_CLASS) -> str:
        vec = _l2(vec)
        pid = self._new_id()
        now = time.time()
        nowstr = _ts_str(now)
        self._conn.execute(
            "INSERT INTO persons (person_id, object_class, created_at, last_seen, "
            "sighting_count, first_seen_str, last_seen_str, last_camera, "
            "num_embeddings) VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, object_class, now, now, 1, nowstr, nowstr, cam, 1))
        self._conn.execute(
            "INSERT INTO embeddings(person_id, object_class, vec, created_at, "
            "created_str, camera) VALUES (?,?,?,?,?,?)",
            (pid, object_class, vec.astype(np.float32).tobytes(), now, nowstr, cam))
        self._conn.commit()

        self._emb_cache[pid].append(vec)
        self._mat_cache.pop(pid, None)

        self._meta_cache[pid] = {
            "last_seen": now, "count": 1, "last_cam": cam,
            "cams": {cam} if cam else set(),
            "first_seen": now, "first_seen_str": nowstr,
            "class": object_class,
        }
        self._record_sighting(pid, cam)
        print(f"[DB] NEW {object_class} {pid}  (total {len(self._meta_cache)})")
        return pid

    def create_person_reidless(self, cam=None, object_class=DEFAULT_OBJECT_CLASS) -> str:
        """Create a global identity with NO embedding yet (Tier-1 local commit).
        Used when a track is confirmed but the ReID engine hasn't produced a
        vector. Embeddings are added later via add_embedding as they arrive."""
        pid = self._new_id()
        now = time.time()
        nowstr = _ts_str(now)
        self._conn.execute(
            "INSERT INTO persons (person_id, object_class, created_at, last_seen, "
            "sighting_count, first_seen_str, last_seen_str, last_camera, "
            "num_embeddings) VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, object_class, now, now, 1, nowstr, nowstr, cam, 0))
        self._conn.commit()
        self._emb_cache[pid] = []
        self._meta_cache[pid] = {
            "last_seen": now, "count": 1, "last_cam": cam,
            "cams": {cam} if cam else set(),
            "first_seen": now, "first_seen_str": nowstr,
            "class": object_class,
        }
        self._record_sighting(pid, cam)
        print(f"[DB] NEW {object_class} {pid} (no-reid, total {len(self._meta_cache)})")
        return pid

    def merge_persons(self, keep_pid, drop_pid):
        """Merge drop_pid INTO keep_pid: move embeddings & sightings, keep the
        earliest first-seen. Returns True on success. Used for cross-camera
        hand-off when a post-commit ReID match reveals two UIDs are one
        object.

        Refuses (returns False) if the two identities belong to different
        object classes -- a cross-class merge would be an identity-safety
        violation ("never mix, merge, or associate embeddings from different
        object classes") no matter how it was triggered upstream."""
        if keep_pid == drop_pid:
            return False
        if keep_pid not in self._meta_cache or drop_pid not in self._meta_cache:
            return False
        km = self._meta_cache[keep_pid]
        dm = self._meta_cache[drop_pid]

        keep_class = km.get("class", DEFAULT_OBJECT_CLASS)
        drop_class = dm.get("class", DEFAULT_OBJECT_CLASS)
        if keep_class != drop_class:
            print(f"[DB] REFUSED merge {drop_pid}({drop_class}) -> "
                  f"{keep_pid}({keep_class}): cross-class merge blocked")
            return False

        if self.is_cannot_link(keep_pid, drop_pid):
            print(f"[DB] REFUSED merge {drop_pid} -> {keep_pid}: cannot-link "
                  f"(seen simultaneously as two different people)")
            return False

        if dm.get("first_seen") and (not km.get("first_seen")
                                      or dm["first_seen"] < km["first_seen"]):
            km["first_seen"] = dm["first_seen"]
            km["first_seen_str"] = dm.get("first_seen_str") or _ts_str(dm["first_seen"])
            self._conn.execute(
                "UPDATE persons SET created_at=?, first_seen_str=? WHERE person_id=?",
                (km["first_seen"], km["first_seen_str"], keep_pid))
        try:
            self._conn.execute("UPDATE embeddings SET person_id=? WHERE person_id=?",
                                (keep_pid, drop_pid))
            self._conn.execute(
                "UPDATE sightings AS k SET "
                "  count = k.count + ("
                "     SELECT d.count FROM sightings d "
                "     WHERE d.person_id=? AND d.camera=k.camera), "
                "  first_seen = MIN(k.first_seen, ("
                "     SELECT d.first_seen FROM sightings d "
                "     WHERE d.person_id=? AND d.camera=k.camera)), "
                "  last_seen = MAX(k.last_seen, ("
                "     SELECT d.last_seen FROM sightings d "
                "     WHERE d.person_id=? AND d.camera=k.camera)) "
                "WHERE k.person_id=? AND EXISTS ("
                "     SELECT 1 FROM sightings d "
                "     WHERE d.person_id=? AND d.camera=k.camera)",
                (drop_pid, drop_pid, drop_pid, keep_pid, drop_pid))
            self._conn.execute(
                "DELETE FROM sightings WHERE person_id=? AND camera IN ("
                "   SELECT camera FROM sightings WHERE person_id=?)",
                (drop_pid, keep_pid))
            self._conn.execute("UPDATE sightings SET person_id=? WHERE person_id=?",
                                (keep_pid, drop_pid))
            self._conn.execute(
                "UPDATE incidents SET person_id=? WHERE person_id=?",
                (keep_pid, drop_pid))
            for key in [k for k in self._open_incidents if k[0] == drop_pid]:
                incident_id = self._open_incidents.pop(key)
                self._open_incidents[(keep_pid, key[1])] = incident_id

            self._conn.execute("DELETE FROM persons WHERE person_id=?", (drop_pid,))
            self._retire_uid(drop_pid, "merged")
            self._conn.commit()
        except Exception as e:
            print(f"[DB] merge_persons({keep_pid},{drop_pid}) error: {e}")
            return False

        merged = self._emb_cache.get(keep_pid, []) + self._emb_cache.get(drop_pid, [])
        if len(merged) > self.max_emb:
            merged = merged[-self.max_emb:]
        self._emb_cache[keep_pid] = merged
        self._emb_cache.pop(drop_pid, None)
        self._mat_cache.pop(keep_pid, None)
        self._mat_cache.pop(drop_pid, None)
        km["cams"] = (km.get("cams") or set()) | (dm.get("cams") or set())
        km["count"] = km.get("count", 0) + dm.get("count", 0)
        km["last_seen"] = max(km.get("last_seen", 0), dm.get("last_seen", 0))
        self._meta_cache.pop(drop_pid, None)
        for _k in [k for k in self._touch_flush_ts if k[0] == drop_pid]:
            self._touch_flush_ts.pop(_k, None)
            self._touch_pending.pop(_k, None)
        try:                                    # cannot-links follow the surviving UID
            _others = list(self._cannot_by_uid.get(drop_pid, ()))
            self._purge_cannot_links(drop_pid)
            for _o in _others:
                if _o != keep_pid:
                    self.add_cannot_link(keep_pid, _o)
        except Exception as e:
            print(f"[DB] cannot-link transfer failed: {e}")
        self._conn.execute(
            "UPDATE persons SET num_embeddings=? WHERE person_id=?",
            (len(merged), keep_pid))
        self._conn.commit()
        print(f"[DB] MERGED {drop_pid} -> {keep_pid} "
              f"(now {len(merged)} embeddings, cams={sorted(km['cams'])})")
        return True

    def cams_of(self, pid):
        m = self._meta_cache.get(pid)
        return set(m.get("cams") or set()) if m else set()

    def recent_pids(self, max_age=None):
        """UIDs seen within max_age seconds (default MATCH_MAX_AGE_SEC; 0 = all).
        Same pool match() searches; callers that loop over identities use this
        instead of every UID ever minted."""
        cut = MATCH_MAX_AGE_SEC if max_age is None else float(max_age)
        if not cut:
            return list(self._meta_cache.keys())
        now = time.time()
        return [p for p, m in self._meta_cache.items()
                if (now - m.get("last_seen", 0)) <= cut]

    # ---- cannot-link registry ------------------------------------------
    def add_cannot_link(self, a, b, now=None):
        """Record that UIDs a and b were seen as two different people at the
        same instant. Returns True if newly recorded."""
        if a == b or a not in self._meta_cache or b not in self._meta_cache:
            return False
        key = (a, b) if a < b else (b, a)
        if key in self._cannot:
            return False
        ts = now if now is not None else time.time()
        self._cannot[key] = ts
        self._cannot_by_uid[a].add(b)
        self._cannot_by_uid[b].add(a)
        self._conn.execute(
            "INSERT OR IGNORE INTO cannot_link(a, b, ts) VALUES (?,?,?)",
            (key[0], key[1], ts))
        self._dirty = True
        return True

    def is_cannot_link(self, a, b):
        return ((a, b) if a < b else (b, a)) in self._cannot

    def cannot_links_of(self, pid):
        return set(self._cannot_by_uid.get(pid, ()))

    def _purge_cannot_links(self, pid):
        for o in list(self._cannot_by_uid.get(pid, ())):
            self._cannot.pop((pid, o) if pid < o else (o, pid), None)
            self._cannot_by_uid[o].discard(pid)
        self._cannot_by_uid.pop(pid, None)
        self._conn.execute("DELETE FROM cannot_link WHERE a=? OR b=?", (pid, pid))

    def add_embedding(self, pid, vec, cam=None, object_class=None,
                       outlier_floor=0.40, guard_contamination=True):
        """Append a new embedding to an EXISTING identity.

        object_class, if given, must match the identity's own stored class or
        the embedding is refused. This is a second, independent guard (on top
        of match()'s hard filter) against a caller accidentally attaching a
        vector from one object class onto another identity's gallery -- e.g.
        a bug upstream that resolves the wrong pid."""
        if pid not in self._meta_cache:
            return False
        stored_class = self._meta_cache[pid].get("class", DEFAULT_OBJECT_CLASS)
        if object_class is not None and object_class != stored_class:
            print(f"[DB] REFUSED add_embedding: {pid} is class={stored_class}, "
                  f"got object_class={object_class}")
            return False

        vec = _l2(vec)
        if vec.shape[0] != self.feat_sz or not np.all(np.isfinite(vec)):
            return False
        if float(np.linalg.norm(vec)) < 1e-6:
            return False
        embs = self._emb_cache[pid]
        if embs:
            avg = _l2(np.mean(np.asarray(embs, dtype=np.float32), axis=0))
            sim = self._cosine(vec, avg)

            redundant_bar = 0.985 if len(embs) < self.seed_target else 0.93
            if sim >= redundant_bar:
                return False
            if sim < outlier_floor:
                return False

            if guard_contamination:
                best_other = 0.0
                _ce_now = time.time()
                for opid, oembs in self._emb_cache.items():
                    if opid == pid or not oembs:
                        continue
                    ometa = self._meta_cache.get(opid, {})
                    if ometa.get("class", DEFAULT_OBJECT_CLASS) != stored_class:
                        continue
                    if MATCH_MAX_AGE_SEC and (_ce_now - ometa.get("last_seen", 0)) > MATCH_MAX_AGE_SEC:
                        continue
                    om = self._mat_cache.get(opid)
                    if om is None or om.shape[0] != len(oembs):
                        om = np.asarray(oembs, dtype=np.float32)
                        self._mat_cache[opid] = om
                    os_ = float(np.max(om @ vec))
                    if os_ > best_other:
                        best_other = os_
                if best_other > sim + CONTAM_MARGIN:
                    return False

        now = time.time()
        self._conn.execute(
            "INSERT INTO embeddings(person_id, object_class, vec, created_at, "
            "created_str, camera) VALUES (?,?,?,?,?,?)",
            (pid, stored_class, vec.astype(np.float32).tobytes(), now, _ts_str(now), cam))
        self._conn.execute("""
            DELETE FROM embeddings WHERE id IN (
                SELECT id FROM embeddings WHERE person_id=?
                ORDER BY created_at DESC LIMIT -1 OFFSET ?
            )""", (pid, self.max_emb))

        embs.append(vec)
        if len(embs) > self.max_emb:
            del embs[0]

        self._mat_cache.pop(pid, None)

        self._conn.execute(
            "UPDATE persons SET num_embeddings=? WHERE person_id=?",
            (len(embs), pid))
        self._dirty = True   # commit deferred to maybe_commit()
        return True

    def learn_fixture(self, vec, cam=None, object_class=DEFAULT_OBJECT_CLASS):
        """Record the appearance of a confirmed non-object (empty chair / bike
        row) for this identity class. Near-duplicate prototypes are folded
        together with a running average so the gallery stays small and each
        real fixture is one entry."""
        if vec is None:
            return
        vec = _l2(np.asarray(vec, dtype=np.float32))
        if vec.shape[0] != self.feat_sz or not np.all(np.isfinite(vec)):
            return
        now = time.time()
        for p in self._fixture_protos:
            if (p["cam"] == cam and p.get("class", DEFAULT_OBJECT_CLASS) == object_class
                    and self._cosine(vec, p["vec"]) >= self._fixture_merge_thr):
                n = p["n"]
                p["vec"] = _l2((p["vec"] * n + vec) / (n + 1))
                p["n"] = n + 1
                p["ts"] = now
                return
        self._fixture_protos.append(
            {"vec": vec, "cam": cam, "class": object_class, "ts": now, "n": 1})
        if len(self._fixture_protos) > self._fixture_max:
            self._fixture_protos.sort(key=lambda p: p["ts"])
            self._fixture_protos = self._fixture_protos[-self._fixture_max:]

    def is_fixture_appearance(self, vec, cam=None, object_class=DEFAULT_OBJECT_CLASS):
        """True if `vec` looks like a known fixture on this camera, for this
        identity class. Prototypes expire after _fixture_ttl so a chair a
        person now occupies is released. Matching is same-camera AND
        same-class only: a chair on cam-04 must not veto a real detection on
        cam-08, and a fixture learned for one object class must never veto a
        detection of a different class."""
        if vec is None or not self._fixture_protos:
            return False, 0.0
        vec = _l2(np.asarray(vec, dtype=np.float32))
        if vec.shape[0] != self.feat_sz or not np.all(np.isfinite(vec)):
            return False, 0.0
        now = time.time()
        self._fixture_protos = [p for p in self._fixture_protos
                                 if now - p["ts"] <= self._fixture_ttl]
        best = 0.0
        for p in self._fixture_protos:
            if p.get("class", DEFAULT_OBJECT_CLASS) != object_class:
                continue
            if cam is not None and p["cam"] is not None and p["cam"] != cam:
                continue
            s = self._cosine(vec, p["vec"])
            if s > best:
                best = s
        return (best >= self._fixture_reject_thr), best

    def touch(self, pid, cam=None):
        now = time.time()
        m = self._meta_cache.get(pid)
        if m:                                   # in-memory state: every call
            m["last_seen"] = now
            m["count"] += 1
            if cam is not None:
                m["last_cam"] = cam
                m.setdefault("cams", set()).add(cam)
        # SQLite state: throttled to one write per flush interval per (pid, cam).
        # Pending frame counts are accumulated so sighting_count / sightings.count
        # keep their "frames tracked" meaning. The first touch on a new camera
        # writes immediately (key not yet in _touch_flush_ts).
        key = (pid, cam)
        self._touch_pending[key] = self._touch_pending.get(key, 0) + 1
        if now - self._touch_flush_ts.get(key, 0.0) < self._touch_flush_every:
            return
        n = self._touch_pending.pop(key, 1)
        self._touch_flush_ts[key] = now
        self._record_sighting(pid, cam, n)
        self._conn.execute(
            "UPDATE persons SET last_seen=?, last_seen_str=?, "
            "sighting_count=sighting_count+?, "
            "last_camera=COALESCE(?, last_camera) WHERE person_id=?",
            (now, _ts_str(now), n, cam, pid))
        self._dirty = True

    def maybe_commit(self, force=False):
        """Commit deferred writes at most once per commit_interval. Called from
        the probe once per batch (cheap no-op when nothing is pending)."""
        now = time.time()
        if self._dirty and (force or now - self._last_commit >= self.commit_interval):
            with self._lock:
                self._conn.commit()
            self._dirty = False
            self._last_commit = now

    def gallery_size(self, pid):
        return len(self._emb_cache.get(pid, ()))

    def first_seen_of(self, pid):
        m = self._meta_cache.get(pid)
        if m and m.get("first_seen"):
            return m["first_seen"], m.get("first_seen_str") or _ts_str(m["first_seen"])
        with self._lock:
            row = self._conn.execute(
                "SELECT created_at, first_seen_str FROM persons WHERE person_id=?",
                (pid,)).fetchone()
        if row:
            ts, s = row[0], row[1] or _ts_str(row[0])
            return ts, s
        return None, ""

    def cameras_for(self, pid):
        return [r[0] for r in self._conn.execute(
            "SELECT camera FROM sightings WHERE person_id=? ORDER BY last_seen DESC",
            (pid,))]

    def last_seen_age(self, pid, now=None):
        """Seconds since this identity was last seen on ANY camera, or None if
        unknown. Used by callers deciding whether a candidate is eligible for
        the 15-minute absence/restore path."""
        m = self._meta_cache.get(pid)
        if not m or not m.get("last_seen"):
            return None
        now = now if now is not None else time.time()
        return now - m["last_seen"]

    def delete_person(self, pid):
        try:
            self._conn.execute("DELETE FROM embeddings WHERE person_id=?", (pid,))
            self._conn.execute("DELETE FROM sightings  WHERE person_id=?", (pid,))
            self._conn.execute("DELETE FROM persons     WHERE person_id=?", (pid,))
            self._retire_uid(pid, "pruned")
            self._conn.commit()
        except Exception as e:
            print(f"[DB] delete_person({pid}) DB error: {e}")
        self._emb_cache.pop(pid, None)
        self._mat_cache.pop(pid, None)
        self._meta_cache.pop(pid, None)
        try:
            self._purge_cannot_links(pid)
        except Exception as e:
            print(f"[DB] cannot-link purge failed: {e}")
        for _k in [k for k in self._touch_flush_ts if k[0] == pid]:
            self._touch_flush_ts.pop(_k, None)
            self._touch_pending.pop(_k, None)
        for key in [k for k in self._open_incidents if k[0] == pid]:
            self._open_incidents.pop(key, None)
        print(f"[DB] DELETED phantom person {pid}  (total {len(self._meta_cache)})")

    def open_incident(self, pid, cam, object_class=DEFAULT_OBJECT_CLASS, now=None):
        """Start a new incident for (pid, cam), or return the id of one
        already open. Status starts DETECTED and the caller is expected to
        call touch_incident() as detections continue, which promotes it to
        ACTIVE."""
        now = now if now is not None else time.time()
        key = (pid, cam)
        existing = self._open_incidents.get(key)
        if existing is not None:
            return existing
        cur = self._conn.execute(
            "INSERT INTO incidents (person_id, object_class, camera_id, "
            "entry_time, last_detected_time, exit_time, dwell_seconds, "
            "detection_count, movement_status, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,NULL,NULL,1,'unknown','DETECTED',?,?)",
            (pid, object_class, cam, now, now, now, now))
        incident_id = cur.lastrowid
        self._open_incidents[key] = incident_id
        self._dirty = True   # commit deferred to maybe_commit()
        return incident_id

    def touch_incident(self, pid, cam, now=None, moved=None):
        """Record a continued detection for the open incident on (pid, cam).
        Promotes DETECTED/POSSIBLY_EXITED -> ACTIVE (a renewed detection means
        the person is confirmed still present, not exited). If no incident is
        open, opens one first -- callers do not need to special-case the
        first call.

        moved: True/False/None. When known, folds into movement_status:
        'moving' if ever moved, else 'stationary' if always still, else
        'unknown'."""
        now = now if now is not None else time.time()
        key = (pid, cam)
        incident_id = self._open_incidents.get(key)
        if incident_id is None:
            incident_id = self.open_incident(pid, cam, now=now)

        move_clause = ""
        params = [now, now]
        if moved is True:
            move_clause = ", movement_status='moving'"
        elif moved is False:
            move_clause = (", movement_status = CASE WHEN movement_status='moving' "
                            "THEN 'moving' ELSE 'stationary' END")
        self._conn.execute(
            f"UPDATE incidents SET last_detected_time=?, "
            f"detection_count=detection_count+1, "
            f"status=CASE WHEN status='CLOSED' THEN 'CLOSED' ELSE 'ACTIVE' END, "
            f"updated_at=?{move_clause} WHERE incident_id=?",
            params + [incident_id])
        self._dirty = True   # commit deferred to maybe_commit()
        return incident_id

    def mark_possibly_exited(self, pid, cam, now=None):
        """Missed-detection grace period entered: the caller has not seen this
        UID on this camera for its configured timeout, but is not yet ready to
        declare an exit. Section 10 of the notes: a single missed detection
        must never close an incident outright."""
        now = now if now is not None else time.time()
        key = (pid, cam)
        incident_id = self._open_incidents.get(key)
        if incident_id is None:
            return None
        self._conn.execute(
            "UPDATE incidents SET status='POSSIBLY_EXITED', updated_at=? "
            "WHERE incident_id=? AND status != 'CLOSED'",
            (now, incident_id))
        self._conn.commit()
        return incident_id

    def close_incident(self, pid, cam, now=None):
        """Confirm the exit: close the open incident on (pid, cam) and compute
        final dwell_seconds = exit_time - entry_time."""
        now = now if now is not None else time.time()
        key = (pid, cam)
        incident_id = self._open_incidents.pop(key, None)
        if incident_id is None:
            return None
        row = self._conn.execute(
            "SELECT entry_time FROM incidents WHERE incident_id=?",
            (incident_id,)).fetchone()
        entry_time = row[0] if row else now
        dwell = max(0.0, now - entry_time)
        self._conn.execute(
            "UPDATE incidents SET exit_time=?, dwell_seconds=?, status='CLOSED', "
            "updated_at=? WHERE incident_id=?",
            (now, dwell, now, incident_id))
        self._conn.commit()
        return incident_id

    def open_incident_id(self, pid, cam):
        """The currently-open incident id for (pid, cam), or None."""
        return self._open_incidents.get((pid, cam))

    def incidents_for(self, pid):
        """All incidents (open or closed) for a global UID, most recent
        first."""
        return self._conn.execute(
            "SELECT incident_id, camera_id, entry_time, last_detected_time, "
            "exit_time, dwell_seconds, detection_count, movement_status, status "
            "FROM incidents WHERE person_id=? ORDER BY entry_time DESC",
            (pid,)).fetchall()

    def sweep_incidents(self, now=None, possibly_exited_after=15.0,
                         close_after=60.0):
        """Periodic housekeeping for the incident state machine (Notes
        section 10: 'a missed frame must not immediately close an incident').

        For every currently-open incident:
          * DETECTED/ACTIVE, not touched in >= possibly_exited_after seconds
            -> POSSIBLY_EXITED (grace period entered, not yet an exit).
          * POSSIBLY_EXITED, not touched in >= close_after seconds (measured
            from the SAME last_detected_time, so this is a total grace
            window, not close_after stacked on top of possibly_exited_after)
            -> CLOSED, with dwell_seconds computed from entry_time.

        Returns (possibly_exited_count, closed_count). Intended to be called
        from the same periodic cleanup cycle that already ages out stale
        tracker state (deepstream_app_reid.py's maybe_cleanup()) -- this
        method touches only the incidents table / _open_incidents index, so
        it cannot affect CSV session output."""
        now = now if now is not None else time.time()
        rows = self._conn.execute(
            "SELECT incident_id, person_id, camera_id, last_detected_time, status "
            "FROM incidents WHERE status IN ('DETECTED','ACTIVE','POSSIBLY_EXITED')"
        ).fetchall()
        n_possibly_exited = 0
        n_closed = 0
        for incident_id, pid, cam, last_detected, status in rows:
            age = now - (last_detected or now)
            if status in ("DETECTED", "ACTIVE") and age >= possibly_exited_after:
                self.mark_possibly_exited(pid, cam, now=now)
                n_possibly_exited += 1
            elif status == "POSSIBLY_EXITED" and age >= close_after:
                self.close_incident(pid, cam, now=now)
                n_closed += 1
        return n_possibly_exited, n_closed

    def close(self):
        with self._lock:
            if self._dirty:
                self._conn.commit()
            self._conn.commit()
            self._conn.close()

    def stats(self):
        by_class = defaultdict(int)
        for m in self._meta_cache.values():
            by_class[m.get("class", DEFAULT_OBJECT_CLASS)] += 1
        return {
            "persons": len(self._meta_cache),
            "persons_by_class": dict(by_class),
            "total_embeddings": sum(len(v) for v in self._emb_cache.values()),
            "open_incidents": len(self._open_incidents),
            "cannot_links": len(self._cannot),
        }


if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mktemp(suffix=".db")
    db = PersonDatabase(db_path=tmp, feature_size=8)
    rng = np.random.default_rng(0)

    a = rng.standard_normal(8).astype(np.float32)
    pid_a = db.create_person(a, object_class="person")

    a2 = a + 0.01 * rng.standard_normal(8).astype(np.float32)
    m, s = db.match(_l2(a2), object_class="person", threshold=0.70)
    assert m == pid_a, f"expected re-id, got {m} score {s:.3f}"

    b = rng.standard_normal(8).astype(np.float32)
    m2, s2 = db.match(_l2(b), object_class="person", threshold=0.70)
    print(f"self-test OK: reid score={s:.3f}, stranger score={s2:.3f}")

    v = rng.standard_normal(8).astype(np.float32)
    pid_person = db.create_person(v, object_class="person")
    pid_vehicle = db.create_person(v, object_class="vehicle")
    m3, s3 = db.match(_l2(v), object_class="vehicle", threshold=0.99)
    assert m3 == pid_vehicle, "vehicle query must never resolve to a person UID"
    assert not db.merge_persons(pid_person, pid_vehicle), \
        "cross-class merge must be refused"
    print("class-partition self-test OK: cross-class match/merge correctly blocked")

    c = rng.standard_normal(8).astype(np.float32)
    pid_c = db.create_person(c, object_class="person")
    m4, s4 = db.restore_candidates(_l2(c), object_class="person",
                                    min_absence_seconds=900)
    assert m4 is None, "recently-seen identity must not satisfy the absence rule"
    db._meta_cache[pid_c]["last_seen"] = time.time() - 960
    m5, s5 = db.restore_candidates(_l2(c), object_class="person",
                                    min_absence_seconds=900, threshold=0.90)
    assert m5 == pid_c, f"expected absence-restore match, got {m5} score {s5:.3f}"
    print("absence-restore self-test OK")

    inc1 = db.open_incident(pid_a, "cam-01")
    inc1b = db.touch_incident(pid_a, "cam-01")
    assert inc1 == inc1b
    db.mark_possibly_exited(pid_a, "cam-01")
    db.close_incident(pid_a, "cam-01")
    rows = db.incidents_for(pid_a)
    assert rows and rows[0][-1] == "CLOSED"
    print("incident self-test OK")

    inc2 = db.open_incident(pid_a, "cam-02")
    db._conn.execute("UPDATE incidents SET last_detected_time=? WHERE incident_id=?",
                      (time.time() - 20, inc2))
    db._conn.commit()
    n_pe, n_closed = db.sweep_incidents(possibly_exited_after=15.0, close_after=60.0)
    assert n_pe == 1 and n_closed == 0
    status = db._conn.execute("SELECT status FROM incidents WHERE incident_id=?",
                               (inc2,)).fetchone()[0]
    assert status == "POSSIBLY_EXITED"
    db._conn.execute("UPDATE incidents SET last_detected_time=? WHERE incident_id=?",
                      (time.time() - 70, inc2))
    db._conn.commit()
    n_pe2, n_closed2 = db.sweep_incidents(possibly_exited_after=15.0, close_after=60.0)
    assert n_closed2 == 1
    assert (pid_a, "cam-02") not in db._open_incidents
    print("incident sweep self-test OK")

    db.close()
    os.remove(tmp)
    print("ALL SELF-TESTS PASSED")
