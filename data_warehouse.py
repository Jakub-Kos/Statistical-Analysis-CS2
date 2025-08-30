#!/usr/bin/env python3
import sqlite3, glob, os, pandas as pd
from pathlib import Path

SRC_GLOB = "data/matches/*.sqlite"
OUT_DB   = "data/warehouse.sqlite"

TABLES = [
    ("rounds",              ["map_name","demo_file","round","winner","start","end"]),  # keep minimal; schema supports more
    ("kills_slim",          ["map_name","demo_file","round","tick","killerSteamID","victimSteamID","killerSide","weapon","headshot","x","y","z"]),
    ("events_shots",        ["map_name","demo_file","round","tick","t_round","t_sec","event_id","shooterSteamID","weapon","x","y","z"]),
    ("events_grenades",     ["map_name","demo_file","round","grenadeType","throwerSteamID","grenadeId","start_tick","end_tick","event_id"]),
    ("events_bomb",         ["map_name","demo_file","round","event","site","playerSteamID","playerName","t_round","t_sec","event_id"]),
    ("grenades_summary",    ["map_name","demo_file","round","grenadeType","throwerSteamID","grenadeId","start_tick","end_tick","event_id"]),
    ("header",              ["demo_file","map_name","header_json","tickrate"]),
    ("round_loadouts",      ["map_name","demo_file","round","steamid","name","team_num","team","health","armor","has_helmet","inventory_json","row_id"]),
]

# Tables where we want INSERT OR IGNORE behavior (to deduplicate)
OR_IGNORE_TABLES = {"events_shots","events_grenades","events_bomb","round_loadouts"}

def ensure_schema(con):
    cur = con.cursor()
    cur.executescript("""
    -- Core tables
    CREATE TABLE IF NOT EXISTS rounds              (match_id TEXT, map_name TEXT, demo_file TEXT, round INT, winner TEXT, start INT, end INT);
    CREATE TABLE IF NOT EXISTS kills_slim          (match_id TEXT, map_name TEXT, demo_file TEXT, round INT, tick INT, killerSteamID TEXT, victimSteamID TEXT, killerSide TEXT, weapon TEXT, headshot INT, x REAL, y REAL, z REAL);
    CREATE TABLE IF NOT EXISTS events_shots        (match_id TEXT, map_name TEXT, demo_file TEXT, round INT, tick INT, t_round REAL, t_sec REAL, event_id TEXT UNIQUE, shooterSteamID TEXT, weapon TEXT, x REAL, y REAL, z REAL);
    CREATE TABLE IF NOT EXISTS events_grenades     (match_id TEXT, map_name TEXT, demo_file TEXT, round INT, grenadeType TEXT, throwerSteamID TEXT, grenadeId TEXT, start_tick INT, end_tick INT, event_id TEXT UNIQUE);
    CREATE TABLE IF NOT EXISTS events_bomb         (match_id TEXT, map_name TEXT, demo_file TEXT, round INT, event TEXT, site TEXT, playerSteamID TEXT, playerName TEXT, t_round REAL, t_sec REAL, event_id TEXT UNIQUE);
    CREATE TABLE IF NOT EXISTS grenades_summary    (match_id TEXT, map_name TEXT, demo_file TEXT, round INT, grenadeType TEXT, throwerSteamID TEXT, grenadeId TEXT, start_tick INT, end_tick INT, event_id TEXT);

    CREATE TABLE IF NOT EXISTS header              (match_id TEXT, demo_file TEXT, map_name TEXT, header_json TEXT, tickrate REAL);

    -- NEW: per-round loadouts
    CREATE TABLE IF NOT EXISTS round_loadouts      (
        match_id TEXT,
        map_name TEXT,
        demo_file TEXT,
        round INT,
        steamid TEXT,
        name TEXT,
        team_num INT,
        team TEXT,
        health REAL,
        armor INT,
        has_helmet INT,
        inventory_json TEXT,
        row_id TEXT
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_rounds_key       ON rounds(match_id, map_name, demo_file, round);
    CREATE INDEX IF NOT EXISTS idx_kills_key        ON kills_slim(match_id, map_name, demo_file, round);
    CREATE INDEX IF NOT EXISTS idx_shots_eid        ON events_shots(event_id);
    CREATE INDEX IF NOT EXISTS idx_nades_eid        ON events_grenades(event_id);
    CREATE INDEX IF NOT EXISTS idx_bomb_eid         ON events_bomb(event_id);

    -- NEW: loadouts indexes
    CREATE INDEX IF NOT EXISTS idx_loadouts_key     ON round_loadouts(match_id, map_name, demo_file, round);
    CREATE INDEX IF NOT EXISTS idx_loadouts_sid     ON round_loadouts(match_id, steamid);
    -- dedup key: one snapshot per (match, demo file, round, player)
    CREATE UNIQUE INDEX IF NOT EXISTS uq_loadouts_row ON round_loadouts(match_id, demo_file, round, steamid);

    """)
    con.commit()

def import_one(src_db, con_out):
    match_id = Path(src_db).stem  # např. "https___www.hltv.org_download_demo_99483"
    con_in = sqlite3.connect(src_db)
    for tname, cols in TABLES:
        # check table existence & columns in source
        tbl_exists = not pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name=?", con_in, params=(tname,)).empty
        if not tbl_exists:
            continue
        cols_in_db = pd.read_sql(f"PRAGMA table_info({tname})", con_in)["name"].tolist()
        use_cols = [c for c in cols if c in cols_in_db]
        if not use_cols:
            continue

        df = pd.read_sql(f"SELECT {', '.join(use_cols)} FROM {tname}", con_in)
        if df.empty:
            continue

        # add match_id
        df.insert(0, "match_id", match_id)

        if tname in OR_IGNORE_TABLES:
            # OR IGNORE path using a temp table to keep column order
            tmp = f"tmp_{tname}"
            df.to_sql(tmp, con_out, if_exists="replace", index=False)
            con_out.execute(f"""
                INSERT OR IGNORE INTO {tname} ({', '.join(df.columns)})
                SELECT {', '.join(df.columns)} FROM {tmp}
            """)
            con_out.execute(f"DROP TABLE {tmp}")
            con_out.commit()
        else:
            df.to_sql(tname, con_out, if_exists="append", index=False)
    con_in.close()

out = sqlite3.connect(OUT_DB)
# faster import
out.execute("PRAGMA journal_mode=OFF;")
out.execute("PRAGMA synchronous=OFF;")
out.execute("PRAGMA temp_store=MEMORY;")
out.execute("PRAGMA cache_size=-200000;")  # ~200MB
ensure_schema(out)

for src in glob.glob(SRC_GLOB):
    import_one(src, out)

# safer settings for subsequent reads
out.execute("PRAGMA journal_mode=WAL;")
out.execute("PRAGMA synchronous=NORMAL;")
out.commit()
out.close()
print(f"Built {OUT_DB}")
