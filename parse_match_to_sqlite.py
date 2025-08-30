#!/usr/bin/env python3
# One-file-per-match SQLite exporter from Awpy demos (CS2) with normalized events.
# Usage:
#   python3 parse_match_to_sqlite.py --outfile data/matches/<match>.sqlite path/to/demo1.dem [demo2.dem ...]

import argparse, json, re, sqlite3, hashlib
from pathlib import Path
import pandas as pd
import numpy as np

SAFE = re.compile(r"[^0-9A-Za-z._-]+")

def slugify(s: str) -> str:
    return SAFE.sub("_", s).strip("_")

def hsh(parts) -> str:
    m = hashlib.sha1()
    for p in parts:
        if p is None: p = "None"
        m.update(str(p).encode("utf-8"))
        m.update(b"|")
    return m.hexdigest()[:16]  # short stable id

def to_pandas(obj) -> pd.DataFrame:
    if obj is None: return pd.DataFrame()
    if isinstance(obj, pd.DataFrame): return obj
    if hasattr(obj, "to_pandas"):
        try: return obj.to_pandas()
        except Exception: pass
    try: return pd.DataFrame(obj)
    except Exception: return pd.DataFrame()

def add_common(df: pd.DataFrame, map_name: str, demo_file: str):
    if df.empty: return df
    df = df.copy()
    if "map_name" not in df.columns: df.insert(0, "map_name", map_name)
    if "demo_file" not in df.columns: df.insert(1, "demo_file", demo_file)
    return df

def tickrate_from_header(header: dict, default=128.0) -> float:
    if not isinstance(header, dict): return float(default)
    for k in ("tickrate","tickRate","ticks_per_second","ticksPerSecond"):
        if k in header and header[k]:
            try: return float(header[k])
            except Exception: pass
    return float(default)

def attach_round_by_tick(events: pd.DataFrame, rmap: pd.DataFrame, tick_col: str = "tick") -> pd.DataFrame:
    """
    Robust mapping of event ticks to rounds using merge_asof on round START.
    """
    if events.empty or "round" in events.columns: return events
    if rmap.empty or tick_col not in events.columns or "start" not in rmap.columns: return events
    ev = events.copy()
    r  = rmap.copy()
    ev["_tick"] = pd.to_numeric(ev[tick_col], errors="coerce").astype("Int64")
    r["_start"] = pd.to_numeric(r["start"], errors="coerce").astype("Int64")
    if "end" in r.columns: r["_end"] = pd.to_numeric(r["end"], errors="coerce").astype("Int64")
    ev = ev.dropna(subset=["_tick"]).copy()
    r  = r.dropna(subset=["_start"]).copy()
    r = r[["round","_start"] + (["_end"] if "_end" in r.columns else [])].sort_values("_start")
    r = r.drop_duplicates(subset=["_start"], keep="first")
    ev = ev.sort_values("_tick")
    mapped = pd.merge_asof(ev, r, left_on="_tick", right_on="_start", direction="backward")
    if "_end" in mapped.columns:
        mapped = mapped[(mapped["_tick"].isna()) | (mapped["_end"].isna()) | (mapped["_tick"] <= mapped["_end"])]
    mapped["round"] = mapped["round"].astype("Int64")
    return mapped.drop(columns=[c for c in ("_tick","_start","_end") if c in mapped.columns])

def grenades_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize per-tick projectile rows into 1-row-per-grenade throw:
    - group by (round, entity_id) if available; otherwise by (round, thrower_steamid, grenade_type)
    - start/end tick = min/max tick in the group
    - start/end position = first/last NON-NaN (X,Y,Z) in the group, if any
    - carry grenade type + thrower steamid when available
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Map your schema → normalized names
    # (add the underscore variants your build uses)
    col_round  = None
    for c in ["round","round_num","roundNumber","r","Round"]:
        if c in df.columns: col_round = c; break
    col_tick   = None
    for c in ["tick","frame","time","t"]:
        if c in df.columns: col_tick = c; break
    col_type   = None
    for c in ["grenadeType","grenade_type","nadeType","projectileType","projectile","grenade","type","weapon","name"]:
        if c in df.columns: col_type = c; break
    col_throw_sid = None
    for c in ["throwerSteamID","thrower_steamid","attackerSteamID","playerSteamID","ownerSteamID","owner","thrower"]:
        if c in df.columns: col_throw_sid = c; break
    col_gid    = None
    for c in ["grenadeId","projectileId","entity_id","entityId","id","entId"]:
        if c in df.columns: col_gid = c; break
    col_x = None
    for c in ["x","X","posX","position_x"]:
        if c in df.columns: col_x = c; break
    col_y = None
    for c in ["y","Y","posY","position_y"]:
        if c in df.columns: col_y = c; break
    col_z = None
    for c in ["z","Z","posZ","position_z"]:
        if c in df.columns: col_z = c; break
    col_event = None
    for c in ["event","action","state","Event"]:
        if c in df.columns: col_event = c; break

    # must have round & tick
    if not col_round or not col_tick:
        return pd.DataFrame(columns=[
            "round","grenadeType","throwerSteamID","grenadeId",
            "start_tick","start_x","start_y","start_z",
            "end_tick","end_x","end_y","end_z","end_event"
        ])

    # minimal working frame
    G = pd.DataFrame({
        "round": df[col_round],
        "tick":  pd.to_numeric(df[col_tick], errors="coerce")
    })
    if col_type:      G["grenadeType"]     = df[col_type].astype(str)
    else:             G["grenadeType"]     = "UNK"
    if col_throw_sid: G["throwerSteamID"]  = df[col_throw_sid]
    if col_gid:       G["grenadeId"]       = df[col_gid]
    if col_event:     G["event"]           = df[col_event].astype(str).str.lower()
    if col_x:         G["x"] = pd.to_numeric(df[col_x], errors="coerce")
    if col_y:         G["y"] = pd.to_numeric(df[col_y], errors="coerce")
    if col_z:         G["z"] = pd.to_numeric(df[col_z], errors="coerce")

    # choose grouping key
    if "grenadeId" in G.columns and G["grenadeId"].notna().any():
        group_cols = ["round","grenadeId"]
    else:
        # fall back when entity_id is missing: thrower + type + first tick
        # (still robust enough for early-window counting)
        group_cols = ["round","throwerSteamID","grenadeType"]

    G = G.dropna(subset=["round","tick"]).sort_values(group_cols + ["tick"])

    # helper: first non-NaN of a column in a group
    def first_valid(s):
        idx = s.first_valid_index()
        return s.loc[idx] if idx is not None else np.nan
    def last_valid(s):
        idx = s.last_valid_index()
        return s.loc[idx] if idx is not None else np.nan

    agg = {
        "tick": ["min","max"],
        "grenadeType": "first"
    }
    if "throwerSteamID" in G.columns: agg["throwerSteamID"] = "first"
    if "x" in G.columns: agg["x"] = [first_valid, last_valid]
    if "y" in G.columns: agg["y"] = [first_valid, last_valid]
    if "z" in G.columns: agg["z"] = [first_valid, last_valid]
    if "event" in G.columns: agg["event"] = "last"

    S = G.groupby(group_cols, dropna=False).agg(agg)

    # flatten columns
    S.columns = ["_".join([c for c in tup if c]).strip("_") for tup in S.columns.to_flat_index()]
    S = S.reset_index()

    # rename to normalized names
    out = pd.DataFrame({
        "round":      S["round"].astype("Int64"),
        "start_tick": S["tick_min"],
        "end_tick":   S["tick_max"],
        "grenadeType": S.get("grenadeType_first","UNK")
    })
    if "throwerSteamID_first" in S.columns:
        out["throwerSteamID"] = S["throwerSteamID_first"]
    if "grenadeId" in S.columns:
        out["grenadeId"] = S["grenadeId"]

    # positions if present
    if "x_first_valid" in S.columns:
        out["start_x"] = S["x_first_valid"]; out["end_x"] = S["x_last_valid"]
    if "y_first_valid" in S.columns:
        out["start_y"] = S["y_first_valid"]; out["end_y"] = S["y_last_valid"]
    if "z_first_valid" in S.columns:
        out["start_z"] = S["z_first_valid"]; out["end_z"] = S["z_last_valid"]

    # event at end if we have it
    if "event_last" in S.columns:
        out["end_event"] = S["event_last"]

    return out.reset_index(drop=True)


def write_df_sqlite(conn, df: pd.DataFrame, table: str):
    if df.empty: return
    df.to_sql(table, conn, if_exists="append", index=False)

def add_indexes(conn):
    cur = conn.cursor()
    for t in ["rounds","kills_slim","damages_slim","shots_slim","bomb","players_slim","grenades_summary","header",
              "events_shots","events_grenades"]:
        try:
            cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{t}_mk ON {t}(map_name, demo_file);')
        except Exception: pass
    for t in ["kills_slim","damages_slim","shots_slim","bomb","players_slim","grenades_summary",
              "events_shots","events_grenades"]:
        try:
            cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{t}_round ON {t}(map_name, demo_file, round);')
        except Exception: pass
    try:
        cur.execute('CREATE INDEX IF NOT EXISTS idx_events_shots_eid ON events_shots(event_id);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_events_grenades_eid ON events_grenades(event_id);')
    except Exception: pass

    try:
        cur.execute('CREATE INDEX IF NOT EXISTS idx_round_loadouts_mk ON round_loadouts(map_name, demo_file);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_round_loadouts_round ON round_loadouts(map_name, demo_file, round);')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_round_loadouts_sid ON round_loadouts(map_name, demo_file, steamid);')
    except Exception:
        pass

    conn.commit()

def write_manifest(conn):
    try:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        rows = []
        for t in tables["name"]:
            try:
                n = pd.read_sql(f"SELECT COUNT(*) AS n FROM [{t}]", conn)["n"].iloc[0]
                rows.append({"table": t, "rows": int(n)})
            except Exception:
                rows.append({"table": t, "rows": None})
        pd.DataFrame(rows).to_sql("manifest", conn, if_exists="replace", index=False)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser(description="Parse one match (one or more .dem) into a single SQLite file (with normalized events).")
    ap.add_argument("--outfile", required=True, help="Output .sqlite path for this match")
    ap.add_argument("demos", nargs="+", help=".dem paths (BO1/BO3...)")
    args = ap.parse_args()

    from awpy import Demo

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out))

    # Process each demo file into tables
    for dem_path in args.demos:
        dem_path = Path(dem_path)
        d = Demo(str(dem_path), verbose=False)
        d.parse(
            player_props=[
                "name", "steamid", "team_num",
                "health", "armor", "has_helmet",
                "inventory",  # list[str] of weapons/utility
            ]
        )

        header = getattr(d, "header", {}) or {}
        map_name = header.get("map_name") or header.get("mapName") if isinstance(header, dict) else "unknown"
        demo_file = dem_path.name
        tickrate = tickrate_from_header(header, 128.0)

        # header table
        pd.DataFrame([{"demo_file": demo_file, "map_name": map_name, "header_json": json.dumps(header),
                       "tickrate": tickrate}]).to_sql("header", conn, if_exists="append", index=False)

        # rounds (keep start/end if present)
        rounds = to_pandas(getattr(d, "rounds", None))
        if not rounds.empty:
            if "round_num" in rounds.columns and "round" not in rounds.columns:
                rounds = rounds.rename(columns={"round_num":"round"})
            keep_cols = [c for c in
                         ("round", "winner", "start", "end", "freeze_end", "bomb_planted", "score_ct", "score_t") if
                         c in rounds.columns]

            if not keep_cols: keep_cols = [c for c in ("round","winner") if c in rounds.columns]
            rounds = rounds[keep_cols]
            rounds = add_common(rounds, map_name, demo_file)
            write_df_sqlite(conn, rounds, "rounds")
        else:
            rounds = pd.DataFrame(columns=["round","winner","start","end"])

        # rmap for this demo (we also want freeze_end if present)
        rmap = pd.read_sql(
            "SELECT round, start, end, freeze_end FROM rounds WHERE demo_file = ? AND map_name = ?",
            conn, params=(demo_file, map_name)
        )

        # === ROUND LOADOUTS ===
        # Vytvoří per-kolo snapshot výzbroje hráčů (inventář, HP/Kevlar/Helma) v ticku >= freeze_end.
        ticks = to_pandas(getattr(d, "ticks", None))
        if not ticks.empty and not rmap.empty:
            T = ticks.copy()

            # --- Normalize/rename columns across AWPy variants ---
            ren = {}
            if "round_num" in T.columns and "round" not in T.columns: ren["round_num"] = "round"
            if "steamID" in T.columns and "steamid" not in T.columns: ren["steamID"] = "steamid"
            T = T.rename(columns=ren)

            # Build/derive team_num if missing (fallbacks from side/team text)
            if "team_num" not in T.columns:
                if "side" in T.columns:
                    side = T["side"].astype(str).str.upper()
                    T["team_num"] = np.where(side.eq("CT"), 3, np.where(side.eq("T"), 2, np.nan))
                elif "team" in T.columns:
                    tm = T["team"].astype(str).str.upper()
                    T["team_num"] = np.where(tm.eq("CT"), 3, np.where(tm.eq("T"), 2, np.nan))

            # We need at minimum: round, tick, steamid, inventory + a few props
            needed = {"round", "tick", "steamid", "team_num", "name", "inventory"}
            if not needed.issubset(T.columns):
                # Try to be graceful: if inventory isn’t there, skip silently for this demo
                pass
            else:
                # Attach freeze_end for each round; fallback to round start if freeze_end is missing
                rm = rmap.copy()
                if "freeze_end" not in rm.columns:
                    rm["freeze_end"] = rm.get("start")  # fallback

                # Ensure numeric for comparisons
                T["tick"] = pd.to_numeric(T["tick"], errors="coerce")
                rm["freeze_end"] = pd.to_numeric(rm["freeze_end"], errors="coerce")

                # Merge freeze_end to each player tick row via round
                T = T.merge(rm[["round", "freeze_end"]], on="round", how="left")

                # Keep only first row per (round, player) with tick >= freeze_end
                # (if freeze_end NaN, we keep the first available tick for that round)
                sel = (
                        (T["freeze_end"].isna()) |
                        (T["tick"] >= T["freeze_end"])
                )
                S = (
                    T[sel]
                    .sort_values(["round", "steamid", "tick"])
                    .groupby(["round", "steamid"], as_index=False)
                    .first()
                    .copy()
                )

                # Keep only live teams (2=T, 3=CT)
                S = S[S["team_num"].isin([2, 3])].copy()

                # Pretty team label & robust inventory JSON
                TEAM_MAP = {2: "T", 3: "CT"}
                S["team"] = S["team_num"].map(TEAM_MAP)

                def inv_to_json(val):
                    if isinstance(val, (list, tuple, np.ndarray)):
                        try:
                            return json.dumps(list(val))
                        except Exception:
                            return json.dumps([])
                    if isinstance(val, str):
                        # sometimes already JSON, sometimes a single item
                        v = val.strip()
                        if v.startswith("[") and v.endswith("]"):
                            try:
                                return json.dumps(json.loads(v))
                            except Exception:
                                return json.dumps([])
                        return json.dumps([val])
                    return json.dumps([])

                S["inventory_json"] = S["inventory"].apply(inv_to_json)

                # Boolean helmet → int (sqlite friendly)
                if "has_helmet" in S.columns:
                    S["has_helmet"] = S["has_helmet"].astype("Int64")

                # Column selection (name them explicitly to be stable)
                keep_cols = ["round", "steamid", "name", "team_num", "team", "health", "armor"]
                if "has_helmet" in S.columns: keep_cols.append("has_helmet")
                keep_cols.append("inventory_json")

                loadouts = add_common(S.loc[:, keep_cols].copy(), map_name, demo_file)

                # Stable synthetic id (demo_file|round|steamid) if you want it (handy for joins)
                loadouts = loadouts.assign(
                    row_id=loadouts.apply(
                        lambda r: hsh([r.get("demo_file"), r.get("round"), r.get("steamid")]),
                        axis=1
                    )
                )

                write_df_sqlite(conn, loadouts, "round_loadouts")

                # (Optional) Fully normalized items table: one row per item
                # Uncomment this block if you want easy SQL counts per weapon.
                """
                rows = []
                for _, r in loadouts.iterrows():
                    inv = []
                    try:
                        inv = json.loads(r["inventory_json"]) or []
                    except Exception:
                        inv = []
                    for item in inv:
                        rows.append({
                            "map_name": r["map_name"],
                            "demo_file": r["demo_file"],
                            "round": int(r["round"]) if pd.notna(r["round"]) else None,
                            "steamid": r["steamid"],
                            "team": r["team"],
                            "item": str(item),
                            "row_id": r["row_id"],
                        })
                if rows:
                    items_df = pd.DataFrame(rows)
                    write_df_sqlite(conn, items_df, "round_loadout_items")
                """

        # KILLS (slim pass-through, with round mapping if needed)
        kills = to_pandas(getattr(d, "kills", None))
        if not kills.empty:
            ren = {}
            if "round_num" in kills.columns and "round" not in kills.columns: ren["round_num"]="round"
            if "killerSide" not in kills.columns and "attacker_side" in kills.columns: ren["attacker_side"]="killerSide"
            if ren: kills = kills.rename(columns=ren)
            if not kills.empty and not rmap.empty:
                kills = attach_round_by_tick(kills, rmap, tick_col="tick")
            keep = [c for c in ["round","tick","killerSteamID","victimSteamID","killerSide","weapon","headshot","x","y","z"] if c in kills.columns]
            if keep:
                write_df_sqlite(conn, add_common(kills[keep], map_name, demo_file), "kills_slim")

        # DAMAGES (slim)
        damages = to_pandas(getattr(d, "damages", None))
        if not damages.empty:
            if not rmap.empty:
                damages = attach_round_by_tick(damages, rmap, tick_col="tick")
            keep = [c for c in ["round","tick","attackerSteamID","victimSteamID","hpDamage","armorDamage","weapon","hitgroup"] if c in damages.columns]
            if keep:
                write_df_sqlite(conn, add_common(damages[keep], map_name, demo_file), "damages_slim")

        # SHOTS (slim → normalized events_shots)
        shots = to_pandas(getattr(d, "shots", None))
        if not shots.empty:
            if "round_num" in shots.columns and "round" not in shots.columns:
                shots = shots.rename(columns={"round_num":"round"})
            if not rmap.empty:
                shots = attach_round_by_tick(shots, rmap, tick_col="tick")
            keep = [c for c in ["round","tick","shooterSteamID","weapon","x","y","z"] if c in shots.columns]
            if keep:
                shots_slim = shots[keep].copy()
                shots_slim = add_common(shots_slim, map_name, demo_file)
                write_df_sqlite(conn, shots_slim, "shots_slim")

                # Build events_shots (deduped: one row per shot)
                S = shots_slim.dropna(subset=["round","tick"]).copy()

                # strongest available dedup key
                if "shooterSteamID" in S.columns and "weapon" in S.columns:
                    S = S.drop_duplicates(subset=["map_name","demo_file","round","tick","shooterSteamID","weapon"])
                elif "shooterSteamID" in S.columns:
                    S = S.drop_duplicates(subset=["map_name","demo_file","round","tick","shooterSteamID"])
                else:
                    S = S.drop_duplicates(subset=["map_name","demo_file","round","tick"])

                # attach round start to compute seconds (if available)
                if not rmap.empty and "start" in rmap.columns:
                    rs = rmap.rename(columns={"start":"round_start"})[["round","round_start"]]
                    S = S.merge(rs, on="round", how="left")

                S["t_round"] = pd.to_numeric(S["tick"], errors="coerce") - pd.to_numeric(S.get("round_start"), errors="coerce")
                S["t_sec"]   = S["t_round"] / float(tickrate)

                # stable event id
                S["event_id"] = S.apply(
                    lambda r: hsh([demo_file, r.get("round"), r.get("tick"), r.get("shooterSteamID",""), r.get("weapon","")]),
                    axis=1
                )

                # ---- DYNAMIC COLUMN SELECTION HERE ----
                base_cols = ["map_name","demo_file","round","tick","t_round","t_sec","event_id"]
                opt_cols  = [c for c in ["shooterSteamID","weapon","x","y","z"] if c in S.columns]
                cols      = base_cols + opt_cols

                events_shots = S[cols].drop_duplicates(subset=["event_id"])
                write_df_sqlite(conn, events_shots, "events_shots")

        # BOMB (raw slim)
        bomb_raw = to_pandas(getattr(d, "bomb", None))
        if not bomb_raw.empty:
            b = bomb_raw.copy()
            # Normalize round col if needed
            if "round_num" in b.columns and "round" not in b.columns:
                b = b.rename(columns={"round_num": "round"})

            # detect columns (several awpy variants exist)
            event_col = next((c for c in ["event","action","state","Event","bombEvent"] if c in b.columns), None)
            site_col  = next((c for c in ["site","bombsite","BombSite","bomb_site"] if c in b.columns), None)
            tick_col  = "tick" if "tick" in b.columns else None
            # possible id/name cols
            pid_col   = next((c for c in ["playerSteamID","defuserSteamID","planterSteamID","steamID"] if c in b.columns), None)
            pname_col = next((c for c in ["player","defuser","planter","name"] if c in b.columns), None)

            # normalize event text
            if event_col:
                b["_event_lc"] = b[event_col].astype(str).str.lower()
            else:
                b["_event_lc"] = ""

            KEEP_EXPLODE = True  # set False if you don't want explode rows

            def norm_evt(e: str) -> str:
                e = (e or "").lower()
                if "plant" in e:   return "planted"
                if "defus" in e:   return "defused"
                if KEEP_EXPLODE and ("explod" in e or "explode" in e or "detonat" in e):
                    return "explode"
                return ""

            b["_evt_norm"] = b["_event_lc"].map(norm_evt)
            b = b[b["_evt_norm"].isin({"planted","defused","explode"} if KEEP_EXPLODE else {"planted","defused"})].copy()

            # Attach round via tick if needed
            if not rmap.empty and tick_col:
                b = attach_round_by_tick(b, rmap, tick_col=tick_col)

            # Build a minimal, column-safe frame
            cols = ["round", "_evt_norm"]
            if tick_col:  cols.append(tick_col)
            if site_col:  cols.append(site_col)
            if pid_col:   cols.append(pid_col)
            if pname_col: cols.append(pname_col)

            cols = [c for c in cols if c in b.columns]  # keep only existing
            b = b[cols].copy()

            # rename normalized names when present
            ren = {"_evt_norm": "event"}
            if site_col:  ren[site_col]  = "site"
            if pid_col:   ren[pid_col]   = "playerSteamID"
            if pname_col: ren[pname_col] = "playerName"
            b = b.rename(columns=ren)

            # add map/demo
            b = add_common(b, map_name, demo_file)

            # timing (seconds since round start) if we have tick
            if tick_col:
                rs = rmap.rename(columns={"start":"round_start"})[["round","round_start"]]
                b = b.merge(rs, on="round", how="left")
                b["t_round"] = pd.to_numeric(b[tick_col], errors="coerce") - pd.to_numeric(b["round_start"], errors="coerce")
                b["t_sec"]   = b["t_round"] / float(tickrate)

            # stable event id (works even if player id is missing)
            def mk_bid(r):
                return hsh([
                    demo_file,
                    r.get("round"),
                    r.get(tick_col, ""),
                    r.get("site",""),
                    r.get("playerSteamID",""),
                    r.get("playerName",""),
                    r.get("event",""),
                ])
            b["event_id"] = b.apply(mk_bid, axis=1)

            # Column-safe write
            out_cols = ["map_name","demo_file","round","event","event_id"]
            opt = [c for c in ["site","playerSteamID","playerName","t_round","t_sec"] if c in b.columns]
            if tick_col and tick_col in b.columns: opt.append(tick_col)
            write_df_sqlite(conn, b[out_cols + opt], "bomb")

            # Optional alias table
            try:
                write_df_sqlite(conn, b[out_cols + opt], "events_bomb")
            except Exception:
                pass


        # PLAYERS (slim)
        players = to_pandas(getattr(d, "players", None))
        if not players.empty:
            if "round_num" in players.columns and "round" not in players.columns:
                players = players.rename(columns={"round_num":"round"})
            keep = [c for c in ["round","steamID","name","team","side"] if c in players.columns]
            if keep:
                write_df_sqlite(conn, add_common(players[keep], map_name, demo_file), "players_slim")

        # GRENADES (projectiles only → raw summary + normalized events)
        gren = to_pandas(getattr(d, "grenades", None))
        if not gren.empty:
            # keep only actual thrown projectiles
            if "grenade_type" in gren.columns:
                gren = gren[gren["grenade_type"].astype(str).str.contains("Projectile", na=False)]
            elif "grenadeType" in gren.columns:
                gren = gren[gren["grenadeType"].astype(str).str.contains("Projectile", na=False)]

            # --- compact per-throw summary (start/end, positions) ---
            gsum = grenades_summary(gren)   # make sure you’re using the newer version that understands entity_id/round_num
            if not gsum.empty:
                gsum = add_common(gsum, map_name, demo_file)
                write_df_sqlite(conn, gsum, "grenades_summary")

            # --- normalized event rows: one row = one projectile instance ---
            if not gren.empty:
                G = gren.copy()

                # stable event id per projectile (entity_id if present; fallback to round/tick/type/thrower)
                def mk_eid(r):
                    return hsh([
                        demo_file,
                        r.get("entity_id", ""),
                        r.get("round_num", ""),
                        r.get("tick", ""),
                        r.get("grenade_type", r.get("grenadeType", "")),
                        r.get("thrower_steamid", r.get("throwerSteamID", "")),
                    ])
                G["event_id"] = G.apply(mk_eid, axis=1)

                # select/rename to normalized columns
                keep_ev = [c for c in ["round_num","tick","grenade_type","thrower_steamid","X","Y","Z","entity_id","event_id"] if c in G.columns]
                events_gren = G[keep_ev].rename(columns={
                    "round_num": "round",
                    "grenade_type": "grenadeType",
                    "thrower_steamid": "throwerSteamID"
                })
                events_gren = add_common(events_gren, map_name, demo_file)

                # de-dup projectiles (prefer entity_id if present)
                if "entity_id" in events_gren.columns:
                    events_gren = events_gren.drop_duplicates(subset=["map_name","demo_file","round","entity_id"])
                else:
                    events_gren = events_gren.drop_duplicates(subset=["map_name","demo_file","round","tick","grenadeType","throwerSteamID"])

                write_df_sqlite(conn, events_gren, "events_grenades")

        # CHAT (optional)
        chat = to_pandas(getattr(d, "chat", None))
        if not chat.empty:
            if "round_num" in chat.columns and "round" not in chat.columns:
                chat = chat.rename(columns={"round_num":"round"})
            keep = [c for c in ["round","tick","player","message"] if c in chat.columns]
            if keep:
                write_df_sqlite(conn, add_common(chat[keep], map_name, demo_file), "chat")

    # finalize
    add_indexes(conn)
    write_manifest(conn)
    conn.close()
    print(f"[OK] Wrote SQLite: {out}")

if __name__ == "__main__":
    main()
