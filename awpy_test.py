#!/usr/bin/env python3
import argparse
from pathlib import Path
from awpy import Demo
import polars as pl

TEAM_MAP = {2: "T", 3: "CT"}

def main():
    ap = argparse.ArgumentParser(description="Vypíše loadout (inventory) hráčů po freeze_end v každém kole z CS2 .dem pomocí AWPy.")
    ap.add_argument("demo", type=Path, help="Cesta k .dem souboru")
    args = ap.parse_args()

    # Parse demo a explicitně si řekneme o 'inventory' a pár metadat hráče
    dem = Demo(str(args.demo), verbose=False)
    dem.parse(
        player_props=[
            "name", "steamid", "team_num",
            "health", "armor", "has_helmet",  # Opraveno: používáme 'armor' místo 'armor_value'
            "inventory",  # <-- klíčové: AWPy/demoparser2 vrací list[str] se zbraněmi
        ]
    )

    rounds: pl.DataFrame = dem.rounds
    ticks: pl.DataFrame = dem.ticks

    # Základní sanity check
    for col in ("round_num", "freeze_end"):
        if col not in rounds.columns:
            raise SystemExit(f"Chybí '{col}' v dem.rounds (sloupce: {rounds.columns})")
    for col in ("round_num", "tick", "steamid", "name", "team_num", "inventory"):
        if col not in ticks.columns:
            raise SystemExit(f"Chybí '{col}' v dem.ticks (sloupce: {ticks.columns})")

    # Pro každé kolo vezmeme první záznam hráče v ticku >= freeze_end
    for r in rounds.iter_rows(named=True):
        rnum = r["round_num"]
        fe = r["freeze_end"]
        print(f"\n=== Kolo {rnum} (freeze_end: {fe}) ===")

        # Vezmeme všechny per-player tick řádky v daném kole s tick >= freeze_end
        subset = (
            ticks
            .filter((pl.col("round_num") == rnum) & (pl.col("tick") >= fe))
            .sort(["steamid", "tick"])
            .group_by("steamid")
            .first()
            .select("name", "steamid", "team_num", "health", "armor", "has_helmet", "inventory")
        )  # Už není potřeba .collect()

        # CT a T zvlášť (3=CT, 2=T – dle demoparser2)
        for side_num in (3, 2):
            side = TEAM_MAP.get(side_num, str(side_num))
            side_df = subset.filter(pl.col("team_num") == side_num)
            print(f"[{side}]")
            if side_df.is_empty():
                print("  (žádní hráči)")
                continue
            for row in side_df.iter_rows(named=True):
                meta = []
                if row["health"] is not None: meta.append(f"HP:{row['health']}")
                if row["armor"] is not None: meta.append(f"Kevlar:{row['armor']}")
                if row["has_helmet"]: meta.append("Helmet")
                inv = row["inventory"] or []
                print(f"  - {row['name']} ({row['steamid']})" + (f" | {' | '.join(meta)}" if meta else ""))
                print(f"    inventory: {inv}")  # přesně tak, jak to vrací AWPy (list[str])

if __name__ == "__main__":
    main()
