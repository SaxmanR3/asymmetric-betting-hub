import csv
import io
import json
import math
import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str:
    try:
        if "ODDSPAPI_API_KEY" in st.secrets:
            return str(st.secrets["ODDSPAPI_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("ODDSPAPI_API_KEY", "").strip()


API_KEY = get_api_key()
BASE_URL = "https://api.oddspapi.io/v4"
LOG_FILE = "bet_log.csv"
LOG_COLUMNS = [
    "Date",
    "Platform",
    "Strategy",
    "Pick",
    "Stake",
    "Odds",
    "Closing_Odds",
    "Result",
    "Profit",
    "CLV_Pct",
]

DEFAULT_BALANCES = {
    "DraftKings": 40.12,
    "FanDuel": 14.80,
    "PrizePicks": 0.00,
    "Kalshi": 0.00,
    "BetMGM": 0.00,
    "Fanatics": 0.00,
}

PLATFORMS = ["FanDuel", "DraftKings", "PrizePicks", "Kalshi", "BetMGM", "Fanatics"]
STRATEGIES = [
    "Combos/Singles",
    "Alt Ladder",
    "5/6-Pick Flex",
    "+EV Straight",
    "Promo/Boost",
    "Neg-Corr SGP",
]
RESULTS = ["Pending", "Win", "Loss", "Push", "Void"]

BOOK_LABELS = {
    "pinnacle": "Pinnacle",
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "kalshi": "Kalshi",
    "betmgm": "BetMGM",
    "fanatics": "Fanatics",
    "caesars": "Caesars",
    "bet365": "bet365",
}

TARGET_SOFT_BOOKS = [
    "draftkings",
    "fanduel",
    "kalshi",
    "betmgm",
    "fanatics",
    "caesars",
    "bet365",
]

DEFAULT_SPORTS = {
    "NBA": {"sportId": 11, "filter": "nba"},
    "NFL": {"sportId": 14, "filter": "nfl"},
    "NHL": {"sportId": 16, "filter": "nhl"},
    "NCAAF": {"sportId": 14, "filter": "ncaaf"},
    "NCAAM": {"sportId": 11, "filter": "ncaam"},
    "NCAAW": {"sportId": 11, "filter": "ncaaw"},
    "WNBA": {"sportId": 11, "filter": "wnba"},
    "MLB": {"sportId": 13, "filter": "mlb"},
    "MLS": {"sportId": 10, "filter": "mls"},
}

st.set_page_config(page_title="Asymmetric Betting Hub", layout="wide")


def amer_to_dec(odds: float) -> float:
    return (1.0 + 100.0 / abs(odds)) if odds < 0 else (1.0 + odds / 100.0)


def amer_to_prob(odds: float) -> float:
    return (abs(odds) / (abs(odds) + 100.0)) if odds < 0 else (100.0 / (odds + 100.0))


def compute_profit_and_clv(stake: float, odds: int, close_odds: int, result: str) -> Tuple[float, float]:
    if result == "Win":
        profit = round(stake * (amer_to_dec(odds) - 1.0), 2)
    elif result == "Loss":
        profit = round(-float(stake), 2)
    else:
        profit = 0.0
    clv_pct = round((amer_to_prob(close_odds) - amer_to_prob(odds)) * 100.0, 2)
    return profit, clv_pct


def fmt_book(slug: str) -> str:
    return BOOK_LABELS.get(slug.lower(), slug.title())


def get_restricted_players() -> List[str]:
    raw = st.session_state.get("restricted_players_raw", "")
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def find_roster_conflicts(text: str) -> List[str]:
    if not text:
        return []
    lower_text = text.lower()
    return [p for p in get_restricted_players() if p in lower_text]


def init_state() -> None:
    st.session_state.setdefault("sport_map", dict(DEFAULT_SPORTS))
    st.session_state.setdefault("restricted_players_raw", "")
    st.session_state.setdefault("auto_sync_enabled", True)
    for plat in DEFAULT_BALANCES:
        st.session_state.setdefault(f"bal_adj_{plat}", 0.0)


def apply_bankroll_delta(plat: str, delta: float) -> None:
    if st.session_state.get("auto_sync_enabled", True) and plat in DEFAULT_BALANCES:
        key = f"bal_adj_{plat}"
        st.session_state[key] = round(float(st.session_state.get(key, 0.0)) + delta, 2)


def get_bet_log() -> pd.DataFrame:
    if "bet_log_df" not in st.session_state:
        if os.path.exists(LOG_FILE):
            try:
                df = pd.read_csv(LOG_FILE)
                for col in LOG_COLUMNS:
                    if col not in df.columns:
                        df[col] = 0.0 if col in ("Stake", "Profit", "CLV_Pct") else (-110 if "Odds" in col else "")
                st.session_state["bet_log_df"] = df[LOG_COLUMNS].copy()
                return st.session_state["bet_log_df"]
            except Exception:
                pass
        st.session_state["bet_log_df"] = pd.DataFrame(columns=LOG_COLUMNS)
    return st.session_state["bet_log_df"]


def save_bet_log(df: pd.DataFrame) -> None:
    clean = df[LOG_COLUMNS].copy()
    st.session_state["bet_log_df"] = clean
    try:
        clean.to_csv(LOG_FILE, index=False)
    except Exception:
        pass


def append_bet_record(b_date, plat: str, strat: str, pick: str, stake: float, odds: int, close_odds: int, res: str) -> None:
    df = get_bet_log()
    profit, clv_pct = compute_profit_and_clv(stake, odds, close_odds, res)
    row = pd.DataFrame(
        [
            {
                "Date": str(b_date),
                "Platform": plat,
                "Strategy": strat,
                "Pick": pick.strip(),
                "Stake": round(float(stake), 2),
                "Odds": int(odds),
                "Closing_Odds": int(close_odds),
                "Result": res,
                "Profit": profit,
                "CLV_Pct": clv_pct,
            }
        ],
        columns=LOG_COLUMNS,
    )
    save_bet_log(pd.concat([df, row], ignore_index=True))
    if res == "Pending":
        apply_bankroll_delta(plat, -round(float(stake), 2))
    elif res == "Win":
        apply_bankroll_delta(plat, round(profit, 2))


def parse_and_import_csv_lines(raw_text: str) -> Tuple[int, List[str]]:
    cleaned = raw_text.strip()
    if not cleaned:
        return 0, ["Paste at least one CSV line first."]
    reader = csv.reader(io.StringIO(cleaned))
    added = 0
    errors: List[str] = []
    for line_no, parts in enumerate(reader, start=1):
        if not parts or all(not p.strip() for p in parts):
            continue
        if parts[0].strip().lower() == "date" and len(parts) >= 4 and parts[1].strip().lower() == "platform":
            continue
        if len(parts) < 8:
            errors.append(f"Line {line_no}: expected at least 8 values (got {len(parts)}).")
            continue
        try:
            b_date = parts[0].strip() or str(date.today())
            plat = next((p for p in PLATFORMS if p.lower() == parts[1].strip().lower()), "DraftKings")
            strat = parts[2].strip() or "+EV Straight"
            pick = ",".join(parts[3 : len(parts) - 4]).strip() if len(parts) > 8 else parts[3].strip()
            stake = float(parts[-4].strip().replace("$", ""))
            odds = int(round(float(parts[-3].strip().replace("+", ""))))
            close_raw = parts[-2].strip().replace("+", "")
            close_odds = int(round(float(close_raw))) if close_raw else odds
            res_raw = parts[-1].strip().title()
            res = res_raw if res_raw in RESULTS else "Pending"
            append_bet_record(b_date, plat, strat, pick, stake, odds, close_odds, res)
            added += 1
        except Exception as exc:
            errors.append(f"Line {line_no}: {exc}")
    return added, errors


def pull_upcoming_fixtures_cached(selected_sport: str, days_ahead: int = 1, force_refresh: bool = False) -> None:
    if not API_KEY:
        st.error("Add ODDSPAPI_API_KEY to your .env file or Streamlit secrets first.")
        return
    cfg = st.session_state["sport_map"][selected_sport]
    sport_id = cfg["sportId"]
    today = date.today()
    cache_key = f"{sport_id}_{today.isoformat()}_{days_ahead}"
    fixtures_cache = st.session_state.setdefault("fixtures_cache", {})
    if not force_refresh and cache_key in fixtures_cache:
        st.session_state["fixtures"] = fixtures_cache[cache_key]
        st.info(f"Loaded {len(fixtures_cache[cache_key])} cached {selected_sport} games (0 API credits).")
        return
    try:
        r = requests.get(
            f"{BASE_URL}/fixtures",
            params={
                "apiKey": API_KEY,
                "sportId": sport_id,
                "from": today.isoformat(),
                "to": (today + timedelta(days=days_ahead)).isoformat(),
                "hasOdds": "true",
            },
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            fixtures = data if isinstance(data, list) else data.get("fixtures", [])
            fixtures_cache[cache_key] = fixtures
            st.session_state["fixtures"] = fixtures
            st.success(f"Loaded {len(fixtures)} upcoming {selected_sport} games (1 API credit).")
        else:
            st.error(f"API Error {r.status_code}: {r.text[:250]}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")


def main() -> None:
    init_state()
    st.title("Asymmetric Betting Hub")
    st.caption(
        "Data-First Flow: **1) Injury & Watchlist Pull** -> **2) OddsPapi Fixture & Odds Pull** -> "
        "**3) Pricing & Edge Filtering** -> **4) Bet Log & Bankroll Sync**"
    )

    with st.sidebar:
        st.header("Bankroll & Unit Rules")
        sync_on = st.checkbox("Auto-sync balances on log/settle", value=bool(st.session_state["auto_sync_enabled"]))
        st.session_state["auto_sync_enabled"] = sync_on

        live_balances: Dict[str, float] = {}
        for plat, base_val in DEFAULT_BALANCES.items():
            adj = float(st.session_state.get(f"bal_adj_{plat}", 0.0)) if sync_on else 0.0
            val = st.number_input(
                f"{plat} ($)",
                min_value=0.0,
                value=max(0.0, round(base_val + adj, 2)),
                step=1.0,
                key=f"bal_input_{plat}",
            )
            live_balances[plat] = float(val)

        total_roll = sum(live_balances.values())
        st.metric("Total Active Roll", f"${total_roll:,.2f}")
        unit_size = st.number_input("Standard Unit ($)", min_value=0.5, value=2.0, step=0.5, key="unit_input")
        max_single_bet = round(total_roll * 0.05, 2)
        st.caption(f"5% Max Single-Ticket Cap: **${max_single_bet:.2f}**")

    with st.expander("Stage 1: Injury Watchlist & OddsPapi Data Pulls", expanded=True):
        st.text_input(
            "Flagged / Out / Doubtful / Questionable Players (comma-separated)",
            placeholder="e.g., Kawhi Leonard, Tyrese Haliburton, Chet Holmgren",
            key="restricted_players_raw",
        )
        c1, c2, c3, c4 = st.columns([1.3, 1.0, 1.7, 1.0])
        s1_sport = c1.selectbox("Sport / League", list(st.session_state["sport_map"].keys()), key="s1_sport")
        s1_days = c2.selectbox("Days Ahead", [1, 2, 7], index=0, key="s1_days")
        if c3.button("Pull Upcoming Fixtures", key="s1_pull", use_container_width=True):
            pull_upcoming_fixtures_cached(s1_sport, days_ahead=s1_days, force_refresh=False)
        c4.metric("Loaded Fixtures", len(st.session_state.get("fixtures", [])))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Data & Scanner",
            "Skew & Ladders",
            "Combos & Singles",
            "Bankroll & Promos",
            "Bet Log & ROI",
        ]
    )

    with tab1:
        st.subheader("OddsPapi Scanner")
        fixtures = st.session_state.get("fixtures", [])
        if fixtures:
            st.dataframe(pd.DataFrame(fixtures).head(25), use_container_width=True, hide_index=True)
        else:
            st.info("Pull fixtures in Stage 1 above to browse upcoming matchups.")

    with tab2:
        st.subheader("Skew & Alt Ladders")
        mean_proj = st.number_input("Mean Projection", min_value=1.0, value=76.64, step=1.0, key="t2_mean")
        sd_val = st.number_input("Standard Deviation", min_value=0.5, value=42.92, step=1.0, key="t2_sd")
        sigma = math.sqrt(math.log(1.0 + (sd_val / mean_proj) ** 2))
        mu = math.log(mean_proj) - 0.5 * (sigma ** 2)
        st.metric("Log-Normal Median", f"{math.exp(mu):.1f}", f"Mean: {mean_proj:.1f}")

    with tab3:
        st.subheader("Combos & Singles")
        pick_txt = st.text_input("Pick / Slip Description", key="t3_pick")
        conflicts = find_roster_conflicts(pick_txt)
        if conflicts:
            st.error(f"ROSTER CONFLICT: {', '.join(conflicts).title()}")
        else:
            st.success("No watchlist conflicts detected.")

    with tab4:
        st.subheader("Bankroll & Promos")
        st.write(f"Active Bankroll: **${total_roll:,.2f}** | Standard Unit: **${unit_size:.2f}** | 5% Cap: **${max_single_bet:.2f}**")

    with tab5:
        st.subheader("Bet Log & ROI")
        pasted = st.text_area(
            "Quick Paste 8-field CSV Lines",
            placeholder="2026-11-20,FanDuel,Combos/Singles,Scottie Barnes Alt Points Floor (TOR @ DET),2.00,-115,-115,Pending",
            height=70,
            key="t5_paste",
        )
        if st.button("Import CSV", key="t5_import"):
            added, errs = parse_and_import_csv_lines(pasted)
            if added:
                st.success(f"Imported {added} row(s).")
                st.rerun()
            for err in errs:
                st.warning(err)

        df_log = get_bet_log()
        if not df_log.empty:
            st.dataframe(df_log, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()