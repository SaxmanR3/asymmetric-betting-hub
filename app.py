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
PRIORITY_BOOKS = [
    "pinnacle",
    "fanduel",
    "draftkings",
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

SD_MULTIPLIERS = {
    "NFL: Pass Attempts": 0.25,
    "NFL: Completions": 0.28,
    "NFL: Passing Yards": 0.31,
    "NFL: Carries": 0.39,
    "NFL: Receptions": 0.53,
    "NFL: Rush + Rec Yards": 0.50,
    "NFL: Rushing Yards (Overall)": 0.56,
    "NFL: Rushing Yards (<60 YPG)": 0.63,
    "NFL: Rushing Yards (60-80 YPG)": 0.52,
    "NFL: Rushing Yards (80+ YPG)": 0.42,
    "NFL: Receiving Yards (WR)": 0.64,
    "NFL: Receiving Yards (TE)": 0.69,
    "NBA: Points": 0.34,
    "NBA: Rebounds": 0.38,
    "NBA: Assists": 0.45,
    "NBA: PRA (Pts+Reb+Ast)": 0.27,
    "NBA: 3-Pointers Made": 0.58,
    "MLB: Pitcher Strikeouts": 0.36,
    "MLB: Hits + Runs + RBIs": 0.48,
    "MLB: Total Bases": 0.68,
    "NHL: Shots on Goal": 0.44,
    "NHL: Points (Goals+Assists)": 0.62,
    "Custom / Game-Log Driven": 0.45,
}

st.set_page_config(page_title="Asymmetric Betting Hub", layout="wide")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def amer_to_dec(odds: float) -> float:
    return (1.0 + 100.0 / abs(odds)) if odds < 0 else (1.0 + odds / 100.0)


def dec_to_amer(dec: float) -> int:
    if dec <= 1.0:
        return -10000
    return round((dec - 1.0) * 100) if dec >= 2.0 else round(-100.0 / (dec - 1.0))


def amer_to_prob(odds: float) -> float:
    return (abs(odds) / (abs(odds) + 100.0)) if odds < 0 else (100.0 / (odds + 100.0))


def prob_to_amer(p: float) -> int:
    p = min(max(p, 0.001), 0.999)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def devig_two_way(odds_a: float, odds_b: float) -> Tuple[float, float, float, float]:
    pa, pb = amer_to_prob(odds_a), amer_to_prob(odds_b)
    overround = pa + pb
    fair_a, fair_b = pa / overround, pb / overround
    if odds_a < 0 and odds_b < 0:
        width = abs(odds_a + 100) + abs(odds_b + 100)
    elif odds_a < 0 and odds_b > 0:
        width = abs(odds_a) - odds_b
    elif odds_b < 0 and odds_a > 0:
        width = abs(odds_b) - odds_a
    else:
        width = abs(odds_a - odds_b)
    return fair_a, fair_b, (overround - 1.0) * 100.0, abs(width)


def lognormal_params(mean: float, sd: float) -> Tuple[float, float]:
    safe_mean = max(mean, 1e-6)
    safe_sd = max(sd, 1e-6)
    sigma = math.sqrt(math.log(1.0 + (safe_sd / safe_mean) ** 2))
    mu = math.log(safe_mean) - 0.5 * (sigma ** 2)
    return mu, sigma


def calc_ev(true_prob: float, book_odds: float) -> float:
    return (true_prob * amer_to_dec(book_odds) - 1.0) * 100.0


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


def matches_league(fx: dict, league_filter: str) -> bool:
    if not league_filter:
        return True
    tourn = (fx.get("tournamentName") or fx.get("tournamentSlug") or "").lower()
    sport_name = (fx.get("sportName") or fx.get("sportSlug") or "").lower()
    blob = f"{tourn} {sport_name}".strip()
    if league_filter == "nba":
        return "nba" in tourn and "wnba" not in tourn
    if league_filter == "wnba":
        return "wnba" in tourn
    if league_filter == "ncaam":
        return ("ncaa" in tourn or "college" in tourn) and (
            "women" not in tourn and "ncaaw" not in tourn and "w-" not in tourn
        )
    if league_filter == "ncaaw":
        return ("ncaa" in tourn or "college" in tourn) and (
            "women" in tourn or "ncaaw" in tourn or "w-" in tourn
        )
    if league_filter == "ncaaf":
        return (
            "ncaa" in tourn
            or "college" in tourn
            or "fbs" in tourn
            or "fcs" in tourn
            or "football" in blob
        ) and (
            "basketball" not in blob
            and "baseball" not in blob
            and "hockey" not in blob
        )
    if league_filter == "mls":
        return "mls" in tourn or "major league soccer" in tourn
    return league_filter in blob


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
                df = df[LOG_COLUMNS].copy()
                df["Stake"] = pd.to_numeric(df["Stake"], errors="coerce").fillna(0.0)
                df["Odds"] = pd.to_numeric(df["Odds"], errors="coerce").fillna(-110).astype(int)
                df["Closing_Odds"] = pd.to_numeric(df["Closing_Odds"], errors="coerce").fillna(-110).astype(int)
                df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0.0)
                df["CLV_Pct"] = pd.to_numeric(df["CLV_Pct"], errors="coerce").fillna(0.0)
                st.session_state["bet_log_df"] = df
                return df
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


def append_bet_record(
    b_date,
    plat: str,
    strat: str,
    pick: str,
    stake: float,
    odds: int,
    close_odds: int,
    res: str,
) -> None:
    df = get_bet_log()
    profit, clv_pct = compute_profit_and_clv(stake, odds, close_odds, res)
    new_row = pd.DataFrame(
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
    save_bet_log(pd.concat([df, new_row], ignore_index=True))
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
    valid_results = {r.lower(): r for r in RESULTS}

    for line_no, parts in enumerate(reader, start=1):
        cells = [p.strip() for p in parts if p is not None]
        if not cells or all(c == "" for c in cells):
            continue
        if cells[0].lower() == "date" and len(cells) >= 4 and cells[1].lower() == "platform":
            continue

        try:
            res_idx = next(
                (i for i, val in enumerate(cells) if val.lower() in valid_results),
                None,
            )
            if res_idx is not None and res_idx >= 6:
                b_date = cells[0] or str(date.today())
                plat = next(
                    (p for p in PLATFORMS if p.lower() == cells[1].lower()),
                    "DraftKings",
                )
                strat = cells[2] or "+EV Straight"
                stake = float(cells[res_idx - 3].replace("$", ""))
                odds = int(round(float(cells[res_idx - 2].replace("+", ""))))
                close_raw = cells[res_idx - 1].replace("+", "")
                close_odds = int(round(float(close_raw))) if close_raw else odds
                res = valid_results[cells[res_idx].lower()]
                pick = ",".join(cells[3 : res_idx - 3]).strip()
            else:
                if len(cells) < 8:
                    errors.append(f"Line {line_no}: expected at least 8 values (got {len(cells)}).")
                    continue
                b_date = cells[0] or str(date.today())
                plat = next(
                    (p for p in PLATFORMS if p.lower() == cells[1].lower()),
                    "DraftKings",
                )
                strat = cells[2] or "+EV Straight"
                pick = ",".join(cells[3 : len(cells) - 4]).strip() if len(cells) > 8 else cells[3]
                stake = float(cells[-4].replace("$", ""))
                odds = int(round(float(cells[-3].replace("+", ""))))
                close_raw = cells[-2].replace("+", "")
                close_odds = int(round(float(close_raw))) if close_raw else odds
                res = valid_results.get(cells[-1].lower(), "Pending")

            if not pick:
                errors.append(f"Line {line_no}: Pick Description is empty.")
                continue

            append_bet_record(b_date, plat, strat, pick, stake, odds, close_odds, res)
            added += 1
        except Exception as exc:
            errors.append(f"Line {line_no}: could not parse ({exc}).")

    return added, errors


def render_quick_paste_csv(key_prefix: str) -> None:
    st.markdown("**Quick Paste CSV Line(s)** *(Paste 8- or 10-field rows and click Import)*")
    st.caption("Format: `Date,Platform,Strategy,Pick,Stake,Odds,Closing_Odds,Result`")
    p_col1, p_col2 = st.columns([4.2, 1.0])
    pasted = p_col1.text_area(
        "Paste CSV rows",
        placeholder="2026-11-20,FanDuel,Combos/Singles,Scottie Barnes Alt Points Floor (TOR @ DET),2.00,-115,-115,Pending",
        height=68,
        label_visibility="collapsed",
        key=f"{key_prefix}_csv_paste",
    )
    p_col2.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if p_col2.button("Import CSV", key=f"{key_prefix}_csv_import_btn", use_container_width=True):
        added, errors = parse_and_import_csv_lines(pasted)
        if added > 0:
            st.success(f"Imported {added} bet(s) into bet_log.csv!")
        for err in errors:
            st.warning(err)
        if added > 0:
            st.rerun()


def render_inline_bet_logger(
    key_prefix: str,
    max_single_cap: float,
    default_plat: str = "FanDuel",
    default_strat: str = "Combos/Singles",
    default_pick: str = "",
    default_stake: float = 2.0,
    default_odds: int = -110,
    expanded: bool = False,
) -> None:
    st.divider()
    with st.expander(f"Quick Bet Logger & Live Log ({len(get_bet_log())} logged)", expanded=expanded):
        render_quick_paste_csv(key_prefix)
        st.markdown("---")
        with st.form(f"{key_prefix}_quick_log_form", clear_on_submit=False):
            c1, c2, c3, c4 = st.columns([1.1, 1.2, 1.3, 2.4])
            b_date = c1.date_input("Date", value=date.today(), key=f"{key_prefix}_date")
            plat_idx = PLATFORMS.index(default_plat) if default_plat in PLATFORMS else 0
            b_plat = c2.selectbox("Platform", PLATFORMS, index=plat_idx, key=f"{key_prefix}_plat")
            strat_idx = STRATEGIES.index(default_strat) if default_strat in STRATEGIES else 0
            b_strat = c3.selectbox("Strategy", STRATEGIES, index=strat_idx, key=f"{key_prefix}_strat")
            b_pick = c4.text_input("Pick Description", value=default_pick, key=f"{key_prefix}_pick")

            c5, c6, c7, c8, c9 = st.columns([1.0, 1.0, 1.1, 1.1, 1.2])
            b_stake = c5.number_input(
                "Stake ($)",
                min_value=0.10,
                value=max(0.10, float(default_stake)),
                step=0.25,
                key=f"{key_prefix}_stake",
            )
            b_odds = c6.number_input("Bet Odds", value=int(default_odds), step=5, key=f"{key_prefix}_odds")
            b_close = c7.number_input("Closing Odds", value=int(default_odds), step=5, key=f"{key_prefix}_close")
            b_res = c8.selectbox("Result", RESULTS, key=f"{key_prefix}_res")
            c9.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            submitted = c9.form_submit_button("Log Bet", use_container_width=True)

            if submitted:
                conflicts = find_roster_conflicts(b_pick)
                if not b_pick.strip():
                    st.warning("Enter a Pick Description before logging.")
                elif conflicts:
                    st.error(
                        f"ROSTER CONFLICT: '{b_pick.strip()}' matches watchlist player(s): {', '.join(conflicts).title()}."
                    )
                else:
                    if max_single_cap > 0 and b_stake > max_single_cap:
                        st.warning(
                            f"Stake ${b_stake:.2f} exceeds 5% cap (${max_single_cap:.2f}); logged anyway for tracking."
                        )
                    append_bet_record(b_date, b_plat, b_strat, b_pick, b_stake, b_odds, b_close, b_res)
                    st.success(f"Logged: {b_pick.strip()} (${b_stake:.2f} @ {int(b_odds):+d}) on {b_plat}")
                    st.rerun()

        df_now = get_bet_log()
        if not df_now.empty:
            st.caption("Recent Entries (Full history, settlement & CSV backup in Bet Log & ROI)")
            st.dataframe(df_now.tail(5).iloc[::-1], use_container_width=True, hide_index=True)


def load_market_catalog(sport_id: int) -> Dict[str, dict]:
    cache_key = f"market_catalog_{sport_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    cache_file = f"markets_cache_{sport_id}.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            st.session_state[cache_key] = catalog
            return catalog
        except Exception:
            pass

    if not API_KEY:
        return {}

    try:
        r = requests.get(
            f"{BASE_URL}/markets",
            params={"apiKey": API_KEY, "sportId": sport_id, "language": "en"},
            timeout=20,
        )
        if r.status_code != 200:
            return {}

        raw = r.json()
        catalog: Dict[str, dict] = {}
        for m in raw:
            mid = str(m.get("marketId"))
            outcomes = {
                str(o.get("outcomeId")): o.get("outcomeName", str(o.get("outcomeId")))
                for o in m.get("outcomes", [])
            }
            catalog[mid] = {
                "name": m.get("marketName") or f"Market {mid}",
                "playerProp": bool(m.get("playerProp", False)),
                "handicap": m.get("handicap"),
                "period": m.get("period", ""),
                "marketLength": m.get("marketLength", len(outcomes)),
                "outcomes": outcomes,
            }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(catalog, f)
        except Exception:
            pass

        st.session_state[cache_key] = catalog
        return catalog
    except Exception:
        return {}


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
        st.session_state["fixtures_sport_id"] = sport_id
        st.session_state["fixtures_sport_label"] = selected_sport
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
            raw_fixtures = data if isinstance(data, list) else data.get("fixtures", [])
            league_filter = cfg.get("filter", "")
            fixtures = [fx for fx in raw_fixtures if matches_league(fx, league_filter)]
            fixtures_cache[cache_key] = fixtures
            st.session_state["fixtures"] = fixtures
            st.session_state["fixtures_sport_id"] = sport_id
            st.session_state["fixtures_sport_label"] = selected_sport
            st.success(f"Loaded {len(fixtures)} upcoming {selected_sport} games (1 API credit).")
        else:
            st.error(f"API Error {r.status_code}: {r.text[:250]}")
    except Exception as exc:
        st.error(f"Connection error: {exc}")


def parse_american_price(node: dict) -> Optional[int]:
    if not isinstance(node, dict) or not node.get("active", False):
        return None
    am_str = node.get("priceAmerican")
    if am_str is not None:
        try:
            am_val = int(round(float(am_str)))
            if am_val <= -100 or am_val >= 100:
                return am_val
        except (ValueError, TypeError):
            pass
    dec = node.get("price")
    if dec is not None:
        try:
            dec_f = float(dec)
            if dec_f > 1.01:
                return dec_to_amer(dec_f)
        except (ValueError, TypeError):
            pass
    return None


def scan_fixture_ev(
    b_odds: dict,
    catalog: dict,
    target_books: list,
    max_width: int,
    min_ev: float,
) -> Tuple[List[dict], List[dict]]:
    normalized_odds = {str(k).lower(): v for k, v in (b_odds or {}).items()}
    candidate_pairs = set()

    for ref_bk in ("pinnacle", "fanduel"):
        markets = normalized_odds.get(ref_bk, {}).get("markets", {})
        for mid, mdata in markets.items():
            outcomes = mdata.get("outcomes", {})
            if len(outcomes) != 2:
                continue
            for odata in outcomes.values():
                for pid, pnode in odata.get("players", {}).items():
                    if isinstance(pnode, dict) and pnode.get("active", False):
                        candidate_pairs.add((str(mid), str(pid)))

    ev_rows: List[dict] = []
    pp_rows: List[dict] = []

    for mid, pid in candidate_pairs:
        meta = catalog.get(mid, {})
        is_prop = meta.get("playerProp", pid != "0") or (pid != "0")
        m_name = meta.get("name", f"Market {mid}")
        out_names = meta.get("outcomes", {})

        ref_order = ["fanduel", "pinnacle"] if is_prop else ["pinnacle", "fanduel"]
        ref_book = None
        ref_sides = None
        player_name = ""

        for candidate_ref in ref_order:
            mdata = normalized_odds.get(candidate_ref, {}).get("markets", {}).get(mid)
            if not mdata:
                continue
            outcomes = mdata.get("outcomes", {})
            if len(outcomes) != 2:
                continue
            oids = list(outcomes.keys())
            n0 = outcomes[oids[0]].get("players", {}).get(pid)
            n1 = outcomes[oids[1]].get("players", {}).get(pid)
            a0 = parse_american_price(n0)
            a1 = parse_american_price(n1)
            if a0 is not None and a1 is not None:
                ref_book = candidate_ref
                ref_sides = (oids[0], a0, oids[1], a1)
                if is_prop:
                    player_name = (
                        n0.get("playerName")
                        or n1.get("playerName")
                        or f"Player {pid}"
                    )
                break

        if not ref_book or not ref_sides:
            continue

        if is_prop and find_roster_conflicts(player_name):
            continue

        oid0, odds0, oid1, odds1 = ref_sides
        fair_p0, fair_p1, _, width_c = devig_two_way(odds0, odds1)
        if width_c > max_width:
            continue

        label0 = out_names.get(oid0, f"Outcome {oid0}")
        label1 = out_names.get(oid1, f"Outcome {oid1}")
        sides_info = [
            (oid0, label0, fair_p0, odds0, odds1),
            (oid1, label1, fair_p1, odds1, odds0),
        ]

        if is_prop:
            for _, s_label, f_prob, r_this, r_opp in sides_info:
                edge_6 = (f_prob - 0.5421) * 100.0
                edge_5 = (f_prob - 0.5425) * 100.0
                if edge_6 > 0:
                    pp_rows.append(
                        {
                            "Player": player_name,
                            "Market": m_name,
                            "Side": s_label,
                            "Ref Book": fmt_book(ref_book),
                            "Ref Line": f"{r_this:+d} / {r_opp:+d}",
                            "True Prob %": round(f_prob * 100.0, 2),
                            "Width (c)": int(round(width_c)),
                            "6-Flex Edge %": round(edge_6, 2),
                            "5-Flex Edge %": round(edge_5, 2),
                        }
                    )

        for tb in target_books:
            tb_key = tb.lower()
            if tb_key == ref_book or tb_key not in normalized_odds:
                continue
            tb_m = normalized_odds[tb_key].get("markets", {}).get(mid)
            if not tb_m:
                continue
            tb_outs = tb_m.get("outcomes", {})
            for oid, s_label, f_prob, r_this, r_opp in sides_info:
                pnode = tb_outs.get(oid, {}).get("players", {}).get(pid)
                soft_am = parse_american_price(pnode)
                if soft_am is None:
                    continue
                ev_pct = calc_ev(f_prob, soft_am)
                if ev_pct >= min_ev:
                    sel_display = f"{player_name} — {s_label}" if is_prop else s_label
                    ev_rows.append(
                        {
                            "Type": "Prop" if is_prop else "Main",
                            "Market": m_name,
                            "Selection": sel_display,
                            "Book": fmt_book(tb_key),
                            "Book Odds": f"{soft_am:+d}",
                            "Book_Odds_Int": int(soft_am),
                            "Sharp Ref": f"{fmt_book(ref_book)} ({r_this:+d}/{r_opp:+d})",
                            "Fair Odds": f"{prob_to_amer(f_prob):+d}",
                            "True %": round(f_prob * 100.0, 1),
                            "Width (c)": int(round(width_c)),
                            "EV %": round(ev_pct, 2),
                        }
                    )

    ev_rows.sort(key=lambda x: x["EV %"], reverse=True)
    pp_rows.sort(key=lambda x: x["True Prob %"], reverse=True)
    return ev_rows, pp_rows


def main() -> None:
    init_state()
    st.title("Asymmetric Betting Hub")
    st.caption(
        "Data-First Flow: **1) Injury & Watchlist Pull** -> **2) OddsPapi Fixture & Odds Pull** -> "
        "**3) Pricing & Edge Filtering** -> **4) Bet Log & Bankroll Sync**"
    )

    with st.sidebar:
        st.header("Bankroll & Unit Rules")
        sync_on = st.checkbox(
            "Auto-sync balances on log/settle",
            value=bool(st.session_state["auto_sync_enabled"]),
            key="auto_sync_chk",
        )
        st.session_state["auto_sync_enabled"] = sync_on

        live_balances: Dict[str, float] = {}
        for plat, base_val in DEFAULT_BALANCES.items():
            adj = (
                float(st.session_state.get(f"bal_adj_{plat}", 0.0))
                if sync_on
                else 0.0
            )
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

        df_side = get_bet_log()
        pending_risk = (
            float(df_side[df_side["Result"] == "Pending"]["Stake"].sum())
            if not df_side.empty
            else 0.0
        )
        st.metric(
            "Pending Risk Exposure",
            f"${pending_risk:,.2f}",
            f"{(pending_risk / total_roll * 100.0) if total_roll > 0 else 0.0:.1f}% of roll",
        )

        st.divider()
        if API_KEY:
            st.success("OddsPapi Key Loaded")
        else:
            st.warning("No ODDSPAPI_API_KEY found")

    with st.expander("Stage 1: Injury Watchlist & OddsPapi Data Pulls", expanded=True):
        st.text_input(
            "Flagged / Out / Doubtful / Questionable Players (comma-separated, auto-blocks conflicts)",
            placeholder="e.g., Kawhi Leonard, Tyrese Haliburton, Chet Holmgren, Kenneth Walker III",
            key="restricted_players_raw",
        )
        d1, d2, d3, d4 = st.columns([1.3, 1.0, 1.7, 1.0])
        stage1_sport = d1.selectbox("Sport / League", list(st.session_state["sport_map"].keys()), key="stage1_sport")
        stage1_days = d2.selectbox("Days Ahead", [1, 2, 7], index=0, key="stage1_days")
        if d3.button("Pull Upcoming Fixtures", key="stage1_pull_fx", use_container_width=True):
            pull_upcoming_fixtures_cached(stage1_sport, days_ahead=stage1_days, force_refresh=False)
        d4.metric("Loaded Fixtures", len(st.session_state.get("fixtures", [])))

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
        st.subheader("Live OddsPapi Feed + Auto Split-Devig & PrizePicks Scanner")
        st.caption(
            "Main Lines -> Devig to **Pinnacle** | Player Props -> Devig to **FanDuel** | "
            "PP 5-Flex BE: **54.25% (-119)** | 6-Flex BE: **54.21% (-118)**"
        )

        st.session_state.setdefault("t2_width_input", 25)
        st.session_state.setdefault("t2_ev_input", 2.0)
        st.session_state.setdefault(
            "t2_books_ms", ["draftkings", "fanduel", "kalshi", "betmgm", "fanatics"]
        )

        pc_a, pc_b, pc_c = st.columns(3)
        if pc_a.button("Preset: Tight Lines (<=15c, >=3% EV)", use_container_width=True):
            st.session_state["t2_width_input"] = 15
            st.session_state["t2_ev_input"] = 3.0
            st.rerun()
        if pc_b.button("Preset: Standard (<=25c, >=2% EV)", use_container_width=True):
            st.session_state["t2_width_input"] = 25
            st.session_state["t2_ev_input"] = 2.0
            st.rerun()
        if pc_c.button("Select All 5 My Books", use_container_width=True):
            st.session_state["t2_books_ms"] = [
                "draftkings",
                "fanduel",
                "kalshi",
                "betmgm",
                "fanatics",
            ]
            st.rerun()

        col_a, col_b, col_c = st.columns(3)
        selected_sport = col_a.selectbox("Sport / League", list(st.session_state["sport_map"].keys()), key="t1_scan_sport")
        max_width = col_b.number_input("Max Market Width (cents)", min_value=5, max_value=60, key="t2_width_input")
        min_ev = col_c.number_input("Min EV % to Flag", min_value=0.0, max_value=20.0, step=0.5, key="t2_ev_input")
        selected_books = st.multiselect("Target Books to Shop", TARGET_SOFT_BOOKS, format_func=fmt_book, key="t2_books_ms")

        t1_prefill_plat = "DraftKings"
        t1_prefill_strat = "+EV Straight"
        t1_prefill_pick = ""
        t1_prefill_odds = -110

        if st.session_state.get("fixtures"):
            search_txt = st.text_input("Filter by Team (optional, 0 credits)", value="", key="t1_team_flt").strip().lower()
            fx_options: Dict[str, str] = {}
            for fx in st.session_state["fixtures"]:
                fid = fx.get("fixtureId")
                if not fid:
                    continue
                p1_name = fx.get("participant1Name", "Away")
                p2_name = fx.get("participant2Name", "Home")
                tourn = fx.get("tournamentName") or fx.get("tournamentSlug") or ""
                stime = fx.get("startTime", "")[:16].replace("T", " ")
                label = f"[{tourn}] {p1_name} vs {p2_name} ({stime})" if tourn else f"{p1_name} vs {p2_name} ({stime})"
                if not search_txt or search_txt in label.lower():
                    fx_options[label] = fid

            if fx_options:
                chosen_label = st.selectbox(f"Select Game to Scan ({len(fx_options)} found)", list(fx_options.keys())[:100])
                if st.button("Scan & Find +EV / PrizePicks Plays (1 API Credit)"):
                    if not API_KEY:
                        st.error("Add ODDSPAPI_API_KEY first.")
                    else:
                        fid = fx_options[chosen_label]
                        sport_id = st.session_state.get(
                            "fixtures_sport_id",
                            st.session_state["sport_map"][selected_sport]["sportId"],
                        )
                        try:
                            catalog = load_market_catalog(sport_id)
                            r = requests.get(
                                f"{BASE_URL}/odds",
                                params={"apiKey": API_KEY, "fixtureId": fid},
                                timeout=25,
                            )
                            if r.status_code == 200:
                                st.session_state["last_scan_game"] = chosen_label
                                st.session_state["last_b_odds"] = r.json().get("bookmakerOdds", {})
                                st.session_state["last_catalog"] = catalog
                            else:
                                st.error(f"API Error {r.status_code}: {r.text[:250]}")
                        except Exception as exc:
                            st.error(f"Error fetching odds: {exc}")

            if st.session_state.get("last_b_odds"):
                b_odds = st.session_state["last_b_odds"]
                catalog = st.session_state.get("last_catalog", {})
                ev_rows, pp_rows = scan_fixture_ev(b_odds, catalog, selected_books, max_width, min_ev)
                scan_game = st.session_state.get("last_scan_game", "")
                st.markdown(f"### Results: {scan_game}")
                key_books = [b for b in PRIORITY_BOOKS if b in {str(k).lower() for k in b_odds}]
                st.caption(f"Live sharp/target books in this scan: **{', '.join(fmt_book(b) for b in key_books)}**")

                if ev_rows:
                    st.dataframe(pd.DataFrame(ev_rows).drop(columns=["Book_Odds_Int"]), use_container_width=True, hide_index=True)
                    ev_choices = [
                        f"{row['Book']} | {row['Market']}: {row['Selection']} ({row['Book Odds']}) [+{row['EV %']}% EV]"
                        for row in ev_rows
                    ]
                    chosen_ev_idx = st.selectbox("Select +EV Play to Pre-Fill Bet Log", range(len(ev_choices)), format_func=lambda i: ev_choices[i])
                    picked_row = ev_rows[chosen_ev_idx]
                    t1_prefill_plat = picked_row["Book"] if picked_row["Book"] in PLATFORMS else "DraftKings"
                    t1_prefill_strat = "+EV Straight"
                    t1_prefill_pick = f"{picked_row['Market']}: {picked_row['Selection']} ({scan_game.split('(')[0].strip()})"
                    t1_prefill_odds = picked_row["Book_Odds_Int"]

                if pp_rows:
                    st.dataframe(pd.DataFrame(pp_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Quick Manual Devig & PrizePicks Edge Checker (0 API Credits)")
        qc1, qc2, qc3, qc4 = st.columns(4)
        sharp_over = qc1.number_input("Sharp/FD Over Odds", value=-135, step=1)
        sharp_under = qc2.number_input("Sharp/FD Under Odds", value=105, step=1)
        target_book_odds = qc3.number_input("Soft Book Over Odds", value=-115, step=1)
        pp_mode = qc4.selectbox("Compare vs. PrizePicks?", ["6-Pick Flex (54.21%)", "5-Pick Flex (54.25%)"])

        fair_o, _, vig_pct, width_c = devig_two_way(sharp_over, sharp_under)
        soft_ev = calc_ev(fair_o, target_book_odds)
        pp_be = 0.5421 if "6-Pick" in pp_mode else 0.5425

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("True Fair Over %", f"{fair_o*100:.2f}%", f"Fair {prob_to_amer(fair_o):+d}")
        r2.metric("Market Width", f"{width_c:.0f} cents", f"Vig: {vig_pct:.2f}%")
        r3.metric("Soft Book EV", f"{soft_ev:+.2f}%")
        r4.metric("PrizePicks Edge", f"{(fair_o - pp_be)*100:+.2f}%", "PLAY" if fair_o > pp_be else "PASS")

        render_inline_bet_logger(
            key_prefix="tab1_scanner",
            max_single_cap=max_single_bet,
            default_plat=t1_prefill_plat,
            default_strat=t1_prefill_strat,
            default_pick=t1_prefill_pick,
            default_stake=min(unit_size, max_single_bet) if max_single_bet > 0 else unit_size,
            default_odds=t1_prefill_odds,
        )

    with tab2:
        st.subheader("Layer 3: Log-Normal Right Skew & Alt-Ladder Pricer")
        lc1, lc2 = st.columns(2)
        with lc1:
            player_label = st.text_input("Player Name (optional, auto-fills Bet Log)", placeholder="e.g., Derrick Henry")
            prop_type = st.selectbox("Prop Type (SD Reference)", list(SD_MULTIPLIERS.keys()), index=6)
            proj_mean = st.number_input("Your Mean Projection", min_value=1.0, value=76.64, step=1.0)
            main_line = st.number_input("Sportsbook Main Line", min_value=0.5, value=70.5, step=0.5)
            main_odds = st.number_input("Main Line Over Odds", value=-114, step=1)
        with lc2:
            raw_logs = st.text_input("Optional: Player Game Logs (comma-separated)", value="")
            if raw_logs.strip():
                try:
                    vals = [float(x.strip()) for x in raw_logs.split(",") if x.strip()]
                    sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else proj_mean * SD_MULTIPLIERS[prop_type]
                except ValueError:
                    sd_val = proj_mean * SD_MULTIPLIERS[prop_type]
            else:
                sd_val = proj_mean * SD_MULTIPLIERS[prop_type]
            st.info(f"Active SD = **{sd_val:.2f}**")

        p_sym = 1.0 - norm_cdf((main_line - proj_mean) / max(sd_val, 1e-6))
        mu, sigma = lognormal_params(proj_mean, sd_val)
        skew_median = math.exp(mu)
        p_skew = 1.0 - norm_cdf((math.log(max(main_line, 1e-6)) - mu) / max(sigma, 1e-6))

        s1, s2, s3 = st.columns(3)
        s1.metric("Symmetric Over % (Naive)", f"{p_sym*100:.1f}%", f"EV: {calc_ev(p_sym, main_odds):+.1f}%")
        s2.metric("Skewed Median (Typical Game)", f"{skew_median:.1f}", f"vs Mean {proj_mean:.1f}")
        s3.metric("Right-Skewed Over % (Honest)", f"{p_skew*100:.1f}%", f"EV: {calc_ev(p_skew, main_odds):+.1f}%")

        step_size = 10.0 if proj_mean >= 40 else 5.0
        base_rung = max(step_size, math.floor((proj_mean * 0.55) / step_size) * step_size) - 0.5
        rungs = [base_rung + i * step_size for i in range(7)]
        ladder_rows = [
            {
                "Alt Line": f"{r + 0.5:.0f}+ ({r:.1f})",
                "Skewed P(Over)": f"{(1.0 - norm_cdf((math.log(max(r, 1e-6)) - mu) / max(sigma, 1e-6)))*100:.1f}%",
                "Fair Over Odds": f"{prob_to_amer(1.0 - norm_cdf((math.log(max(r, 1e-6)) - mu) / max(sigma, 1e-6))):+d}",
            }
            for r in rungs
        ]
        st.dataframe(pd.DataFrame(ladder_rows), use_container_width=True, hide_index=True)

        selected_rung = st.selectbox("Select Rung to Pre-Fill Bet Log", [row["Alt Line"] for row in ladder_rows], index=2)
        p_prefix = f"{player_label.strip()} " if player_label.strip() else ""
        t2_pick = f"{p_prefix}{selected_rung} {prop_type}".strip()
        if find_roster_conflicts(t2_pick):
            st.error("ROSTER CONFLICT detected against your watchlist.")

        render_inline_bet_logger(
            key_prefix="tab2_skew",
            max_single_cap=max_single_bet,
            default_plat="DraftKings",
            default_strat="Alt Ladder",
            default_pick=t2_pick,
            default_stake=min(unit_size, max_single_bet) if max_single_bet > 0 else unit_size,
            default_odds=int(main_odds),
        )

    with tab3:
        st.subheader("Combos & Singles Guardrail (Per-Leg Hit Rate)")
        c1, c2 = st.columns(2)
        with c1:
            num_legs = st.radio("Number of Floor Legs", [1, 2, 3], horizontal=True, index=2)
            p1 = st.slider("Leg 1 Hit Rate (%)", 55, 95, 80) / 100.0
            p2 = (st.slider("Leg 2 Hit Rate (%)", 55, 95, 80) / 100.0) if num_legs >= 2 else 1.0
            p3 = (st.slider("Leg 3 Hit Rate (%)", 55, 95, 80) / 100.0) if num_legs >= 3 else 1.0
        with c2:
            t3_plat = st.selectbox("Target Book", PLATFORMS, index=0, key="t3_plat_pick")
            posted_odds = st.number_input("Posted Ticket Odds (American)", value=-115, step=5)
            t3_desc = st.text_input("Combo / Single Description (auto-fills Bet Log)")
            injury_clear = st.checkbox("All players confirmed ACTIVE (No Q/D/O tags or blowout traps)", value=False)

        conflicts_t3 = find_roster_conflicts(t3_desc)
        comb_prob = p1 * p2 * p3
        fair_odds = prob_to_amer(comb_prob)
        ev_pct = calc_ev(comb_prob, posted_odds)

        m1, m2, m3 = st.columns(3)
        m1.metric("Combined Hit Rate", f"{comb_prob*100:.1f}%")
        m2.metric("Fair Breakeven Odds", f"{fair_odds:+d}")
        m3.metric("Estimated EV", f"{ev_pct:+.1f}%")

        straight_stake = min(unit_size, max_single_bet) if max_single_bet > 0 else unit_size
        kicker_stake = (
            min(round(straight_stake * 0.25, 2), max_single_bet)
            if max_single_bet > 0
            else round(straight_stake * 0.25, 2)
        )

        if conflicts_t3:
            st.error(f"GATE LOCKED — ROSTER CONFLICT: {', '.join(conflicts_t3).title()}.")
        elif not injury_clear:
            st.error("GATE LOCKED: Verify official injury reports and spread/blowout risk before firing.")
        elif ev_pct < 0:
            st.warning(f"JUICE LEAK: Posted {posted_odds:+d} vs fair {fair_odds:+d}.")
        else:
            st.success(f"GREEN LIGHT (+{ev_pct:.1f}% EV). Straight: **${straight_stake:.2f}** | Kicker: **${kicker_stake:.2f}**.")

        render_inline_bet_logger(
            key_prefix="tab3_combos",
            max_single_cap=max_single_bet,
            default_plat=t3_plat,
            default_strat="Combos/Singles",
            default_pick=t3_desc,
            default_stake=kicker_stake if num_legs > 1 else straight_stake,
            default_odds=int(posted_odds),
        )

    with tab4:
        st.subheader("Promo Converter & Fractional Kelly Sizer")
        pc1, pc2 = st.columns(2)
        with pc1:
            k_prob = st.slider("Your True Win Prob (%)", 10, 90, 56) / 100.0
            k_odds = st.number_input("Posted Odds", value=100, step=5)
            kelly_frac = st.select_slider("Kelly Fraction", options=[0.25, 0.5, 1.0], value=0.25)
            b = amer_to_dec(k_odds) - 1.0
            full_k = max(0.0, (b * k_prob - (1.0 - k_prob)) / max(b, 1e-6))
            rec_stake = round(total_roll * full_k * kelly_frac, 2)
            st.metric("Recommended Stake", f"${rec_stake:.2f}", f"Full Kelly: {full_k*100:.2f}%")
        with pc2:
            promo_type = st.selectbox("Promo Type", ["Bonus Bet (Stake Not Returned)", "Site Credit (Stake Returned)"])
            promo_amt = st.number_input("Promo Amount ($)", min_value=5.0, value=50.0, step=5.0)
            promo_odds = st.number_input("Promo Side Odds (+)", min_value=100, value=300, step=10)
            hedge_odds = st.number_input("Hedge Side Odds (-)", max_value=-100, value=-350, step=10)
            p_dec = amer_to_dec(promo_odds)
            h_dec = amer_to_dec(hedge_odds)
            gross_win = promo_amt * (p_dec - 1.0) if "Bonus Bet" in promo_type else promo_amt * p_dec
            hedge_stake = gross_win / h_dec
            guaranteed_profit = gross_win - hedge_stake
            st.metric("Hedge Stake Needed", f"${hedge_stake:.2f}")
            st.metric("Guaranteed Cash Profit", f"${guaranteed_profit:.2f}")

    with tab5:
        st.subheader("Master Bet Log, Strategy ROI & Closing Line Value (CLV)")
        df_log = get_bet_log()

        bk1, bk2 = st.columns([1.2, 2.8])
        bk1.download_button(
            label="Download bet_log.csv Backup",
            data=df_log.to_csv(index=False).encode("utf-8"),
            file_name=f"bet_log_backup_{date.today().isoformat()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        uploaded_csv = bk2.file_uploader("Restore / Merge Backup CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_csv is not None:
            try:
                restored = pd.read_csv(uploaded_csv)
                if set(LOG_COLUMNS).issubset(set(restored.columns)):
                    save_bet_log(restored[LOG_COLUMNS])
                    st.success(f"Restored {len(restored)} bets from backup CSV!")
                    st.rerun()
                else:
                    st.error("CSV columns do not match bet_log.csv format.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

        with st.expander("Quick Paste CSV Line(s) — Bulk Import Without Typing", expanded=True):
            render_quick_paste_csv("tab5")

        df_log = get_bet_log()
        if not df_log.empty:
            settled = df_log[df_log["Result"].isin(["Win", "Loss", "Push", "Void"])]
            decided = df_log[df_log["Result"].isin(["Win", "Loss"])]
            tot_staked = decided["Stake"].sum()
            tot_profit = settled["Profit"].sum()
            roi = (tot_profit / tot_staked * 100.0) if tot_staked > 0 else 0.0
            avg_clv = df_log["CLV_Pct"].mean()

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Net Profit", f"${tot_profit:+.2f}")
            t2.metric("Overall ROI", f"{roi:+.1f}%")
            t3.metric("Avg CLV", f"{avg_clv:+.2f}%")
            t4.metric("Pending Tickets", f"{len(df_log[df_log['Result'] == 'Pending'])}")

            edited_df = st.data_editor(
                df_log,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "Date": st.column_config.TextColumn("Date"),
                    "Platform": st.column_config.SelectboxColumn("Platform", options=PLATFORMS),
                    "Strategy": st.column_config.SelectboxColumn("Strategy", options=STRATEGIES),
                    "Pick": st.column_config.TextColumn("Pick"),
                    "Stake": st.column_config.NumberColumn("Stake", min_value=0.10, step=0.25, format="$%.2f"),
                    "Odds": st.column_config.NumberColumn("Odds", step=5),
                    "Closing_Odds": st.column_config.NumberColumn("Closing_Odds", step=5),
                    "Result": st.column_config.SelectboxColumn("Result", options=RESULTS),
                    "Profit": st.column_config.NumberColumn("Profit", format="$%.2f"),
                    "CLV_Pct": st.column_config.NumberColumn("CLV_Pct", format="%.2f%%"),
                },
                key="master_log_editor",
            )
            if st.button("Save Ledger Changes"):
                merged_df = df_log.copy()
                for idx, row in edited_df.iterrows():
                    try:
                        st_val = float(row["Stake"])
                        od_val = int(round(float(row["Odds"])))
                        cl_val = int(round(float(row["Closing_Odds"])))
                        res_val = str(row["Result"])
                        prof_val, clv_val = compute_profit_and_clv(st_val, od_val, cl_val, res_val)
                        if idx in merged_df.index:
                            merged_df.at[idx, "Date"] = str(row["Date"])
                            merged_df.at[idx, "Platform"] = str(row["Platform"])
                            merged_df.at[idx, "Strategy"] = str(row["Strategy"])
                            merged_df.at[idx, "Pick"] = str(row["Pick"])
                            merged_df.at[idx, "Stake"] = round(st_val, 2)
                            merged_df.at[idx, "Odds"] = od_val
                            merged_df.at[idx, "Closing_Odds"] = cl_val
                            merged_df.at[idx, "Result"] = res_val
                            merged_df.at[idx, "Profit"] = prof_val
                            merged_df.at[idx, "CLV_Pct"] = clv_val
                    except Exception:
                        pass
                save_bet_log(merged_df)
                st.success("Ledger updated and saved to bet_log.csv!")
                st.rerun()


if __name__ == "__main__":
    main()