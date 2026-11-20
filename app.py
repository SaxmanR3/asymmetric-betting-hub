import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Asymmetric Betting Hub",
    page_icon="🏀",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "bet_log.csv"
CACHE_FILES = [
    BASE_DIR / "markets_cache_11.json",
    BASE_DIR / "markets_cache_14.json",
]

DEFAULT_BANKROLL = 54.92
MAX_BET_PCT = 0.05


def get_api_key() -> str | None:
    try:
        if "ODDSPAPI_API_KEY" in st.secrets:
            val = str(st.secrets["ODDSPAPI_API_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass
    val = os.getenv("ODDSPAPI_API_KEY", "").strip()
    return val or None


def american_to_implied_prob(odds: float) -> float:
    if odds == 0:
        return 0.0
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        return 1.0
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def fractional_kelly_stake(
    win_prob: float,
    american_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_cap: float = 2.75,
) -> tuple[float, float]:
    dec_odds = american_to_decimal(american_odds)
    b = dec_odds - 1.0
    if b <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0, 0.0
    q = 1.0 - win_prob
    full_kelly_pct = ((b * win_prob) - q) / b
    if full_kelly_pct <= 0:
        return 0.0, full_kelly_pct * 100.0
    raw_stake = bankroll * full_kelly_pct * kelly_fraction
    capped_stake = min(raw_stake, max_cap, bankroll)
    return round(max(capped_stake, 0.0), 2), round(full_kelly_pct * 100.0, 2)


def load_local_cache_summary() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in CACHE_FILES:
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                count = (
                    len(raw)
                    if isinstance(raw, list)
                    else len(raw.keys())
                    if isinstance(raw, dict)
                    else 1
                )
                summaries.append(
                    {
                        "file": path.name,
                        "size_kb": round(path.stat().st_size / 1024, 1),
                        "records": count,
                    }
                )
            except Exception as exc:
                summaries.append({"file": path.name, "error": str(exc)})
    return summaries


def fetch_oddspapi_fixtures(
    api_key: str, sport: str = "basketball_nba"
) -> tuple[bool, Any]:
    url = "https://api.oddspapi.io/v4/fixtures"
    params = {"apiKey": api_key, "sport": sport}
    try:
        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as exc:
        return False, str(exc)


def load_bet_log() -> pd.DataFrame:
    cols = [
        "timestamp",
        "slate",
        "player_or_market",
        "line_or_tier",
        "american_odds",
        "floor_prob_pct",
        "kelly_frac",
        "stake_usd",
        "status",
    ]
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(LOG_FILE)
        for col in cols:
            if col not in df.columns:
                df[col] = None
        return df[cols]
    except Exception:
        return pd.DataFrame(columns=cols)


def append_bet_log(row: dict[str, Any]) -> None:
    df = load_bet_log()
    updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(LOG_FILE, index=False)


api_key = get_api_key()

st.title("Asymmetric Betting Hub")
st.caption(
    "NBA Cup Slate · Friday, Nov 20, 2026 · 80% Floor Engine · Layer 3 Skew Ladder · Fractional Kelly"
)

with st.sidebar:
    st.header("Bankroll & Risk Controls")
    bankroll = st.number_input(
        "Active Bankroll ($)",
        min_value=0.0,
        value=DEFAULT_BANKROLL,
        step=0.50,
        format="%.2f",
    )
    max_cap_pct = st.slider(
        "Max Single-Ticket Cap (%)",
        min_value=1.0,
        max_value=10.0,
        value=MAX_BET_PCT * 100,
        step=0.5,
    )
    max_bet_cap = round(bankroll * (max_cap_pct / 100.0), 2)
    kelly_frac = st.selectbox(
        "Fractional Kelly",
        options=[0.10, 0.25, 0.50, 1.0],
        index=1,
        format_func=lambda x: f"{int(x * 100)}% Kelly",
    )

    st.metric("Max Single Ticket", f"${max_bet_cap:.2f}")

    st.divider()
    st.subheader("OddsPapi Connection")
    if api_key:
        st.success(
            f"API Key Loaded ({api_key[:4]}...{api_key[-4:]})"
            if len(api_key) >= 8
            else "API Key Loaded"
        )
    else:
        st.warning("No ODDSPAPI_API_KEY found in Streamlit Secrets or .env")

tab_engine, tab_ladder, tab_scanner, tab_log = st.tabs(
    [
        "80% Floor & Kelly Sizer",
        "Layer 3 Skew / Alt Ladder",
        "OddsPapi & Cache Scanner",
        "ROI & Bet Log",
    ]
)

with tab_engine:
    st.subheader("80% Floor & Fractional Kelly Calculator")
    col1, col2, col3 = st.columns(3)
    with col1:
        market_label = st.text_input("Player / Market", value="NBA Cup Player Prop")
        est_floor_prob = st.slider(
            "Estimated Floor Hit Prob (%)",
            min_value=50.0,
            max_value=99.0,
            value=82.0,
            step=0.5,
        )
    with col2:
        american_odds = st.number_input(
            "American Odds", min_value=-2000, max_value=2000, value=-220, step=5
        )
        slate_label = st.text_input("Slate", value="NBA Cup Fri 2026-11-20")
    with col3:
        implied_prob = american_to_implied_prob(float(american_odds)) * 100.0
        edge_pct = est_floor_prob - implied_prob
        passes_floor = est_floor_prob >= 80.0
        rec_stake, full_kelly_pct = fractional_kelly_stake(
            win_prob=est_floor_prob / 100.0,
            american_odds=float(american_odds),
            bankroll=float(bankroll),
            kelly_fraction=float(kelly_frac),
            max_cap=max_bet_cap,
        )
        st.metric("Implied Prob", f"{implied_prob:.1f}%")
        st.metric("Raw Edge", f"{edge_pct:+.1f}%")
        st.metric("Recommended Stake", f"${rec_stake:.2f}")

    if passes_floor:
        st.success(
            f"PASSES 80% FLOOR ({est_floor_prob:.1f}% >= 80.0%) · Full Kelly: {full_kelly_pct:.1f}%"
        )
    else:
        st.error(
            f"BELOW 80% FLOOR ({est_floor_prob:.1f}% < 80.0%) · Pass or lower alt line"
        )

    if st.button("Log Candidate to Bet Tracker", type="primary"):
        append_bet_log(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "slate": slate_label,
                "player_or_market": market_label,
                "line_or_tier": "Floor / Single",
                "american_odds": american_odds,
                "floor_prob_pct": est_floor_prob,
                "kelly_frac": kelly_frac,
                "stake_usd": rec_stake,
                "status": "OPEN",
            }
        )
        st.success("Logged to bet_log.csv")

with tab_ladder:
    st.subheader("Layer 3 Skew / Alt Ladder")
    base_line = st.number_input(
        "Base Projection / Line", min_value=0.5, max_value=100.0, value=20.0, step=0.5
    )
    step_size = st.number_input(
        "Ladder Step Increment", min_value=0.5, max_value=25.0, value=2.5, step=0.5
    )
    ladder_rows = []
    for idx, mult in enumerate([0.75, 0.85, 0.95, 1.0, 1.10], start=1):
        alt_line = round(max(0.5, base_line * mult), 1)
        alt_odds = -260 if mult <= 0.85 else (-160 if mult <= 1.0 else +135)
        est_p = min(95.0, max(35.0, 88.0 - (idx - 1) * 11.0))
        stake_val, _ = fractional_kelly_stake(
            est_p / 100.0, alt_odds, bankroll, kelly_frac, max_bet_cap
        )
        ladder_rows.append(
            {
                "Rung": f"Rung {idx}",
                "Alt Line": f"{alt_line}+",
                "Est. Hit %": f"{est_p:.1f}%",
                "Sample Odds": alt_odds,
                "Suggested Stake ($)": stake_val,
            }
        )
    st.dataframe(pd.DataFrame(ladder_rows), use_container_width=True, hide_index=True)

with tab_scanner:
    st.subheader("OddsPapi Scanner & Local Market Cache")
    cache_info = load_local_cache_summary()
    if cache_info:
        st.write("Local cache files detected in workspace:")
        st.dataframe(pd.DataFrame(cache_info), use_container_width=True, hide_index=True)
    else:
        st.info("No local markets_cache_*.json files found in this directory.")

    if st.button("Test Live OddsPapi NBA Fixtures"):
        if not api_key:
            st.error("Add ODDSPAPI_API_KEY in Streamlit Secrets or .env first.")
        else:
            ok, payload = fetch_oddspapi_fixtures(api_key, sport="basketball_nba")
            if ok:
                count = len(payload) if isinstance(payload, list) else "response received"
                st.success(f"Connected to OddsPapi ({count})")
                st.json(payload[:3] if isinstance(payload, list) else payload)
            else:
                st.error(f"OddsPapi request failed: {payload}")

with tab_log:
    st.subheader("ROI & Bet Log")
    log_df = load_bet_log()
    if log_df.empty:
        st.info(
            "No logged bets yet. Use 'Log Candidate to Bet Tracker' to record entries."
        )
    else:
        total_staked = (
            pd.to_numeric(log_df["stake_usd"], errors="coerce").fillna(0.0).sum()
        )
        st.metric("Total Logged Stake", f"${total_staked:.2f}")
        st.dataframe(log_df, use_container_width=True, hide_index=True)
Commit and push app.py in PowerShell: Run this inside C:\Users\ricky\OneDrive\Desktop\Smart Bets:   
Windows PowerShell.jpg

PowerShell
python -m py_compile app.py
git add app.py
git commit -m "Fix truncated string literal in app.py"
git push
Verification: py_compile exits with zero output/errors, and git push updates main on SaxmanR3/asymmetric-betting-hub.

Reload your app tab: Press Ctrl + Shift + R on r9nz3qtxzofafgr5jqca27.streamlit.app (or click ⋮ -> Rerun in the app menu).
Verification: The Script execution error modal disappears and the Asymmetric Betting Hub title, sidebar ($54.92 bankroll / $2.75 max ticket), and four tabs render cleanly.   
Streamlit - Google Chrome.png
+ 1