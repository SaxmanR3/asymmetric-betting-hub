import json, os
from datetime import datetime
from pathlib import Path
import pandas as pd, requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Asymmetric Betting Hub", page_icon="🏀", layout="wide")

BASE = Path(__file__).resolve().parent
LOG_PATH = BASE / "bet_log.csv"
CACHES = [BASE / "markets_cache_11.json", BASE / "markets_cache_14.json"]

def get_key():
    try:
        if "ODDSPAPI_API_KEY" in st.secrets:
            v = str(st.secrets["ODDSPAPI_API_KEY"]).strip()
            if v:
                return v
    except Exception:
        pass
    v = os.getenv("ODDSPAPI_API_KEY", "").strip()
    return v or None

def imp_prob(odds: float) -> float:
    if odds == 0:
        return 0.0
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)

def kelly_stake(p: float, odds: float, bankroll: float, frac: float, cap: float) -> float:
    dec = (1.0 + odds / 100.0) if odds > 0 else (1.0 + 100.0 / abs(odds)) if odds < 0 else 1.0
    b = dec - 1.0
    if b <= 0 or not (0 < p < 1):
        return 0.0
    edge = (b * p - (1.0 - p)) / b
    if edge <= 0:
        return 0.0
    return round(min(bankroll * edge * frac, cap, bankroll), 2)

st.title("Asymmetric Betting Hub")
st.caption("NBA Cup Slate · Nov 20, 2026 · 80% Floor · Layer 3 Ladder · Fractional Kelly")

with st.sidebar:
    bankroll = st.number_input("Bankroll ($)", 0.0, 100000.0, 54.92, 0.5, "%.2f")
    cap_pct = st.slider("Max Stake Cap (%)", 1.0, 10.0, 5.0, 0.5)
    cap_usd = round(bankroll * cap_pct / 100.0, 2)
    kelly_f = st.selectbox("Kelly Fraction", [0.1, 0.25, 0.5, 1.0], 1, format_func=lambda x: f"{int(x*100)}%")
    st.metric("Max Bet Cap", f"${cap_usd:.2f}")
    api_key = get_key()
    if api_key:
        st.success("OddsPapi Key Ready")
    else:
        st.warning("OddsPapi Key Missing")

t1, t2, t3, t4 = st.tabs(["80% Floor & Kelly", "Layer 3 Ladder", "Scanner & Cache", "Bet Log"])

with t1:
    c1, c2, c3 = st.columns(3)
    mkt = c1.text_input("Market", "NBA Cup Prop")
    prob = c2.slider("Floor Hit Prob (%)", 50.0, 99.0, 82.0, 0.5)
    odds = c3.number_input("American Odds", -2000, 2000, -220, 5)
    stake = kelly_stake(prob / 100.0, float(odds), float(bankroll), float(kelly_f), cap_usd)
    st.metric("Implied Prob", f"{imp_prob(float(odds))*100:.1f}%")
    st.metric("Suggested Stake", f"${stake:.2f}")
    if prob >= 80.0:
        st.success("PASSES 80% FLOOR")
    else:
        st.error("BELOW 80% FLOOR")
    if st.button("Log Bet"):
        row = pd.DataFrame([{"ts": datetime.now().isoformat(timespec="seconds"), "mkt": mkt, "odds": odds, "prob": prob, "stake": stake}])
        out = pd.concat([pd.read_csv(LOG_PATH), row], ignore_index=True) if LOG_PATH.exists() else row
        out.to_csv(LOG_PATH, index=False)
        st.success("Saved to bet_log.csv")

with t2:
    base_line = st.number_input("Base Line", 0.5, 100.0, 20.0, 0.5)
    rows = [
        {
            "Rung": i,
            "Alt": f"{round(base_line*m,1)}+",
            "Est %": f"{88-(i-1)*11:.1f}%",
            "Stake": kelly_stake((88-(i-1)*11)/100, -240 if i<=2 else -140, bankroll, kelly_f, cap_usd),
        }
        for i, m in enumerate([0.75, 0.85, 0.95, 1.0], 1)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t3:
    found = [{"File": p.name, "KB": round(p.stat().st_size/1024, 1)} for p in CACHES if p.exists()]
    if found:
        st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
    if st.button("Test OddsPapi"):
        if not api_key:
            st.error("Set ODDSPAPI_API_KEY in Secrets or .env")
        else:
            r = requests.get("https://api.oddspapi.io/v4/fixtures", params={"apiKey": api_key, "sport": "basketball_nba"}, timeout=12)
            st.write(f"Status: {r.status_code}")

with t4:
    if LOG_PATH.exists():
        st.dataframe(pd.read_csv(LOG_PATH), use_container_width=True, hide_index=True)
    else:
        st.info("No bets logged yet.")
