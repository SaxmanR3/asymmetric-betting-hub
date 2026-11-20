import json, os
from datetime import datetime
from pathlib import Path
import pandas as pd, requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Asymmetric Betting Hub", page_icon="🏀", layout="wide")

BASE = Path(__file__).resolve().parent
LOG_PATH = BASE / "bet_log.csv"
BR_PATH = BASE / "bankrolls.json"
CACHES = [BASE / "markets_cache_11.json", BASE / "markets_cache_14.json"]
DEFAULT_BR = {"FanDuel": 25.00, "DraftKings": 19.92, "Kalshi": 10.00, "Polymarket": 0.00}

def get_key():
    try:
        if "ODDSPAPI_API_KEY" in st.secrets:
            v = str(st.secrets["ODDSPAPI_API_KEY"]).strip()
            if v: return v
    except Exception: pass
    v = os.getenv("ODDSPAPI_API_KEY", "").strip()
    return v or None

def load_saved_br():
    if BR_PATH.exists():
        try:
            data = json.loads(BR_PATH.read_text(encoding="utf-8"))
            return {k: round(float(data.get(k, DEFAULT_BR[k])), 2) for k in DEFAULT_BR}
        except Exception: pass
    return dict(DEFAULT_BR)

def save_br(br_dict):
    clean = {k: round(float(v), 2) for k, v in br_dict.items()}
    BR_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")

def imp_prob(odds: float) -> float:
    if odds == 0: return 0.0
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)

def decimal_to_american(dec: float) -> int:
    if dec <= 1.0: return -100
    if dec >= 2.0: return int(round((dec - 1.0) * 100))
    return int(round(-100.0 / (dec - 1.0)))

def payout_delta(stake: float, odds: float, result: str) -> float:
    dec = (1.0 + odds / 100.0) if odds > 0 else (1.0 + 100.0 / abs(odds)) if odds < 0 else 1.0
    if result == "WIN": return round(stake * (dec - 1.0), 2)
    if result == "LOSS": return round(-stake, 2)
    return 0.0

def kelly_stake(p: float, odds: float, bankroll: float, frac: float, cap: float) -> float:
    dec = (1.0 + odds / 100.0) if odds > 0 else (1.0 + 100.0 / abs(odds)) if odds < 0 else 1.0
    b = dec - 1.0
    if b <= 0 or not (0 < p < 1): return 0.0
    edge = (b * p - (1.0 - p)) / b
    if edge <= 0: return 0.0
    return round(min(bankroll * edge * frac, cap, bankroll), 2)

def parse_fixtures(payload):
    items = payload if isinstance(payload, list) else payload.get("fixtures") or payload.get("data") or payload.get("items") or []
    rows = []
    for idx, f in enumerate(items[:25]):
        if not isinstance(f, dict): continue
        fid = f.get("fixtureId") or f.get("id") or f"game_{idx+1}"
        home = f.get("homeTeamName") or f.get("home") or (f.get("homeTeam", {}) or {}).get("name") or "Home"
        away = f.get("awayTeamName") or f.get("away") or (f.get("awayTeam", {}) or {}).get("name") or "Away"
        start = f.get("startTime") or f.get("commenceTime") or f.get("date") or ""
        books = f.get("bookmakers") or f.get("odds") or {}
        b_count = len(books) if isinstance(books, (list, dict)) else 0
        rows.append({"id": str(fid), "Matchup": f"{away} @ {home}", "Start": str(start)[:16], "Books": b_count})
    return rows

if "br_state" not in st.session_state:
    st.session_state.br_state = load_saved_br()
if "sizer_mkt" not in st.session_state:
    st.session_state.sizer_mkt = "NBA Cup Prop"
if "sizer_odds" not in st.session_state:
    st.session_state.sizer_odds = -220
if "sizer_app" not in st.session_state:
    st.session_state.sizer_app = "FanDuel"
if "scan_rows" not in st.session_state:
    st.session_state.scan_rows = []

st.title("Asymmetric Betting Hub")
st.caption("NBA Cup Slate · Nov 20, 2026 · 80% Floor · Layer 3 Ladder · Fractional Kelly")

with st.sidebar:
    st.subheader("Bankroll by App")
    fd_br = st.number_input("FanDuel ($)", 0.0, 100000.0, float(st.session_state.br_state["FanDuel"]), 0.5, "%.2f")
    dk_br = st.number_input("DraftKings ($)", 0.0, 100000.0, float(st.session_state.br_state["DraftKings"]), 0.5, "%.2f")
    kl_br = st.number_input("Kalshi ($)", 0.0, 100000.0, float(st.session_state.br_state["Kalshi"]), 0.5, "%.2f")
    pm_br = st.number_input("Polymarket ($)", 0.0, 100000.0, float(st.session_state.br_state["Polymarket"]), 0.5, "%.2f")
    app_bankrolls = {"FanDuel": round(fd_br, 2), "DraftKings": round(dk_br, 2), "Kalshi": round(kl_br, 2), "Polymarket": round(pm_br, 2)}
    if app_bankrolls != st.session_state.br_state:
        st.session_state.br_state = dict(app_bankrolls)
        save_br(app_bankrolls)
    total_bankroll = round(sum(app_bankrolls.values()), 2)
    cap_pct = st.slider("Max Stake Cap (%)", 1.0, 10.0, 5.0, 0.5)
    kelly_f = st.selectbox("Kelly Fraction", [0.1, 0.25, 0.5, 1.0], 1, format_func=lambda x: f"{int(x*100)}%")
    st.divider()
    st.metric("Combined Total Bankroll", f"${total_bankroll:.2f}")
    api_key = get_key()
    if api_key:
        st.success("OddsPapi Key Ready")
    else:
        st.warning("OddsPapi Key Missing")

t1, t2, t3, t4 = st.tabs(["80% Floor & Kelly", "Layer 3 Ladder", "Scanner & Cache", "Bet Log"])

with t1:
    c0, c1, c2, c3 = st.columns(4)
    app_idx = list(app_bankrolls.keys()).index(st.session_state.sizer_app) if st.session_state.sizer_app in app_bankrolls else 0
    app_name = c0.selectbox("Betting App", list(app_bankrolls.keys()), index=app_idx)
    mkt = c1.text_input("Market", value=st.session_state.sizer_mkt)
    prob = c2.slider("Floor Hit Prob (%)", 50.0, 99.0, 82.0, 0.5)
    odds = c3.number_input("American Odds", -2000, 2000, int(st.session_state.sizer_odds), 5)
    active_br = float(app_bankrolls[app_name])
    active_cap = round(active_br * cap_pct / 100.0, 2)
    stake = kelly_stake(prob / 100.0, float(odds), active_br, float(kelly_f), active_cap)
    st.caption(f"{app_name} Bankroll: ${active_br:.2f} · {app_name} Max Cap ({cap_pct:.1f}%): ${active_cap:.2f}")
    st.metric("Implied Prob", f"{imp_prob(float(odds))*100:.1f}%")
    st.metric("Suggested Stake", f"${stake:.2f}")
    if prob >= 80.0:
        st.success("PASSES 80% FLOOR")
    else:
        st.error("BELOW 80% FLOOR")
    if st.button("Log Bet"):
        row = pd.DataFrame([{"id": int(datetime.now().timestamp()*1000), "ts": datetime.now().isoformat(timespec="seconds"), "app": app_name, "mkt": mkt, "odds": int(odds), "prob": float(prob), "stake": float(stake), "status": "OPEN", "pnl": 0.0}])
        out = pd.concat([pd.read_csv(LOG_PATH), row], ignore_index=True) if LOG_PATH.exists() else row
        out.to_csv(LOG_PATH, index=False)
        st.success(f"Logged {mkt} (${stake:.2f}) on {app_name}")

with t2:
    base_line = st.number_input("Base Line", 0.5, 100.0, 20.0, 0.5)
    rows = [{"Rung": i, "Alt": f"{round(base_line*m,1)}+", "Est %": f"{88-(i-1)*11:.1f}%", "Stake": kelly_stake((88-(i-1)*11)/100, -240 if i<=2 else -140, total_bankroll, kelly_f, round(total_bankroll*cap_pct/100.0, 2))} for i, m in enumerate([0.75, 0.85, 0.95, 1.0], 1)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t3:
    found = [{"File": p.name, "KB": round(p.stat().st_size/1024, 1)} for p in CACHES if p.exists()]
    if found: st.dataframe(pd.DataFrame(found), use_container_width=True, hide_index=True)
    if st.button("Test OddsPapi"):
        if not api_key: st.error("Set ODDSPAPI_API_KEY in Secrets or .env")
        else:
            r = requests.get("https://api.oddspapi.io/v4/fixtures", params={"apiKey": api_key, "tournamentId": 132}, timeout=12)
            st.write(f"Status: {r.status_code}")
            if r.status_code == 200:
                st.session_state.scan_rows = parse_fixtures(r.json())
    if st.session_state.scan_rows:
        df_scan = pd.DataFrame(st.session_state.scan_rows)[["Matchup", "Start", "Books"]]
        st.dataframe(df_scan, use_container_width=True, hide_index=True)
        pick_lbl = st.selectbox("Select Fixture to Send to Sizer", [x["Matchup"] for x in st.session_state.scan_rows])
        pick_app = st.selectbox("Target Book / App", list(app_bankrolls.keys()), key="scan_app")
        pick_odds = st.number_input("Market Odds (American)", -2000, 2000, -150, 5, key="scan_odds")
        if st.button("Load into Sizer"):
            st.session_state.sizer_mkt = f"{pick_lbl} ML/Prop"
            st.session_state.sizer_app = pick_app
            st.session_state.sizer_odds = int(pick_odds)
            st.success(f"Loaded {pick_lbl} into 80% Floor & Kelly tab.")

with t4:
    if LOG_PATH.exists():
        df_log = pd.read_csv(LOG_PATH)
        if "status" not in df_log.columns: df_log["status"] = "OPEN"
        if "pnl" not in df_log.columns: df_log["pnl"] = 0.0
        if "id" not in df_log.columns: df_log["id"] = range(1, len(df_log)+1)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        open_bets = df_log[df_log["status"] == "OPEN"]
        if not open_bets.empty:
            st.subheader("Settle Open Bet & Update App Bankroll")
            opt_map = {f"#{int(r.id)} | {r.app} | {r.mkt} (${r.stake:.2f} @ {int(r.odds)})": int(r.id) for _, r in open_bets.iterrows()}
            sel_label = st.selectbox("Open Bet", list(opt_map.keys()))
            outcome = st.radio("Result", ["WIN", "LOSS", "PUSH"], horizontal=True)
            if st.button("Settle Bet"):
                bet_id = opt_map[sel_label]
                mask = df_log["id"] == bet_id
                row = df_log[mask].iloc[0]
                delta = payout_delta(float(row["stake"]), float(row["odds"]), outcome)
                df_log.loc[mask, "status"] = outcome
                df_log.loc[mask, "pnl"] = delta
                df_log.to_csv(LOG_PATH, index=False)
                b_app = str(row["app"])
                if b_app in st.session_state.br_state:
                    st.session_state.br_state[b_app] = round(max(0.0, float(st.session_state.br_state[b_app]) + delta), 2)
                    save_br(st.session_state.br_state)
                st.success(f"Settled as {outcome} (P&L: ${delta:+.2f}). Updated {b_app} bankroll and rerunning...")
                st.rerun()
    else:
        st.info("No bets logged yet.")
