import json, os
from datetime import datetime, timezone
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
BOOK_MAP = {"fanduel": "FanDuel", "draftkings": "DraftKings", "kalshi": "Kalshi", "polymarket": "Polymarket"}

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

def dec_to_am(dec: float) -> int:
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

def pick_team(f, keys):
    for k in keys:
        v = f.get(k)
        if isinstance(v, dict): v = v.get("name") or v.get("title")
        if isinstance(v, str) and v.strip(): return v.strip()
    return None

def parse_fixtures(payload):
    items = payload if isinstance(payload, list) else payload.get("fixtures") or payload.get("data") or payload.get("items") or []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    upcoming, all_parsed = [], []
    for idx, f in enumerate(items):
        if not isinstance(f, dict): continue
        fid = str(f.get("fixtureId") or f.get("id") or f"game_{idx+1}")
        away = pick_team(f, ["awayTeamName", "participant2Name", "away", "opponentName"]) or "Away"
        home = pick_team(f, ["homeTeamName", "participant1Name", "home", "teamName"]) or "Home"
        start = str(f.get("startTime") or f.get("commenceTime") or f.get("date") or "")[:16]
        has_odds = bool(f.get("hasOdds", False))
        rec = {"fixtureId": fid, "Matchup": f"{away} @ {home}", "Start": start, "HasOdds": has_odds, "Label": f"{away} @ {home} ({start})"}
        all_parsed.append(rec)
        if start[:10] >= now_iso:
            upcoming.append(rec)
    chosen = upcoming if upcoming else all_parsed
    return chosen[:30], len(upcoming), len(all_parsed)

def parse_odds_payload(payload, matchup: str):
    rows = []
    books_obj = payload.get("bookmakerOdds") or payload.get("bookmakers") or {}
    if not isinstance(books_obj, dict): return rows
    for b_slug, b_data in books_obj.items():
        if not isinstance(b_data, dict): continue
        app_lbl = BOOK_MAP.get(b_slug.lower(), b_slug)
        markets = b_data.get("markets") or {}
        if not isinstance(markets, dict): continue
        for m_id, m_data in markets.items():
            if not isinstance(m_data, dict): continue
            m_name = m_data.get("marketName") or m_data.get("name") or f"Market {m_id}"
            outcomes = m_data.get("outcomes") or {}
            if not isinstance(outcomes, dict): continue
            for o_id, o_data in outcomes.items():
                if not isinstance(o_data, dict): continue
                players = o_data.get("players") or {"0": o_data}
                if not isinstance(players, dict): players = {"0": o_data}
                for p_id, p_data in players.items():
                    if not isinstance(p_data, dict): continue
                    if p_data.get("active") is False: continue
                    am = p_data.get("priceAmerican")
                    dec = p_data.get("price")
                    if am is None and dec is not None:
                        try: am = dec_to_am(float(dec))
                        except Exception: continue
                    if am is None: continue
                    try: am_int = int(round(float(am)))
                    except Exception: continue
                    o_name = p_data.get("outcomeName") or o_data.get("outcomeName") or o_data.get("name") or f"Outcome {o_id}"
                    p_name = p_data.get("playerName") or ""
                    line = p_data.get("handicap") or m_data.get("handicap") or ""
                    desc = " ".join(x for x in [p_name, m_name, o_name, str(line)] if str(x).strip()).strip()
                    rows.append({"Book": app_lbl, "Market": desc, "Odds": am_int, "Dec": dec, "Label": f"[{app_lbl}] {desc} ({am_int:+d})", "FullMkt": f"{matchup} · {desc}"})
    return rows

if "br_state" not in st.session_state: st.session_state.br_state = load_saved_br()
if "sizer_mkt" not in st.session_state: st.session_state.sizer_mkt = "NBA Cup Prop"
if "sizer_odds" not in st.session_state: st.session_state.sizer_odds = -220
if "sizer_app" not in st.session_state: st.session_state.sizer_app = "FanDuel"
if "scan_rows" not in st.session_state: st.session_state.scan_rows = []
if "scan_meta" not in st.session_state: st.session_state.scan_meta = ""
if "live_odds_rows" not in st.session_state: st.session_state.live_odds_rows = []
if "live_odds_raw" not in st.session_state: st.session_state.live_odds_raw = None

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
    if api_key: st.success("OddsPapi Key Ready")
    else: st.warning("OddsPapi Key Missing")

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
    if prob >= 80.0: st.success("PASSES 80% FLOOR")
    else: st.error("BELOW 80% FLOOR")
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
    if st.button("Pull Upcoming NBA Games"):
        if not api_key: st.error("Set ODDSPAPI_API_KEY in Secrets or .env")
        else:
            r = requests.get("https://api.oddspapi.io/v4/fixtures", params={"apiKey": api_key, "tournamentId": 132, "statusId": 0}, timeout=12)
            st.write(f"Status: {r.status_code}")
            if r.status_code == 200:
                parsed, up_cnt, tot_cnt = parse_fixtures(r.json())
                st.session_state.scan_rows = parsed
                st.session_state.scan_meta = f"Upcoming from today: {up_cnt} | Total returned: {tot_cnt}"
                st.session_state.live_odds_rows = []
    if st.session_state.scan_meta: st.caption(st.session_state.scan_meta)
    if st.session_state.scan_rows:
        df_scan = pd.DataFrame(st.session_state.scan_rows)[["Matchup", "Start", "HasOdds"]]
        st.dataframe(df_scan, use_container_width=True, hide_index=True)
        label_map = {x["Label"]: x for x in st.session_state.scan_rows}
        pick_lbl = st.selectbox("Select Fixture", list(label_map.keys()))
        chosen = label_map[pick_lbl]
        if st.button("Fetch Live Odds for Selected Game"):
            ro = requests.get("https://api.oddspapi.io/v4/odds", params={"apiKey": api_key, "fixtureId": chosen["fixtureId"], "bookmakers": "fanduel,draftkings,kalshi,polymarket", "oddsFormat": "american", "verbosity": 3}, timeout=15)
            st.write(f"Live Odds Status: {ro.status_code}")
            if ro.status_code == 200:
                payload_o = ro.json()
                st.session_state.live_odds_raw = payload_o
                st.session_state.live_odds_rows = parse_odds_payload(payload_o, chosen["Matchup"])
            else:
                st.error(ro.text[:300])
        if st.session_state.live_odds_rows:
            st.subheader(f"Live Book Lines ({len(st.session_state.live_odds_rows)} outcomes)")
            df_o = pd.DataFrame(st.session_state.live_odds_rows)[["Book", "Market", "Odds", "Dec"]]
            st.dataframe(df_o, use_container_width=True, hide_index=True)
            o_map = {x["Label"]: x for x in st.session_state.live_odds_rows}
            pick_o_lbl = st.selectbox("Select Live Line to Send to Sizer", list(o_map.keys()))
            if st.button("Load Live Line into Sizer"):
                sel_o = o_map[pick_o_lbl]
                st.session_state.sizer_mkt = sel_o["FullMkt"]
                st.session_state.sizer_app = sel_o["Book"] if sel_o["Book"] in app_bankrolls else "FanDuel"
                st.session_state.sizer_odds = int(sel_o["Odds"])
                st.success(f"Loaded {sel_o[\x27Label\x27]} into 80% Floor & Kelly tab.")
        elif st.session_state.live_odds_raw is not None:
            st.info("No active lines returned for those 4 books yet on this fixture.")
            with st.expander("Inspect Raw Odds JSON"):
                st.json(st.session_state.live_odds_raw)
        st.divider()
        st.caption("Or send selected matchup manually with custom odds:")
        pick_app = st.selectbox("Target Book / App", list(app_bankrolls.keys()), key="scan_app")
        pick_odds = st.number_input("Custom Market Odds (American)", -2000, 2000, -150, 5, key="scan_odds")
        if st.button("Load Manual Line into Sizer"):
            st.session_state.sizer_mkt = f"{chosen[\x27Matchup\x27]} ML/Prop"
            st.session_state.sizer_app = pick_app
            st.session_state.sizer_odds = int(pick_odds)
            st.success(f"Loaded {chosen[\x27Matchup\x27]} into 80% Floor & Kelly tab.")

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
