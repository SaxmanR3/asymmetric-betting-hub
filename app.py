import os
import json
import math
from datetime import date, timedelta
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from dotenv import load_dotenv

load_dotenv()
BASE_URL = "https://api.oddspapi.io/v4"
LOG_FILE = "bet_log.csv"

def get_api_key() -> str:
    try:
        if "ODDSPAPI_API_KEY" in st.secrets:
            val = str(st.secrets["ODDSPAPI_API_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass
    return os.getenv("ODDSPAPI_API_KEY", "").strip()

API_KEY = get_api_key()

st.set_page_config(page_title="Asymmetric Betting Hub", layout="wide")
st.title("Asymmetric Betting Hub")

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

def devig_two_way(odds_a: float, odds_b: float):
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

def lognormal_params(mean: float, sd: float):
    sigma = math.sqrt(math.log(1.0 + (sd / mean) ** 2))
    mu = math.log(mean) - 0.5 * (sigma ** 2)
    return mu, sigma

def calc_ev(true_prob: float, book_odds: float) -> float:
    return (true_prob * amer_to_dec(book_odds) - 1.0) * 100.0

def matches_league(fx: dict, league_filter: str) -> bool:
    if not league_filter:
        return True
    tourn = (fx.get("tournamentName") or fx.get("tournamentSlug") or "").lower()
    if league_filter == "nba":
        return "nba" in tourn and "wnba" not in tourn
    if league_filter == "wnba":
        return "wnba" in tourn
    if league_filter == "ncaa":
        return any(k in tourn for k in ("ncaa", "college", "fbs", "fcs", "division i", "cbb", "ncaab", "ncaaf"))
    if league_filter == "nfl":
        return "nfl" in tourn and "ncaa" not in tourn
    return league_filter in tourn

def classify_level(fx: dict, default_level: str = "Pro") -> str:
    tourn = (fx.get("tournamentName") or fx.get("tournamentSlug") or "").lower()
    if any(k in tourn for k in ("ncaa", "college", "fbs", "fcs", "division i", "ncaab", "ncaaf")):
        return "College"
    if any(k in tourn for k in ("nfl", "nba", "wnba", "nhl", "mlb", "mls", "epl", "ucl", "atp", "wta", "ufc")):
        return "Pro"
    return default_level

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

def fmt_book(slug: str) -> str:
    return BOOK_LABELS.get(slug.lower(), slug.title())

def load_market_catalog(sport_id: int) -> dict:
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
    r = requests.get(
        f"{BASE_URL}/markets",
        params={"apiKey": API_KEY, "sportId": sport_id, "language": "en"},
        timeout=20,
    )
    if r.status_code != 200:
        return {}
    raw = r.json()
    catalog = {}
    for m in raw:
        mid = str(m.get("marketId"))
        outcomes = {str(o.get("outcomeId")): o.get("outcomeName", str(o.get("outcomeId"))) for o in m.get("outcomes", [])}
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

def parse_american_price(node: dict):
    if not isinstance(node, dict) or not node.get("active", False):
        return None
    dec = node.get("price")
    am_str = node.get("priceAmerican")
    if am_str is not None:
        try:
            am_val = int(round(float(am_str)))
            if am_val <= -100 or am_val >= 100:
                return am_val
        except (ValueError, TypeError):
            pass
    if dec is not None:
        try:
            dec_f = float(dec)
            if dec_f > 1.01:
                return dec_to_amer(dec_f)
        except (ValueError, TypeError):
            pass
    return None

def scan_fixture_ev(b_odds: dict, catalog: dict, target_books: list, max_width: int, min_ev: float):
    candidate_pairs = set()
    for ref_bk in ("pinnacle", "fanduel"):
        markets = b_odds.get(ref_bk, {}).get("markets", {})
        for mid, mdata in markets.items():
            outcomes = mdata.get("outcomes", {})
            if len(outcomes) != 2:
                continue
            for odata in outcomes.values():
                for pid, pnode in odata.get("players", {}).items():
                    if isinstance(pnode, dict) and pnode.get("active", False):
                        candidate_pairs.add((mid, pid))

    ev_rows = []
    pp_rows = []

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
            mdata = b_odds.get(candidate_ref, {}).get("markets", {}).get(mid)
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
                    player_name = (n0.get("playerName") or n1.get("playerName") or f"Player {pid}")
                break

        if not ref_book or not ref_sides:
            continue

        oid0, odds0, oid1, odds1 = ref_sides
        fair_p0, fair_p1, vig_pct, width_c = devig_two_way(odds0, odds1)

        if width_c > max_width:
            continue

        label0 = out_names.get(oid0, f"Outcome {oid0}")
        label1 = out_names.get(oid1, f"Outcome {oid1}")
        sides_info = [
            (oid0, label0, fair_p0, odds0, odds1),
            (oid1, label1, fair_p1, odds1, odds0),
        ]

        if is_prop:
            for oid, s_label, f_prob, r_this, r_opp in sides_info:
                edge_6 = (f_prob - 0.5421) * 100.0
                edge_5 = (f_prob - 0.5425) * 100.0
                if edge_6 > 0:
                    pp_rows.append({
                        "Player": player_name,
                        "Market": m_name,
                        "Side": s_label,
                        "Ref Book": fmt_book(ref_book),
                        "Ref Line": f"{r_this:+d} / {r_opp:+d}",
                        "True Prob %": round(f_prob * 100.0, 2),
                        "Width (c)": int(round(width_c)),
                        "6-Flex Edge %": round(edge_6, 2),
                        "5-Flex Edge %": round(edge_5, 2),
                    })

        for tb in target_books:
            if tb == ref_book or tb not in b_odds:
                continue
            tb_m = b_odds[tb].get("markets", {}).get(mid)
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
                    ev_rows.append({
                        "Type": "Prop" if is_prop else "Main",
                        "Market": m_name,
                        "Selection": sel_display,
                        "Book": fmt_book(tb),
                        "Book Odds": f"{soft_am:+d}",
                        "Sharp Ref": f"{fmt_book(ref_book)} ({r_this:+d}/{r_opp:+d})",
                        "Fair Odds": f"{prob_to_amer(f_prob):+d}",
                        "True %": round(f_prob * 100.0, 1),
                        "Width (c)": int(round(width_c)),
                        "EV %": round(ev_pct, 2),
                    })

    ev_rows.sort(key=lambda x: x["EV %"], reverse=True)
    pp_rows.sort(key=lambda x: x["True Prob %"], reverse=True)
    return ev_rows, pp_rows

SD_MULTIPLIERS = {
    "Pass Attempts": 0.25,
    "Completions": 0.28,
    "Passing Yards": 0.31,
    "Carries": 0.39,
    "Receptions": 0.53,
    "Rush + Rec Yards": 0.50,
    "Rushing Yards (Overall)": 0.56,
    "Rushing Yards (<60 YPG)": 0.63,
    "Rushing Yards (60-80 YPG)": 0.52,
    "Rushing Yards (80+ YPG)": 0.42,
    "Receiving Yards (WR)": 0.64,
    "Receiving Yards (TE)": 0.69,
}

DEFAULT_SPORTS = {
    "Pro Football (NFL)": {"sportId": 14, "filter": "nfl", "level": "Pro"},
    "College Football (NCAAF)": {"sportId": 14, "filter": "ncaa", "level": "College"},
    "Pro Basketball (NBA)": {"sportId": 11, "filter": "nba", "level": "Pro"},
    "College Basketball (NCAAB)": {"sportId": 11, "filter": "ncaa", "level": "College"},
    "Pro Women's Basketball (WNBA)": {"sportId": 11, "filter": "wnba", "level": "Pro"},
    "Pro Hockey (NHL)": {"sportId": 15, "filter": "nhl", "level": "Pro"},
    "Pro Baseball (MLB)": {"sportId": 13, "filter": "mlb", "level": "Pro"},
    "Soccer (MLS / EPL / UCL)": {"sportId": 10, "filter": "", "level": "Pro"},
    "Tennis (ATP / WTA)": {"sportId": 12, "filter": "", "level": "Pro"},
    "Boxing": {"sportId": 21, "filter": "", "level": "Pro"},
}

TARGET_SOFT_BOOKS = ["draftkings", "fanduel", "kalshi", "betmgm", "fanatics", "caesars", "bet365"]
PRIORITY_BOOKS = ["pinnacle", "fanduel", "draftkings", "kalshi", "betmgm", "fanatics", "caesars", "bet365"]

with st.sidebar:
    st.header("Bankroll & Unit Rules")
    fd_bal = st.number_input("FanDuel ($)", min_value=0.0, value=25.00, step=0.5, format="%.2f")
    dk_bal = st.number_input("DraftKings ($)", min_value=0.0, value=19.92, step=0.5, format="%.2f")
    kalshi_bal = st.number_input("Kalshi ($)", min_value=0.0, value=10.00, step=0.5, format="%.2f")
    pp_bal = st.number_input("PrizePicks ($)", min_value=0.0, value=0.00, step=0.5, format="%.2f")
    mgm_bal = st.number_input("BetMGM ($)", min_value=0.0, value=0.00, step=0.5, format="%.2f")
    fanatics_bal = st.number_input("Fanatics ($)", min_value=0.0, value=0.00, step=0.5, format="%.2f")
    total_roll = round(fd_bal + dk_bal + kalshi_bal + pp_bal + mgm_bal + fanatics_bal, 2)
    st.metric("Total Active Roll", f"${total_roll:,.2f}")
    unit_size = st.number_input("Standard Unit ($)", min_value=0.5, value=2.0, step=0.5)
    max_single_bet = round(total_roll * 0.05, 2)
    st.caption(f"5% Max Single-Ticket Cap: **${max_single_bet:.2f}**")
    st.divider()
    if API_KEY:
        st.success("OddsPapi Key Loaded")
    else:
        st.warning("No ODDSPAPI_API_KEY in Secrets or .env")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Combos & Singles",
    "2. PrizePicks & +EV Scanner",
    "3. Layer 3 Skew & Alt Ladder",
    "4. Promo & Bankroll Guardrails",
    "5. ROI & CLV Tracker",
])

with tab1:
    st.subheader("Combos & Singles Guardrail (PA / PRA / Rush+Rec)")
    st.caption("80% single breakeven: **-400** | Two 80% legs: **64.0% (-178)** | Three 80% legs: **51.2% (-105)**")
    c1, c2 = st.columns(2)
    with c1:
        num_legs = st.radio("Number of Floor Legs", [1, 2, 3], horizontal=True, index=2)
        est_hit = st.slider("Estimated Hit Rate per Leg (%)", 65, 90, 80) / 100.0
    with c2:
        posted_odds = st.number_input("Posted Book Odds (American)", value=-115, step=5)
        injury_clear = st.checkbox("All players confirmed ACTIVE (No Q/D/O tags or blowout traps)", value=False)
    comb_prob = est_hit ** num_legs
    fair_odds = prob_to_amer(comb_prob)
    ev_pct = calc_ev(comb_prob, posted_odds)
    m1, m2, m3 = st.columns(3)
    m1.metric("Combined Hit Rate", f"{comb_prob*100:.1f}%")
    m2.metric("Fair Breakeven Odds", f"{fair_odds:+d}")
    m3.metric("Estimated EV", f"{ev_pct:+.1f}%")
    if not injury_clear:
        st.error("GATE LOCKED: Verify official injury reports and spread/blowout risk before firing.")
    elif ev_pct < 0:
        st.warning(f"JUICE LEAK: Posted {posted_odds:+d} is worse than your fair price ({fair_odds:+d}). Shop books or pass.")
    else:
        st.success(f"GREEN LIGHT (+{ev_pct:.1f}% EV). Recommended Singles-First Split: **${unit_size:.2f}** per straight + **${unit_size*0.25:.2f}** combo kicker.")

with tab2:
    st.subheader("Live OddsPapi Feed + Auto Split-Devig & PrizePicks Scanner")
    st.caption("College & Pro Separated | Main Lines -> Devig to **Pinnacle** | Player Props -> Devig to **FanDuel** | PP 5-Flex BE: **54.25%** | 6-Flex BE: **54.21**")
    if "sport_map" not in st.session_state:
        st.session_state["sport_map"] = dict(DEFAULT_SPORTS)
    col_a, col_b, col_c, col_d = st.columns(4)
    selected_sport = col_a.selectbox("Sport / League", list(st.session_state["sport_map"].keys()))
    level_filter = col_b.selectbox("Level Split", ["All", "College Only", "Pro Only"])
    max_width = col_c.number_input("Max Market Width (cents)", min_value=5, max_value=60, value=25)
    min_ev = col_d.number_input("Min EV % to Flag", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    selected_books = st.multiselect(
        "Target Books to Shop",
        TARGET_SOFT_BOOKS,
        default=["draftkings", "fanduel", "kalshi", "betmgm", "fanatics"],
        format_func=fmt_book,
    )
    b1, b2 = st.columns([2, 1])
    if b1.button("Pull Upcoming Fixtures (1 API Credit)"):
        if not API_KEY:
            st.error("Add ODDSPAPI_API_KEY to Streamlit Secrets or .env first.")
        else:
            today = date.today()
            cfg = st.session_state["sport_map"][selected_sport]
            params = {
                "apiKey": API_KEY,
                "sportId": cfg["sportId"],
                "from": today.isoformat(),
                "to": (today + timedelta(days=7)).isoformat(),
                "hasOdds": "true",
            }
            try:
                r = requests.get(f"{BASE_URL}/fixtures", params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    raw_fixtures = data if isinstance(data, list) else data.get("fixtures", [])
                    league_filter = cfg.get("filter", "")
                    default_lvl = cfg.get("level", "Pro")
                    fixtures = []
                    for fx in raw_fixtures:
                        if not matches_league(fx, league_filter):
                            continue
                        lvl = classify_level(fx, default_lvl)
                        if level_filter == "College Only" and lvl != "College":
                            continue
                        if level_filter == "Pro Only" and lvl != "Pro":
                            continue
                        fx_copy = dict(fx)
                        fx_copy["_level"] = lvl
                        fixtures.append(fx_copy)
                    st.session_state["fixtures"] = fixtures
                    st.session_state["fixtures_sport_id"] = cfg["sportId"]
                    if fixtures:
                        st.success(f"Loaded {len(fixtures)} upcoming {selected_sport} games ({level_filter}).")
                    else:
                        st.warning(f"No upcoming {selected_sport} games found in the next 7 days.")
                else:
                    st.error(f"API Error {r.status_code}: {r.text[:250]}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    if b2.button("Load All Sports (UFC/Golf/etc.)"):
        if not API_KEY:
            st.error("Add ODDSPAPI_API_KEY to Streamlit Secrets or .env first.")
        else:
            try:
                r = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY}, timeout=15)
                if r.status_code == 200:
                    all_sports = r.json()
                    merged = dict(DEFAULT_SPORTS)
                    existing_ids = {v["sportId"] for v in merged.values()}
                    for s in all_sports:
                        name = s.get("sportName")
                        sid = s.get("sportId")
                        if name and sid and sid not in existing_ids:
                            merged[name] = {"sportId": sid, "filter": "", "level": "Pro"}
                    st.session_state["sport_map"] = merged
                    st.rerun()
                else:
                    st.error(f"API Error {r.status_code}: {r.text[:200]}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    if "fixtures" in st.session_state and st.session_state["fixtures"]:
        search_txt = st.text_input("Filter by Team (optional, 0 credits)", value="").strip().lower()
        fx_options = {}
        for fx in st.session_state["fixtures"]:
            fid = fx.get("fixtureId")
            if not fid:
                continue
            p1 = fx.get("participant1Name", "Away")
            p2 = fx.get("participant2Name", "Home")
            lvl = fx.get("_level", "Pro")
            tourn = fx.get("tournamentName") or fx.get("tournamentSlug") or ""
            stime = fx.get("startTime", "")[:16].replace("T", " ")
            label = f"[{lvl} | {tourn}] {p1} vs {p2} ({stime})" if tourn else f"[{lvl}] {p1} vs {p2} ({stime})"
            if not search_txt or search_txt in label.lower():
                fx_options[label] = fid
        if fx_options:
            chosen_label = st.selectbox(f"Select Game to Scan ({len(fx_options)} found)", list(fx_options.keys())[:100])
            if st.button("Scan & Find +EV / PrizePicks Plays (1 API Credit)"):
                fid = fx_options[chosen_label]
                sport_id = st.session_state.get("fixtures_sport_id", st.session_state["sport_map"][selected_sport]["sportId"])
                with st.spinner("Pulling live board & running split-devig math..."):
                    try:
                        catalog = load_market_catalog(sport_id)
                        r = requests.get(
                            f"{BASE_URL}/odds",
                            params={"apiKey": API_KEY, "fixtureId": fid},
                            timeout=25,
                        )
                        if r.status_code == 200:
                            data = r.json()
                            b_odds = data.get("bookmakerOdds", {})
                            st.session_state["last_scan_game"] = chosen_label
                            st.session_state["last_b_odds"] = b_odds
                            st.session_state["last_catalog"] = catalog
                        else:
                            st.error(f"API Error {r.status_code}: {r.text[:250]}")
                    except Exception as e:
                        st.error(f"Error fetching odds: {e}")
        if "last_b_odds" in st.session_state and st.session_state["last_b_odds"]:
            b_odds = st.session_state["last_b_odds"]
            catalog = st.session_state.get("last_catalog", {})
            ev_rows, pp_rows = scan_fixture_ev(b_odds, catalog, selected_books, max_width, min_ev)
            st.markdown(f"### Results: {st.session_state.get('last_scan_game', '')}")
            key_books = [b for b in PRIORITY_BOOKS if b in b_odds]
            st.caption(f"Live sharp/target books in this scan: **{ ', '.join(fmt_book(b) for b in key_books) }** — Adjust Max Width, Min EV, or Target Books above to re-filter for 0 credits.")
            st.markdown(f"#### +EV Soft Book Plays ({len(ev_rows)} found)")
            if ev_rows:
                st.dataframe(pd.DataFrame(ev_rows), width="stretch")
            else:
                st.info(f"No plays met >= {min_ev:.1f}% EV under {max_width}c width on {', '.join(fmt_book(b) for b in selected_books)}.")
            st.markdown(f"#### PrizePicks 5/6-Flex Qualifying Props ({len(pp_rows)} found)")
            if pp_rows:
                st.dataframe(pd.DataFrame(pp_rows), width="stretch")
            else:
                st.info(f"No two-way player props cleared 54.21% true probability under {max_width}c width yet.")
            with st.expander("View Active Books & Market Counts"):
                ordered_books = key_books + [b for b in b_odds.keys() if b not in key_books]
                summary_rows = [{"Bookmaker": fmt_book(bk), "Active Markets": len(b_odds[bk].get("markets", {}))} for bk in ordered_books[:20]]
                st.dataframe(pd.DataFrame(summary_rows), width="stretch")
    st.divider()
    st.markdown("#### Quick Manual Devig & PrizePicks Edge Checker (0 API Credits)")
    qc1, qc2, qc3, qc4 = st.columns(4)
    sharp_over = qc1.number_input("Sharp/FD Over Odds", value=-135, step=1)
    sharp_under = qc2.number_input("Sharp/FD Under Odds", value=105, step=1)
    target_book_odds = qc3.number_input("Soft Book Over Odds", value=-115, step=1)
    pp_mode = qc4.selectbox("Compare vs. PrizePicks?", ["6-Pick Flex (54.21%)", "5-Pick Flex (54.25%)"])
    fair_o, fair_u, vig_pct, width_c = devig_two_way(sharp_over, sharp_under)
    soft_ev = calc_ev(fair_o, target_book_odds)
    pp_be = 0.5421 if "6-Pick" in pp_mode else 0.5425
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("True Fair Over %", f"{fair_o*100:.2f}%", f"Fair {prob_to_amer(fair_o):+d}")
    r2.metric("Market Width", f"{width_c:.0f} cents", f"Vig: {vig_pct:.2f}%")
    r3.metric("Soft Book EV", f"{soft_ev:+.2f}%")
    r4.metric("PrizePicks Edge", f"{(fair_o - pp_be)*100:+.2f}%", "PLAY" if fair_o > pp_be else "PASS")
    if width_c > max_width:
        st.warning(f"WIDTH WARNING: {width_c:.0f}c exceeds your {max_width}c confidence cap.")

with tab3:
    st.subheader("Layer 3: Log-Normal Right Skew & Alt-Ladder Pricer")
    lc1, lc2 = st.columns(2)
    with lc1:
        prop_type = st.selectbox("Prop Type (SD Reference)", list(SD_MULTIPLIERS.keys()), index=6)
        proj_mean = st.number_input("Your Mean Projection", min_value=1.0, value=76.64, step=1.0)
        main_line = st.number_input("Sportsbook Main Line", min_value=0.5, value=70.5, step=0.5)
        main_odds = st.number_input("Main Line Over Odds", value=-114, step=1)
    with lc2:
        raw_logs = st.text_input("Optional: Player Game Logs (comma-separated, overrides multiplier)", value="")
        if raw_logs.strip():
            try:
                vals = [float(x.strip()) for x in raw_logs.split(",") if x.strip()]
                sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else proj_mean * SD_MULTIPLIERS[prop_type]
                st.info(f"Measured SD from {len(vals)} games: **{sd_val:.2f}** (Mult: {sd_val/np.mean(vals):.2f})")
            except ValueError:
                sd_val = proj_mean * SD_MULTIPLIERS[prop_type]
                st.error("Invalid game logs format. Using league multiplier.")
        else:
            sd_val = proj_mean * SD_MULTIPLIERS[prop_type]
            st.info(f"Using League Multiplier **{SD_MULTIPLIERS[prop_type]}** -> SD = **{sd_val:.2f}**")
    p_sym = 1.0 - norm.cdf(main_line, loc=proj_mean, scale=sd_val)
    mu, sigma = lognormal_params(proj_mean, sd_val)
    skew_median = math.exp(mu)
    p_skew = 1.0 - norm.cdf(math.log(main_line), loc=mu, scale=sigma)
    ev_sym = calc_ev(p_sym, main_odds)
    ev_skew = calc_ev(p_skew, main_odds)
    s1, s2, s3 = st.columns(3)
    s1.metric("Symmetric Over % (Naive)", f"{p_sym*100:.1f}%", f"EV: {ev_sym:+.1f}%")
    s2.metric("Skewed Median (Typical Game)", f"{skew_median:.1f}", f"vs Mean {proj_mean:.1f}")
    s3.metric("Right-Skewed Over % (Honest)", f"{p_skew*100:.1f}%", f"EV: {ev_skew:+.1f}%")
    st.markdown("#### Fair Alternate Ladder (Compare Against DK & FD)")
    step_size = 10.0 if proj_mean >= 40 else 5.0
    base_rung = max(step_size, math.floor((proj_mean * 0.55) / step_size) * step_size) - 0.5
    rungs = [base_rung + i * step_size for i in range(7)]
    ladder_rows = []
    for r in rungs:
        p_r = 1.0 - norm.cdf(math.log(r), loc=mu, scale=sigma)
        ladder_rows.append({
            "Alt Line": f"{r + 0.5:.0f}+ ({r:.1f})",
            "Skewed P(Over)": f"{p_r*100:.1f}%",
            "Fair Over Odds": f"{prob_to_amer(p_r):+d}",
            "Fair Under Odds": f"{prob_to_amer(1 - p_r):+d}",
        })
    st.dataframe(pd.DataFrame(ladder_rows), width="stretch")

with tab4:
    st.subheader("Promo Converter & Fractional Kelly Sizer")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("##### Fractional Kelly Sizer")
        k_prob = st.slider("Your True Win Prob (%)", 10, 90, 56) / 100.0
        k_odds = st.number_input("Posted Odds", value=100, step=5)
        kelly_frac = st.select_slider("Kelly Fraction", options=[0.25, 0.5, 1.0], value=0.25)
        b = amer_to_dec(k_odds) - 1.0
        full_k = max(0.0, (b * k_prob - (1.0 - k_prob)) / b)
        rec_stake = round(total_roll * full_k * kelly_frac, 2)
        st.metric("Recommended Stake", f"${rec_stake:.2f}", f"Full Kelly: {full_k*100:.2f}%")
        if rec_stake > max_single_bet:
            st.warning(f"Capped at 5% guardrail: **${max_single_bet:.2f}**")
    with pc2:
        st.markdown("##### Bonus Bet / Site Credit Hedge Converter")
        promo_type = st.selectbox("Promo Type", ["Bonus Bet (Stake Not Returned)", "Site Credit (Stake Returned)"])
        promo_amt = st.number_input("Promo Amount ($)", min_value=5.0, value=50.0, step=5.0)
        promo_odds = st.number_input("Promo Side Odds (+)", min_value=100, value=300, step=10)
        hedge_odds = st.number_input("Hedge Side Odds (-)", max_value=-100, value=-350, step=10)
        p_dec = amer_to_dec(promo_odds)
        h_dec = amer_to_dec(hedge_odds)
        gross_win = promo_amt * (p_dec - 1.0) if "Bonus Bet" in promo_type else promo_amt * p_dec
        hedge_stake = gross_win / h_dec
        guaranteed_profit = gross_win - hedge_stake
        conv_rate = (guaranteed_profit / promo_amt) * 100.0
        st.metric("Hedge Stake Needed", f"${hedge_stake:.2f}")
        st.metric("Guaranteed Cash Profit", f"${guaranteed_profit:.2f}", f"{conv_rate:.1f}% Conversion")

with tab5:
    st.subheader("Bet Log, Strategy ROI & Closing Line Value (CLV)")
    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
    else:
        df_log = pd.DataFrame(columns=[
            "Date", "Platform", "Strategy", "Pick", "Stake", "Odds", "Closing_Odds", "Result", "Profit", "CLV_Pct"
        ])
    with st.form("log_bet_form", clear_on_submit=True):
        f1, f2, f3, f4 = st.columns(4)
        b_date = f1.date_input("Date")
        b_plat = f2.selectbox("Platform", ["FanDuel", "DraftKings", "PrizePicks", "Kalshi", "BetMGM", "Fanatics"])
        b_strat = f3.selectbox("Strategy", ["Combos/Singles", "Alt Ladder", "5/6-Pick Flex", "Promo/Boost", "Neg-Corr SGP"])
        b_pick = f4.text_input("Pick Description")
        f5, f6, f7, f8 = st.columns(4)
        b_stake = f5.number_input("Stake ($)", min_value=0.5, value=2.0, step=0.5)
        b_odds = f6.number_input("Bet Odds", value=-110, step=5)
        b_close = f7.number_input("Closing Odds (for CLV)", value=-115, step=5)
        b_res = f8.selectbox("Result", ["Pending", "Win", "Loss", "Void"])
        submitted = st.form_submit_button("Log Bet")
        if submitted and b_pick:
            profit = 0.0
            if b_res == "Win":
                profit = round(b_stake * (amer_to_dec(b_odds) - 1.0), 2)
            elif b_res == "Loss":
                profit = -b_stake
            clv_pct = round((amer_to_prob(b_close) - amer_to_prob(b_odds)) * 100.0, 2)
            new_row = pd.DataFrame([{
                "Date": str(b_date), "Platform": b_plat, "Strategy": b_strat,
                "Pick": b_pick, "Stake": b_stake, "Odds": b_odds,
                "Closing_Odds": b_close, "Result": b_res, "Profit": profit, "CLV_Pct": clv_pct
            }])
            df_log = pd.concat([df_log, new_row], ignore_index=True)
            df_log.to_csv(LOG_FILE, index=False)
            st.success("Bet logged to bet_log.csv!")
    if not df_log.empty:
        settled = df_log[df_log["Result"].isin(["Win", "Loss"])]
        tot_staked = settled["Stake"].sum()
        tot_profit = settled["Profit"].sum()
        roi = (tot_profit / tot_staked * 100.0) if tot_staked > 0 else 0.0
        avg_clv = df_log["CLV_Pct"].mean()
        t1, t2, t3 = st.columns(3)
        t1.metric("Net Profit", f"${tot_profit:+.2f}")
        t2.metric("Overall ROI", f"{roi:+.1f}%")
        t3.metric("Avg CLV", f"{avg_clv:+.2f}%")
        st.dataframe(df_log, width="stretch")

