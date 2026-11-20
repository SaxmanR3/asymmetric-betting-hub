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

    params = {
        "apiKey": API_KEY,
        "sportId": sport_id,
        "from": today.isoformat(),
        "to": (today + timedelta(days=days_ahead)).isoformat(),
        "hasOdds": "true",
    }
    try:
        r = requests.get(f"{BASE_URL}/fixtures", params=params, timeout=15)
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


def get_fixture_odds_cached(fid: str, sport_id: int, force_refresh: bool = False) -> Tuple[dict, dict, bool]:
    odds_cache = st.session_state.setdefault("odds_cache", {})
    catalog = load_market_catalog(sport_id)
    if not force_refresh and str(fid) in odds_cache:
        return odds_cache[str(fid)], catalog, True

    r = requests.get(
        f"{BASE_URL}/odds",
        params={"apiKey": API_KEY, "fixtureId": fid},
        timeout=25,
    )
    r.raise_for_status()
    b_odds = r.json().get("bookmakerOdds", {})
    odds_cache[str(fid)] = b_odds
    return b_odds, catalog, False


def scan_fixture_ev_cached(
    fid: str,
    b_odds: dict,
    catalog: dict,
    target_books: list,
    max_width: int,
    min_ev: float,
) -> Tuple[List[dict], List[dict]]:
    scan_cache = st.session_state.setdefault("scan_cache", {})
    cache_key = (
        str(fid),
        int(max_width),
        float(min_ev),
        tuple(sorted(b.lower() for b in target_books)),
        tuple(get_restricted_players()),
    )
    if cache_key not in scan_cache:
        scan_cache[cache_key] = scan_fixture_ev(b_odds, catalog, target_books, max_width, min_ev)
    return scan_cache[cache_key]