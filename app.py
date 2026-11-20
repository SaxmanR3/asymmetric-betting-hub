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