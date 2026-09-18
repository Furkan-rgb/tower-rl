#!/usr/bin/env bash
# Prints the open (non-Done) items on the tower-rl task board (project #3),
# so every turn starts with the board's current state in context.
#
# Must never fail the turn: any problem (gh missing, unauthenticated,
# offline, malformed output) results in no output and a clean exit.
set -u

OWNER="Furkan-rgb"
PROJECT=3

command -v gh >/dev/null 2>&1 || exit 0
command -v python3 >/dev/null 2>&1 || exit 0

json="$(timeout 4 gh project item-list "$PROJECT" --owner "$OWNER" --format json --limit 100 2>/dev/null)" || exit 0
[ -n "$json" ] || exit 0

printf '%s' "$json" | python3 -c '
import json, sys

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

items = []
for it in data.get("items", []):
    status = it.get("status")
    if status in (None, "Done"):
        continue
    content = it.get("content", {}) or {}
    number = content.get("number")
    title = content.get("title")
    if number is None or not title:
        continue
    items.append((status, number, title))

if not items:
    sys.exit(0)

order = {"Next": 0, "In progress": 1, "Blocked": 2, "Backlog": 3}
items.sort(key=lambda r: (order.get(r[0], 99), r[1]))

print("Board (open items):")

if len(items) <= 12:
    for status, number, title in items:
        print(f"#{number} [{status}] {title}")
else:
    next_items = [r for r in items if r[0] == "Next"]
    rest = [r for r in items if r[0] != "Next"]
    for status, number, title in next_items:
        print(f"#{number} [{status}] {title}")
    if rest:
        print(f"...and {len(rest)} more in Backlog/Blocked, including:")
        for status, number, title in rest[:3]:
            print(f"#{number} [{status}] {title}")
' 2>/dev/null

exit 0
