#!/usr/bin/env bash
# The project board, as JSON, from a short-lived shared cache.
#
# Two readers want the same answer: `board-state.sh` (once per turn) and `commit-msg` (once per
# commit and once per merge). Hitting the GitHub API from each of them earns a secondary
# rate-limit -- observed on 2026-09-20, and `gh` reports it as the unhelpful `unknown owner
# type`. A rate-limited board gate is a gate that blocks correct work, which is the failure that
# gets a guard switched off.
#
# Fetched over the REST Projects v2 API (`/users/Furkan-rgb/projectsV2/3/...`), not GraphQL --
# GraphQL has its own, separately-exhausted points bucket, observed pinned for hours under
# concurrent agent load on 2026-09-24 while REST calls on the same token kept working. The
# Status field id and its option ids are resolved once per refresh and kept in the cached JSON
# alongside the items, so a writer reading this cache pays nothing extra for them.
#
# So: one file, 60 seconds, shared by uid. That is short enough that a status moved on the board
# is honoured within a minute, and long enough that a burst of commits costs one API call.
#
# ── the contract, which callers must honour ───────────────────────────────────────────────────
#
#   exit 0 — the JSON on stdout, fetched just now. Trust it.
#   exit 3 — the JSON on stdout, but it is STALE: the refresh failed and this is the last good
#            copy. Trust a status it reports, but not an absence. An item created since the
#            cache was written is missing from it, and refusing a commit for citing an item
#            that "is not on the board" would be refusing on evidence we know is out of date.
#   exit 1 — nothing usable on stdout, the reason on stderr. Skip the check entirely.
#
# Exit 3 rather than a `{"stale":true}` wrapper because the readers already branch on the exit
# code, and wrapping would change the shape of the JSON for every caller to serve one of them.
#
# ── where the file lives ──────────────────────────────────────────────────────────────────────
#
# `XDG_RUNTIME_DIR` when it exists: the standard per-user, mode-700 location. Otherwise a
# directory we create ourselves under TMPDIR with mode 700 and verify -- /tmp is world-writable,
# and a board file another user can rewrite is a board file that can be made to say In Progress.
# If either is unusable the cache is simply skipped; the fetch still happens, it is just not kept.
set -u

OWNER="Furkan-rgb"
PROJECT="3"
MAX_AGE=60

note() { printf '%s\n' "$1" >&2; }

# ── locate the cache file, or decide there is none ────────────────────────────────────────────
CACHE=""
uid="$(id -u)"
board_tag="board-$OWNER-$PROJECT-$uid"
if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
    # Set but absent is a real configuration on some systems. Treat it as "no cache" rather than
    # letting a redirect fail with a bare shell error the caller cannot interpret.
    if [ -d "$XDG_RUNTIME_DIR" ] && [ -w "$XDG_RUNTIME_DIR" ]; then
        CACHE="$XDG_RUNTIME_DIR/$board_tag.json"
    else
        note "board cache disabled: XDG_RUNTIME_DIR is set to '$XDG_RUNTIME_DIR', which is not a writable directory"
    fi
else
    dir="${TMPDIR:-/tmp}/$board_tag"
    if [ -d "$dir" ] || mkdir -m 700 "$dir" 2>/dev/null; then
        owner="$(stat -c '%u' "$dir" 2>/dev/null || echo "")"
        mode="$(stat -c '%a' "$dir" 2>/dev/null || echo "")"
        if [ "$owner" = "$uid" ] && [ "$mode" = "700" ]; then
            CACHE="$dir/board.json"
        else
            note "board cache disabled: $dir is owned by uid '$owner' with mode '$mode', not $uid/700"
        fi
    fi
fi

# `--force` (used by commit-msg's direct_status fallback, once, for one item it could not find
# in the cached copy) skips the freshness check and always re-fetches.
force=0
[ "${1:-}" = "--force" ] && force=1

fresh_cache() {
    [ "$force" = "1" ] && return 1
    [ -n "$CACHE" ] && [ -s "$CACHE" ] || return 1
    mtime=$(date -r "$CACHE" +%s 2>/dev/null || echo 0)
    [ $(($(date +%s) - mtime)) -lt "$MAX_AGE" ]
}

# Serve without touching the network when the copy on disk is young enough.
if fresh_cache; then
    cat "$CACHE"
    exit 0
fi

serve_stale() {
    if [ -n "$CACHE" ] && [ -s "$CACHE" ]; then
        note "board cache is stale ($1); using the last good copy"
        cat "$CACHE"
        exit 3
    fi
    note "$1"
    exit 1
}

missing=""
command -v gh >/dev/null 2>&1 || missing="gh not installed"
command -v jq >/dev/null 2>&1 || missing="jq not installed"
[ -n "$missing" ] && serve_stale "$missing"

# umask 077: the fetched board is readable only by us, wherever it lands.
umask 077
tmp="$(mktemp "${CACHE:-${TMPDIR:-/tmp}/$board_tag}.XXXXXX" 2>/dev/null)" || tmp=""
[ -n "$tmp" ] || serve_stale "cannot create a temporary file"

# The Status field id (and its option ids, which a writer needs) first -- one small call,
# `--paginate` already flattens a REST array endpoint's pages into one JSON array.
fields_json="$(timeout 12 gh api "/users/$OWNER/projectsV2/$PROJECT/fields" \
    -H "X-GitHub-Api-Version: 2022-11-28" --paginate 2>/dev/null)"
status_field_id="$(printf '%s' "$fields_json" | jq -r '[.[] | select(.name=="Status")][0].id // empty' 2>/dev/null)"
if [ -z "$status_field_id" ]; then
    rm -f "$tmp"
    serve_stale "could not resolve the Status field id over REST"
fi
status_options="$(printf '%s' "$fields_json" | jq -c --arg fid "$status_field_id" \
    '[.[] | select((.id|tostring)==$fid)][0].options // []' 2>/dev/null)"

items_json="$(timeout 12 gh api "/users/$OWNER/projectsV2/$PROJECT/items?fields[]=$status_field_id&per_page=100" \
    -H "X-GitHub-Api-Version: 2022-11-28" --paginate 2>/dev/null)"
if [ -z "$items_json" ]; then
    rm -f "$tmp"
    serve_stale "items fetch failed or timed out"
fi

# Reshape into the same JSON the readers already consume (items[].status, .content.number,
# .content.title) so `board-state.sh` and `commit-msg` need no change on the read side.
jq -c --argjson opts "${status_options:-[]}" --arg fid "$status_field_id" '
  {
    status_field_id: ($fid | tonumber),
    status_options: $opts,
    items: [ .[] | {
      status: ([.fields[]? | select(.name=="Status") | .value.name.raw] | first),
      content: (if .content then {number: .content.number, title: .content.title} else null end),
      title: (.content.title // ([.fields[]? | select(.name=="Title") | .value] | first) // null),
      project_item_id: .id
    } ]
  }' <<<"$items_json" >"$tmp" 2>/dev/null

if ! [ -s "$tmp" ] || ! jq -e '.items' <"$tmp" >/dev/null 2>&1; then
    rm -f "$tmp"
    serve_stale "items output not understood"
fi

# Keeping the copy is a best effort; answering the caller is not.
if [ -n "$CACHE" ] && mv -f "$tmp" "$CACHE" 2>/dev/null; then
    cat "$CACHE"
else
    cat "$tmp"
    rm -f "$tmp"
fi
exit 0
