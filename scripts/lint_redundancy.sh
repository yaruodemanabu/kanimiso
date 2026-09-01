#!/usr/bin/env bash
# 冗長性 lint（R1 / R4 / R5 / R7 / R10）+ スタック拡張 + faer 単一版。全て ratchet 予算。
set -uo pipefail
SRC="kanimiso/src signlred/src ojizou-san/src oldwood/src mayoi-no-mori/src"
ALLOW=scripts/lint_allowlist.txt        # 固有名詞の数値（Exp3, Ucb1, Catch22, X13, 2Sls, Chi2, F1, R2 …）を 1 行 1 固定文字列
[ -f "$ALLOW" ] || : > "$ALLOW"

# ---- 予算（下げるのみ。初期値 = 2026-09-01 実測）-----------------------------
MAX_R1_NUMERAL_IDENTS=915      # 目標 0  （Garch11 のような「パラメータ値」は allowlist に入れない）
MAX_R4_PUB_ITEMS=254           # 目標 1000; PR 11 measured after Fit::fit(&self)
MAX_R5_FILES_OVER_3000=12      # 目標 0
MAX_R7_DENSITY_FLOORS=156      # 目標 0; clamps moved into mayoi named constants
MAX_R10_DISTINCT_FROM=741      # 目標 0
ALLOW_RUST_MIN_STACK=0         # generated HMM gone; no stack hack
# ------------------------------------------------------------------------------
fail=0
budget() { # name actual max
  if [ "$2" -gt "$3" ]; then echo "FAIL $1: $2 > budget $3"; fail=1; else echo "ok   $1: $2 (budget $3)"; fi
}

r1=$(grep -rhE '^\s*pub (struct|enum|trait|fn|type|mod) [A-Za-z_]*[0-9]+[A-Za-z_0-9]*' --include='*.rs' $SRC \
     | grep -vF -f "$ALLOW" | wc -l)
budget "R1 numeral in public identifier" "$r1" "$MAX_R1_NUMERAL_IDENTS"

r4=$(grep -rhoE '^\s*pub (struct|enum|trait|fn|type|const|mod|static) ' --include='*.rs' kanimiso/src | wc -l)
budget "R4 pub items (kanimiso)" "$r4" "$MAX_R4_PUB_ITEMS"

r5=$(find $SRC -name '*.rs' | xargs wc -l | grep -v ' total$' | awk '$1>3000' | wc -l)
budget "R5 files over 3000 lines" "$r5" "$MAX_R5_FILES_OVER_3000"

r7=$(grep -rhE '\.max\(1e-[0-9]+\)\.ln\(\)|\.clamp\(0\.0, 1\.0 - 1e' --include='*.rs' $SRC | wc -l)
budget "R7 density floor / probability clamp" "$r7" "$MAX_R7_DENSITY_FLOORS"

r10=$(grep -rh 'Distinct from \[`' --include='*.rs' $SRC | wc -l)
budget "R10 'Distinct from' docstrings" "$r10" "$MAX_R10_DISTINCT_FROM"

if [ "$ALLOW_RUST_MIN_STACK" -eq 0 ] && grep -q RUST_MIN_STACK .cargo/config.toml 2>/dev/null; then
  echo "FAIL RUST_MIN_STACK is forbidden"; fail=1; fi

# Invert tree lists dependents too; count distinct `faer v…` roots (D6).
faer_versions=$(cargo tree -i faer -e normal --prefix none 2>/dev/null | awk '/^faer /' | sort -u | wc -l)
if [ "$faer_versions" -gt 1 ]; then echo "FAIL multiple faer versions in the graph"; fail=1; fi

exit $fail
