#!/usr/bin/env bash
# Poll both grouped-topk A/B experiments until each writes its DONE flag, then
# pull the evalscope eval-out dirs down and drop a READY marker so the parent
# session can run compare_gtopk_eval.py.
set -uo pipefail

DIS_EXP="exp-h8r2wk3zr1"   # kernel OFF (pure JAX)
EN_EXP="exp-cvs4sd0t7c"    # kernel ON  (Pallas)
REMOTE_BASE="/tmp/sglang-jax/eval-out"
LOCAL_DIR="./eval-results"
POLL_SECS=300
MAX_HOURS=6

mkdir -p "$LOCAL_DIR"
LOG="$LOCAL_DIR/poll.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# remote_done <exp> <variant> -> 0 if DONE flag present
remote_done() {
  falcon exp exec "$1" --rank 0 -- test -f "$REMOTE_BASE/$2/DONE" >/dev/null 2>&1
}
# exp_failed <exp> -> 0 if exp is in a terminal failed/aborted state
exp_failed() {
  falcon exp get "$1" --output json 2>/dev/null \
    | grep -Eiq '"status"[[:space:]]*:[[:space:]]*"(failed|aborted|error)"'
}
# fetch <exp> <variant>: tar the eval-out on the box, cp down, extract
fetch() {
  local exp="$1" v="$2"
  say "fetching $v from $exp ..."
  falcon exp exec "$exp" --rank 0 -- \
    tar czf "/tmp/$v.tgz" -C "$REMOTE_BASE" "$v" >>"$LOG" 2>&1
  falcon exp cp "$exp:/tmp/$v.tgz" "$LOCAL_DIR/$v.tgz" >>"$LOG" 2>&1
  tar xzf "$LOCAL_DIR/$v.tgz" -C "$LOCAL_DIR" >>"$LOG" 2>&1 \
    && say "extracted -> $LOCAL_DIR/$v" \
    || say "WARN: extract failed for $v (see $LOG)"
}

say "polling every ${POLL_SECS}s (max ${MAX_HOURS}h): dis=$DIS_EXP en=$EN_EXP"
deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
dis_ok=0; en_ok=0
while :; do
  if [ "$dis_ok" = 0 ] && remote_done "$DIS_EXP" disabled; then dis_ok=1; say "disabled DONE"; fi
  if [ "$en_ok"  = 0 ] && remote_done "$EN_EXP"  enabled;  then en_ok=1;  say "enabled DONE";  fi

  if [ "$dis_ok" = 1 ] && [ "$en_ok" = 1 ]; then
    say "both DONE -> fetching results"
    fetch "$DIS_EXP" disabled
    fetch "$EN_EXP"  enabled
    touch "$LOCAL_DIR/READY"
    say "READY. run: python compare_gtopk_eval.py $LOCAL_DIR/disabled $LOCAL_DIR/enabled"
    exit 0
  fi

  # bail early only if an exp hit a terminal failure before finishing
  if [ "$dis_ok" = 0 ] && exp_failed "$DIS_EXP"; then say "!! $DIS_EXP FAILED"; touch "$LOCAL_DIR/FAILED"; exit 1; fi
  if [ "$en_ok"  = 0 ] && exp_failed "$EN_EXP";  then say "!! $EN_EXP FAILED";  touch "$LOCAL_DIR/FAILED"; exit 1; fi

  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "!! timeout after ${MAX_HOURS}h (dis_ok=$dis_ok en_ok=$en_ok)"; touch "$LOCAL_DIR/TIMEOUT"; exit 2
  fi
  sleep "$POLL_SECS"
done
