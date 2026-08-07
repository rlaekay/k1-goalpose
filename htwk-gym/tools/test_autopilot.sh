#!/usr/bin/env bash
# autopilot 의 승격·복구 경로를 **격리된 사본**에서 실제로 실행해 본다.
#
#   bash tools/test_autopilot.sh
#
# ⛔ 왜 필요한가: 2026-08-07 기준 autopilot 은 켜진 지 13시간이 지나도록 승격을
# **한 번도 하지 않았다.** 유휴가 감지된 적이 없기 때문인데(그 원인은 pgrep 이
# 셸까지 세던 버그), 그 말은 승격 코드가 한 번도 실행된 적이 없다는 뜻이다.
# 정작 필요할 때 처음 도는 코드는 신뢰할 수 없다.
#
# 라이브 큐를 건드리지 않는다: 임시 디렉터리에 queue 구조를 복제하고 거기서 돌린다.

set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/tools" "$TMP/queue/gpu0" "$TMP/queue/gpu1" \
         "$TMP/queue/plan/gpu0" "$TMP/queue/plan/gpu1" "$TMP/queue/done"
cp "$SRC/tools/autopilot.sh" "$TMP/tools/"

pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then echo "  ✅ $1"; pass=$((pass+1));
        else echo "  ⛔ $1  (기대 '$3', 실제 '$2')"; fail=$((fail+1)); fi; }

state() {   # state <idle_seconds> <stall_seconds>
    cat > "$TMP/queue/idle_state.json" <<EOF
{"idle_seconds": $1, "stall_seconds": $2}
EOF
}

run_once() {   # autopilot 을 한 바퀴만 돌린다
    ( cd "$TMP" && timeout 12 env PROMOTE_AFTER="${PROMOTE_AFTER:-1800}" INTERVAL=600 \
        bash tools/autopilot.sh >/dev/null 2>&1 || true )
}

echo "== 1. 유휴가 아니면 아무것도 안 한다 =="
echo x > "$TMP/queue/plan/gpu0/900-a.sh"
state 0 0
run_once
chk "plan 이 그대로다" "$(ls -1 "$TMP/queue/plan/gpu0" | wc -l | tr -d ' ')" "1"
chk "live 가 비어 있다" "$(ls -1 "$TMP/queue/gpu0" | wc -l | tr -d ' ')" "0"

echo "== 2. 유휴 30분이면 plan 에서 하나 승격한다 =="
echo x > "$TMP/queue/plan/gpu0/910-b.sh"
state 1900 0
run_once
chk "live 에 하나 올라왔다" "$(ls -1 "$TMP/queue/gpu0" | wc -l | tr -d ' ')" "1"
chk "사전순 첫 번째가 올라왔다" "$(ls -1 "$TMP/queue/gpu0")" "900-a.sh"
chk "plan 에 하나 남았다" "$(ls -1 "$TMP/queue/plan/gpu0" | wc -l | tr -d ' ')" "1"

echo "== 3. live 에 대기가 있으면 더 승격하지 않는다 =="
state 1900 0
run_once
chk "live 는 여전히 하나" "$(ls -1 "$TMP/queue/gpu0" | wc -l | tr -d ' ')" "1"

echo "== 4. 정체면 고아 .running 을 큐로 되돌린다 =="
rm -f "$TMP/queue/gpu0/"*.sh
echo x > "$TMP/queue/done/030-orphan.running"
state 0 400
run_once
chk "고아가 큐로 복구됐다" "$(ls -1 "$TMP/queue/gpu0")" "030-orphan.sh"
chk "표식이 사라졌다" "$(ls -1 "$TMP/queue/done"/*.running 2>/dev/null | wc -l | tr -d ' ')" "0"

echo "== 5. 정체 복구가 승격보다 먼저다 =="
# 유휴와 정체가 동시에 참일 수는 없지만, 정체가 참이면 승격 블록에 도달하면 안 된다.
rm -f "$TMP/queue/gpu0/"*.sh
echo x > "$TMP/queue/plan/gpu0/920-c.sh"
echo x > "$TMP/queue/done/031-orphan2.running"
state 1900 400
run_once
chk "승격 대신 복구가 일어났다" "$(ls -1 "$TMP/queue/gpu0")" "031-orphan2.sh"
chk "plan 은 안 건드렸다" "$(ls -1 "$TMP/queue/plan/gpu0" | wc -l | tr -d ' ')" "2"

echo
echo "통과 $pass, 실패 $fail"
[ "$fail" -eq 0 ]
