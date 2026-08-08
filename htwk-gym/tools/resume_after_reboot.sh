#!/usr/bin/env bash
# 서버가 돌아왔을 때 **한 번** 돌리는 복구 스크립트. 멱등하다 -- 이미 살아 있는 것은
# 건드리지 않고, 없는 것만 띄운다.
#
#   ssh a6000 'bash /mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym/tools/resume_after_reboot.sh'
#
# ---- 왜 필요한가 -------------------------------------------------------------
#
# 2026-08-08 18:09 에 서버가 45분 넘게 접속 불가가 됐다(ping·포트 6666 무응답,
# 이쪽 네트워크는 정상). 그때 확인된 것: **재부팅되면 아무것도 자동으로 안 뜬다.**
# 감시자(idle_watch·autopilot)도, 워커 4개도 전부 사람이 ssh 로 띄운 프로세스다.
# 즉 서버가 살아 돌아와도 **큐에 대기 작업이 있는 채로 GPU 두 장이 논다** --
# 13.8시간을 날린 §8-44 와 결과가 같은 상태다.
#
# 그래서 복구를 기억이 아니라 **코드**에 둔다. 이 파일 하나가 순서를 안다.
#
# ⚠️ 이 스크립트는 실제 재부팅 뒤에서 검증된 적이 없다(작성 시점에 서버가 죽어
#    있었다). 파괴적 동작은 하나도 없다 -- 죽이지 않고, 지우지 않고, 없는 것만 띄운다.
#
# `set -u` 는 쓰지 않는다 -- conda.sh 가 미설정 변수를 참조해 활성화 전에 죽는다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
CONDA_SH="${CONDA_SH:-/mnt/DATA/workspace/ws_eungkyu/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-k1goalpose}"

say() { printf '[resume %s] %s\n' "$(TZ=Asia/Seoul date +'%F %T')" "$*"; }

say "복구 시작: $ROOT"

# ---- 0. 전제 확인 -----------------------------------------------------------
# ⛔ conda 를 못 찾으면 워커가 떠도 작업이 전부 `python: command not found` 로 죽는다.
# 그건 "돌고 있는데 아무것도 안 되는" 최악의 모드다. 여기서 먼저 막는다.
if [ ! -f "$CONDA_SH" ]; then
    say "⛔ conda.sh 가 없다: $CONDA_SH -- 워커를 띄워도 작업이 못 돈다. 중단."
    exit 1
fi
if ! nvidia-smi -L > /dev/null 2>&1; then
    say "⛔ nvidia-smi 가 안 된다. 드라이버가 아직 안 올라왔을 수 있다. 중단."
    exit 1
fi
say "GPU: $(nvidia-smi -L | wc -l) 장"

# ---- 1. 레인별로 살아 있는 워커를 센다 ---------------------------------------
# 레인은 스크립트 이름이 아니라 환경변수 LANE 에 있다. `gpu_worker.sh 0` 은 big 으로도
# small 로도 돈다 -- 이름만 보면 작은 레인 워커를 큰 레인 워커로 오인한다.
worker_alive() {          # $1 = 카드, $2 = 레인(big|small)
    local g="$1" want="$2" p lane
    for p in $(ps -eo pid=,args= | awk -v g="$g" '$3 ~ /tools\/gpu_(queue|worker)\.sh$/ && $4 == g {print $1}'); do
        lane=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^LANE=//p' | head -1)
        [ -z "$lane" ] && lane=big
        [ "$lane" = "$want" ] && return 0
    done
    return 1
}

for g in 0 1; do
    if worker_alive "$g" big; then
        say "큰 레인 워커 $g: 살아 있음"
    else
        ( cd "$ROOT" && setsid nohup bash tools/gpu_queue.sh "$g" \
            < /dev/null >> "queue/worker$g.log" 2>&1 & )
        say "큰 레인 워커 $g: 기동"
    fi
    if worker_alive "$g" small; then
        say "작은 레인 워커 $g: 살아 있음"
    else
        ( cd "$ROOT" && LANE=small setsid nohup bash tools/gpu_worker.sh "$g" \
            < /dev/null >> "queue/worker_small$g.log" 2>&1 & )
        say "작은 레인 워커 $g: 기동"
    fi
done

# ---- 2. 감시자 둘 ------------------------------------------------------------
for n in idle_watch autopilot; do
    # 상대경로(`bash tools/idle_watch.sh`)로도 절대경로로도 떠 있을 수 있다.
    # `$2 == "tools/idle_watch.sh"` 로만 보면 절대경로로 뜬 것을 못 보고 **중복 기동**한다
    # (감시자 둘이 같은 idle_state.json 을 쓰게 된다). 끝만 맞추면 둘 다 잡힌다.
    if ps -eo args= | awk -v s="$n" '$2 ~ ("(^|/)tools/" s "\\.sh$") {found=1} END{exit !found}'; then
        say "$n: 살아 있음"
    else
        ( cd "$ROOT" && setsid nohup bash "tools/$n.sh" \
            < /dev/null >> "queue/$n.log" 2>&1 & )
        say "$n: 기동"
    fi
done

# ---- 3. 고아 `.running` 표식 --------------------------------------------------
# 워커가 작업 도중 죽으면(= 재부팅) 표식이 남고 **그 작업은 영원히 사라진다** --
# 큐에도 없고 done 에도 완료로 안 남는다. 아무 신호도 안 나가는 모드다.
#
# ⛔ 되돌리기 전에 **정말 안 도는지** 확인한다. 학습이 살아 있는데 표식을 큐로
# 되돌리면 같은 작업이 한 카드에서 둘 돈다. 재부팅 직후에는 안 돌지만, 이 스크립트를
# 실수로 학습 중에 돌릴 수도 있다.
sleep 3   # 방금 띄운 워커가 큐에서 작업을 집을 틈을 준다
NPROC=$(ps -eo comm=,args= | awk '$1 ~ /^python/ && /train_v7\.py|eval_goal_pose\.py|select_best_checkpoint\.py/ {n++} END{print n+0}')
if [ "$NPROC" -gt 0 ]; then
    say "GPU 작업 ${NPROC}개가 돌고 있다 -- 고아 표식 복구는 건너뛴다(중복 실행 방지)."
else
    n_recovered=0
    for f in "$ROOT/queue/done"/*.running; do
        [ -e "$f" ] || continue
        name=$(basename "$f" .running)
        own="$ROOT/queue/done/$name.owner"
        dest="$ROOT/queue/gpu0"
        if [ -f "$own" ]; then
            read -r _opid ogpu olane < "$own"
            [ -z "$ogpu" ] && ogpu=0
            if [ "$olane" = "small" ]; then dest="$ROOT/queue/small/gpu$ogpu"
            else                            dest="$ROOT/queue/gpu$ogpu"; fi
            mkdir -p "$dest"
        fi
        if mv "$f" "$dest/$name.sh" 2>/dev/null; then
            rm -f "$own"
            say "고아 표식 복구: $name -> ${dest#$ROOT/}/"
            n_recovered=$((n_recovered + 1))
        fi
    done
    [ "$n_recovered" -eq 0 ] && say "고아 표식 없음"
fi

# ---- 4. 상태를 찍는다 ---------------------------------------------------------
say "복구 끝. 아래는 현재 상태다."
echo
bash "$ROOT/tools/round_status.sh"
