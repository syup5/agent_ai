#!/bin/bash
# Stop hook: 작업 완료 후 자동으로 git push 수행
# .claude/settings.json의 Stop hook에서 호출됨

set -e

INPUT=$(cat)

# 프로젝트 디렉토리로 이동
cd "$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"

# git repo가 아니면 종료
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    exit 0
fi

# remote가 설정되어 있지 않으면 종료
if ! git remote get-url origin &>/dev/null; then
    exit 0
fi

# 커밋되지 않은 변경이 있으면 자동 커밋
if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    git add -A
    git commit -m "auto-commit: $(date '+%Y-%m-%d %H:%M:%S')" 2>/dev/null || true
fi

# push할 커밋이 있으면 push
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
git push origin "$BRANCH" 2>/dev/null || true

exit 0
