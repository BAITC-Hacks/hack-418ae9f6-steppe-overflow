#!/usr/bin/env bash
# Выкладка текущего коммита на сервер: git archive → ssh → docker compose up -d --build.
# Файл .env на сервере (ключи) не трогается.
#   DEPLOY_HOST=deploy@<сервер> ./deploy/deploy.sh
set -euo pipefail

HOST="${DEPLOY_HOST:-deploy@steppewind.energy}"
DIR="${DEPLOY_DIR:-steppewind}"

cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Есть незакоммиченные изменения — выкладывается только последний коммит." >&2
fi
REV=$(git rev-parse --short HEAD)
echo "Выкладываю $REV на $HOST:~/$DIR"
git archive --format=tar HEAD | ssh "$HOST" "mkdir -p $DIR && tar -x -C $DIR && echo $REV > $DIR/REVISION"
ssh "$HOST" "cd $DIR && docker compose up -d --build --remove-orphans && docker compose ps"
