#!/bin/bash -e
echo "This script CAN NOT execute directly, just an demonstration for this page: https://www.megengine.org.cn/doc/stable/zh/development/docs/maintainer-responsibility.html"
echo "Execute following command BY YOUR HAND after read the document!"
exit -1


OLD_VERSION="1.10"
CURRENT_VERSION="1.11"

echo "[BEGIN] Step1: backup current main branch"
git fetch
git checkout main
git reset --hard origin/main
git checkout -b "release/v$OLD_VERSION"
git push origin "release/v$OLD_VERSION"
echo "[END] Step1 done"

echo "[BEGIN] Step2: sync dev/main brain"
git checkout dev
git reset --hard origin/dev
git rebase origin/main
git checkout main
git merge --ff dev
echo "[END] Step2 done"

echo "[BEGIN] Step3: update version stuff"
# TODO How to update README???

# TODO How to update conf.py ???

# TODO How to update requirements.txt ???

git add README.md source/conf.py requirements.txt
git commit -m "chore: bump version"
echo "[END] Step3 done"

echo "[BEGIN] Step4: translation stuff"
make gettext
sphinx-intl update -p build/gettext -l zh_CN -l en
git add locales
git commit -m "trans: update po files"
echo "[END] Step4 done"

echo "[BEGIN] Step5: setup main/dev"
git push origin main
git branch -D dev
git checkout -b dev
git push origin dev -f
echo "[END] Step5 done"
