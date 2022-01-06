.. _workflow:

============
开发流程概述
============

.. _pull-request-guide:

Pull Request 的一生
-------------------

假设你已经 ``fork`` 并 ``clone`` 了 ``<your-username>/MegEngine`` 的代码：

.. code-block:: shell

   # 将 MegEngine 官方存储库添加为上游分支
   git remote add upstream git@github.com:MegEngine/Documentation.git
   git remote -v

   # 基于上游的 master 分支，创建并切换到一个新分支用于开发
   git fetch upstream
   git checkout -b <branch-name> upstream/master

   # 做一些相应的代码修改
   # ....

   # 做一些相应的本地测试

   # 将修改过的代码提交记录到本地，并推送到默认的远端分支
   git add <filenames>
   git commit -m '<message>'
   git push origin <branch-name>

   # 跑到 https://github.com/<your-username>/MegEngine
   # 根据相应的提示创建一个 Pull Request 并留下说明
   # 开发者将 Review 你的代码并给出相应的答复
   # 如果无需新的改动，则会被合入上游 master 分支

   # 删除本地分支和远端分支
   git branch -D <branch-name>
   git push origin -d <branch-name>

如果你在发起 Pull Request 后有新增的改动，如果 Commits 记录很混乱，
为了保持历史记录的整洁，请使用 ``squash`` 或者 ``rebase`` 进行整理。

.. note::

   一些开源项目要求避免使用 ``squash`` 和 ``rebase``
   以保证历史记录的完整性，在 MegEngine 中提倡使用这些操作，
   目的是防止出现过多的 ``fix`` 性质的提交历史，请勿滥用。

保持和上游同步
--------------

如果上游 ``master`` 分支有更新，为了避免存在冲突导致无法合并，
请同时更新你的本地 ``master`` 分支：

.. code-block:: shell

   git checkout master
   git pull upstream master  # pull = fetch + merge

接下来需要用 ``rebase`` 命令更新你所开发的分支：

.. code-block:: shell

   git checkout <branch-name>
   git rebase master
   git push origin --force <branch-name>

这个过程中可能需要解决一些冲突。

.. note::

   如果你已经用 ``git push origin <branch-name>`` 将开发分支推送到远端，
   则在进行 ``rebase`` 后再次 ``push`` 时需要添加 ``--force`` 参数。

仅修改最后的提交
----------------

有些时候你只需要小修小补最后一次的提交，但不希望多出一次提交记录，可以使用：

.. code-block:: shell

   git add <filenames>
   git commit --amend
   git push origin --force <brach-name>
