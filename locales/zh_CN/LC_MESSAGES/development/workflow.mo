��          �               �      �   �        �     �  U     �   h  �   �  �   �     N  B   a  r   �  0     �  H     �  �   �     �     �  U   �  �   M  �   �  �   �	     3
  B   F
  r   �
  0   �
   Pull Request 的一生 一些开源项目要求避免使用 ``squash`` 和 ``rebase`` 以保证历史记录的完整性，在 MegEngine 中提倡使用这些操作， 目的是防止出现过多的 ``fix`` 性质的提交历史，请勿滥用。 仅修改最后的提交 保持和上游同步 假设你已经 ``fork`` 并 ``clone`` 了 ``<your-username>/MegEngine`` 的代码： 如果上游 ``master`` 分支有更新，为了避免存在冲突导致无法合并， 请同时更新你的本地 ``master`` 分支： 如果你在发起 Pull Request 后有新增的改动，如果 Commits 记录很混乱， 为了保持历史记录的整洁，请使用 ``squash`` 或者 ``rebase`` 进行整理。 如果你已经用 ``git push origin <branch-name>`` 将开发分支推送到远端， 则在进行 ``rebase`` 后再次 ``push`` 时需要添加 ``--force`` 参数。 开发流程概述 接下来需要用 ``rebase`` 命令更新你所开发的分支： 有些时候你只需要小修小补最后一次的提交，但不希望多出一次提交记录，可以使用： 这个过程中可能需要解决一些冲突。 Project-Id-Version: MegEngine 1.3.0
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2021-04-09 17:59+0800
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_Hans_CN
Language-Team: zh_Hans_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.4.0
 Pull Request 的一生 一些开源项目要求避免使用 ``squash`` 和 ``rebase`` 以保证历史记录的完整性，在 MegEngine 中提倡使用这些操作， 目的是防止出现过多的 ``fix`` 性质的提交历史，请勿滥用。 仅修改最后的提交 保持和上游同步 假设你已经 ``fork`` 并 ``clone`` 了 ``<your-username>/MegEngine`` 的代码： 如果上游 ``master`` 分支有更新，为了避免存在冲突导致无法合并， 请同时更新你的本地 ``master`` 分支： 如果你在发起 Pull Request 后有新增的改动，如果 Commits 记录很混乱， 为了保持历史记录的整洁，请使用 ``squash`` 或者 ``rebase`` 进行整理。 如果你已经用 ``git push origin <branch-name>`` 将开发分支推送到远端， 则在进行 ``rebase`` 后再次 ``push`` 时需要添加 ``--force`` 参数。 开发流程概述 接下来需要用 ``rebase`` 命令更新你所开发的分支： 有些时候你只需要小修小补最后一次的提交，但不希望多出一次提交记录，可以使用： 这个过程中可能需要解决一些冲突。 