# 成为贡献者！

对 MegEngine 的 Tutorials 项目感兴趣？欢迎做出贡献 :clap:

我们欢迎各种形式的贡献，包括但不限于教程与文档的完善，Bug 反馈和特性需求等等...

## 使用 Sphinx 在本地生成网页

1. 克隆当前代码库到本地：
```shell
git clone https://github.com/MegEngine/Tutorials
cd Tutorials
```
2. 安装对应的依赖：
```shell
pip install -r requirementes.txt
```
3. 生成 HTML 网页：
```shell
make html
```
这将调用 `sphinx-build` 执行生成过程, 可在 `build/html/index.html` 找到主页。