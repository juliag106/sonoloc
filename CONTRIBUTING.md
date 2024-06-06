# 贡献指南

感谢参与 sonoloc！无论是修 bug、加特征还是补文档，都很欢迎。

## 开发环境

```bash
git clone https://github.com/juliag106/sonoloc.git
cd sonoloc
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## 本地检查

提交前请确保以下命令通过（与 CI 一致）：

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=sonoloc
```

## 代码风格

- 遵循 ruff 的 lint 与格式化配置，行宽 100。
- 公开函数请写类型注解与简短 docstring；新增算法请在注释或 `docs/design-notes.md`
  里给出参考文献或公式。
- 变量、函数、类名用英文；面向用户的文档用中文。

## 提交与 PR

- commit message 建议使用 [Conventional Commits](https://www.conventionalcommits.org/)
  风格（如 `feat(localization): ...`、`fix(metrics): ...`），但不强制。
- 一个 PR 聚焦一件事，附带对应的测试。
- 涉及算法或数值行为的改动，请说明如何验证（例如仿真回收 DOA 的误差范围）。

## 报告问题

请附上 sonoloc / Python 版本、音频通道数与采样率，以及最小复现脚本。
