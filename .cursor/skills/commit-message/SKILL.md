---
name: commit-message
description: >-
  Writes Chinese Conventional Commits messages for this repo. Use when the user
  asks for a commit message、写一下 message、commit message、提交说明, or after a change
  they want to commit.
---

# Commit Message

用户原话：**直接给我能粘贴的 message 就行**；**要写明是 feat**（按 type 写，新功能必须是 feat）。

## 输出

只回一行，可直接粘贴进 git 的 message 框。不要 `git add` / `git commit`、不要代码块围命令、不要解释。

未说「提交 / commit / 帮我提交」时，不要执行 git commit。

## 格式

```
<type>: <中文说明>
```

- 必须中文。禁止英文主语（`Add` / `Fix` / `Initial`）。
- 必须带 type。新功能、新接口、新能力用 `feat`。
- 一行说清为什么，不写文件清单。
- 句末用句号。

| type | 何时 |
|---|---|
| feat | 新功能 |
| fix | 修 bug |
| docs | 只改文档 |
| refactor | 行为不变的结构调整 |
| chore | 依赖、配置、仓库杂务 |
| test | 只改测试 |

## 例子

改动：上传文档入库，`/ask` 从 stub 变成检索+生成。

```
feat: 接入最基础 RAG，上传文档入库，提问时先检索再生成。
```

改动：空库提问 500。

```
fix: 知识库为空时 /ask 直接返回提示，不再调 embedding。
```
