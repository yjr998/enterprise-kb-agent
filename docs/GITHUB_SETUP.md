# GitHub 上传指南

见根目录 `GITHUB_SETUP.md` 的副本；完整步骤如下。

## 最快启动（PowerShell）

```powershell
cd d:\Rag_agent\enterprise-kb-agent
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ollama pull qwen2.5:3b
python scripts/build_vectordb.py
python run.py
```

**不要用** `start.bat`  alone in PowerShell；优先 `python run.py`。

## Push

```powershell
git init
git add .
git status   # 确认无 .env
git commit -m "feat: enterprise KB agent"
git remote add origin https://github.com/你的用户名/enterprise-kb-agent.git
git push -u origin main
```

详细说明见：`d:\Rag_agent\GITHUB_SETUP.md`
