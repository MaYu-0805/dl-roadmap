# dl-roadmap



## 学习日志

### 2026-10-28 · Day 1：环境搭建

- 安装 Anaconda（E:\Anaconda），配置 conda / pip 清华镜像源
- 创建独立环境 `dl`（Python 3.11），与系统环境隔离
- 收获：理解虚拟环境的意义——不同项目依赖互不污染

### 2026-10-29 · Day 2：安装 PyTorch

- `nvidia-smi` 确认硬件：RTX 3070（8GB），驱动支持 CUDA ≤ 13.2
- 安装 `torch 2.11.0+cu128`，验证 `torch.cuda.is_available() == True`
- 收获：`+cu128` 后缀说明装的是 GPU 轮子；镜像源没有 GPU 版，必须走官方 `--index-url`

### 2026-10-30 · Day 3：第一个训练 demo

- VS Code 配置 Python 解释器，跑通线性回归：`y = 3x + 2 + noise`
- 完整走通 **前向 → loss → 反向 → 更新** 的最小训练循环
- 动手实验与发现：
  - loss 卡在 ~1.115 不再下降 → **噪声地板**：噪声方差为 1，这是不可约误差
  - b 参数爬升极慢 → w、b 梯度**耦合**，收敛速度由损失曲面最平方向决定
  - lr=0.05 → loss 变 NaN → **学习率过大导致梯度爆炸**（数值溢出 inf-inf=NaN）
  - 缩小 x 范围到 [0,1] → w、b 螺旋震荡收敛 → **特征尺度影响收敛路径**

### 2026-10-31 · Day 4：Git + GitHub 工作流

- 安装配置 Git，创建远程仓库并完成首次 push
- 踩坑与解决：
  - 本地分支 `master` 与远程 `main` 不一致 → `git branch -m master main`
  - 合并无关联历史 → `--allow-unrelated-histories`
  - 误提交 `.vscode/` 配置 → 加入 `.gitignore` + `git rm -r --cached` 移除跟踪
- 收获：gitignore 只对未跟踪文件生效；GitHub 网页提交会使远程领先本地，先 pull 再干活
