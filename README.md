# dl-roadmap

深度学习学习路线：PyTorch / Transformer / LLM 应用

## 学习日志

### 2026-10-28 · Day 1：环境搭建

- 安装 Anaconda（E:\Anaconda），配置 conda / pip 清华镜像源
- 创建独立环境 `dl`（Python 3.11），与系统环境隔离
- 收获：理解虚拟环境的意义——不同项目依赖互不污染

### 2026-10-29 · Day 2：安装 PyTorch

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

### 2026-11-01 · Day 5：Python 工程化自测

- 手写计时装饰器 `timer_decorator.py`：`@functools.wraps` 保留函数元信息，`*args, **kwargs` 透传任意参数
- 手写大文件逐行读取生成器 `file_generator.py`：10 万行文件用 `yield` 惰性读取
- 对比实验：生成器对象约 200 字节，等量列表约 800MB——惰性求值的内存优势
- 自测结论：
  - 不加 `@functools.wraps` 功能不受影响，但 `__name__` 变成 `wrapper`，调试/日志/框架集成会踩坑
  - `return` 结束函数一次性返回；`yield` 暂停函数分批产出，且冻结局部变量状态
  - 列表即时全量存储，生成器惰性按需生成，一次只算一个
- 收获：装饰器本质是 `train = timer(train)`；生成器是懒执行和状态保留

### 2026-11-02 · Day 6：asyncio 并发编程

- 手写 `async_demo.py`，跑通 `asyncio.gather` 并发：4 个任务总耗时 ≈ 最长任务（3s），而非任务之和（6s）
- 对照实验：
  - 顺序 `await` 三个任务 → 总耗时 ≈ 6s：等待不重叠
  - 不加 `await` 直接调用协程 → 只创建协程对象不执行，报 `RuntimeWarning: coroutine was never awaited`
  - gather 混入 CPU 密集任务（5000 万次加法）→ **事件循环被独占**：A 的定时器 1s 就到期，却被卡到 CPU 任务算完才被处理
- 关键认知：任务耗时 = 纯计算 + 等待；asyncio 只重叠"等待时间"，sleep 期间任务不占 CPU、不存在于运行中，只是登记了一条"到期后跑收尾代码"的记录
- 收获：定时器到期 ≠ 被叫醒——CPU 密集代码会阻塞事件循环，拖垮所有 IO 任务的响应延迟，解法是进程池隔离

### 2026-11-07 · W2 Day 1：激活函数 + LeetCode 开刷

- 手写 `activations.py`，绘制 ReLU / sigmoid / tanh 的函数曲线与导数曲线
- 去掉激活函数 → 多层线性变换可合并为一层仿射变换，深度失效
- ReLU 正区导数恒为 1，梯度无衰减穿透；sigmoid 导数最大仅 0.25，10 层叠加梯度衰减百万倍  
  |  |  |  |  
  | --- | --- | --- |
- sigmoid 在 |x|>6 时导数 < 0.002 → 梯度消失；对称知识点：ReLU 负区恒 0 → 死亡 ReLU
- LeetCode 哈希专题起步：「两数之和」（哈希一遍遍历 O(n)，边走边查互补数）+「字母异位词分组」
- 收获：三种激活函数各有各的"消失方式"——sigmoid/tanh 败在饱和区，ReLU 败在负区

### 2026-11-09 · W2 Day 2：MLP 从零实现

- 手写实现 MLP（784→256→10）：手动初始化参数、手写 relu 前向传播、手写训练循环，Fashion-MNIST 5 epoch 测试准确率 **0.813**（0.743→0.771→0.789→0.819→0.813）
- 权重初始化乘 0.01 的原因：全零初始化会让所有神经元输出相同、学到相同的东西
- 踩坑与解决：
  - Windows 下 DataLoader 多进程 spawn 报错 → 子进程会重新 import 主脚本，执行代码必须包进 `if __name__ == "__main__"`
  - d2l 新版没有 `evaluate_accuracy` → 手写 accuracy 函数（argmax 取预测类别，统计正确数）
- 观察到 epoch 5 准确率微降（0.819→0.813）：过拟合的前兆，Week 后续用权重衰减/dropout 处理
- 收获：从零实现走通了"参数是 nn.Parameter 的张量、前向是矩阵乘法、训练是三行循环"的全过程——框架只是把这些封装了起来

### 2026-11-11 · W2 Day 3：简洁实现

- 用 `nn.Sequential` 重写 MLP（Flatten + Linear + ReLU + Linear），SGD lr=0.1 跑 10 epoch，测试准确率 **0.849**（从零版 0.813 / 5 epoch）
- 观察到 0.842→0.829→0.845 的锯齿波动
- `nn.Linear` 封装了参数创建与默认初始化；`CrossEntropyLoss` 内置 softmax 故网络末层不加；`net.parameters()` 把全部参数交给优化器统一更新
- 收获：MLP 在 Fashion-MNIST 上的天花板约 0.88~0.89，纯加 epoch 收益递减——想再涨要靠更好的结构（CNN）而不是更久的训练

### 2026-11-16 · W2 Day 4：过拟合实验 + 卷积入门

- 隐藏层加大到 512 + 训练数据砍到 6000 条，10 epoch 后 train 0.809 / test 0.789——test 比全量数据的 0.849 掉了 6%，train 持续上升而 test 在 epoch 5 后停滞，分叉前奏已现；权重衰减与 dropout 对照实验待补
- 过拟合三问写进 notes.md：过拟合 = 训练好、新数据差（欠拟合则两者都差）；L2 惩罚迫使权重变小、函数更平滑；dropout 训练丢神经元等效训练子网络，推理要确定性输出故不丢
- `conv_basics.py` 手写双重循环卷积与 `nn.functional.conv2d` 逐元素对比，`allclose=True` 验证通过——卷积就是"滑动窗口内乘加"
- LeetCode 累计第 3 题「赎金信」：`Counter(ransomNote) - Counter(magazine)` 为空即满足，先用长度剪枝
- 收获：数据量从 60000 砍到 6000，test acc 直降 6%——泛化能力是买不来的，正则化和更多数据才是解药

### 2026-11-18 · W2 Day 5：优化器进化史

- `optimizers_exp.py`：同一个 MLP（784→256→10）分别用 SGD(lr=0.1)、动量法(lr=0.1, momentum=0.9)、Adam(lr=0.001) 训练 10 epoch，记录逐 batch loss 并画曲线对比
- 曲线观察：三者起点都是 2.3（=ln10，瞎猜水平）；动量法下降最快且曲线最平滑，Adam 紧随其后，SGD 全程噪声最大、末期 loss 略高（最终 ≈0.3）
- 认知三连写进 notes.md：Adam 前期掉 loss 最快（自适应学习率起步猛）；动量=惯性，冲过 loss 曲面小沟壑；Adam 的 lr 比 SGD 小两个数量级，因为按梯度二阶矩自适应缩放步长
- 记录逐 batch loss（而非逐 epoch），曲线细节更丰富；每组实验重新 `make_net()` 保证初始化公平
- 收获：优化器没有绝对优劣——Adam 是默认选项，但理解 SGD/动量才能讲清 Adam 为什么好用：它只是把调 lr这件事自动化了

### 2026-11-19 · W2 Day 6：双指针专题日

- LeetCode 进入双指针专题，新建 `leetcode/two_pointers/`
- 283\. 移动零：快慢指针一遍扫描——左指针指向已处理序列尾部，右指针遇到非零数就和左指针交换并右移；原地操作 O(n) 时间 O(1) 空间
- 15\. 三数之和：排序 + 固定首元素 + 双指针夹逼
- 收获：双指针的本质是"用有序性消灭一层循环"——暴力 O(n²) 靠排序降为 O(n)，两数之和、三数之和都是这个思想的变奏

### 2026-11-20 · W2 Day 7：周复盘 

**MLP 与非线性**：去掉所有激活函数，无论多少层都等价于单层线性模型，深度失去意义

**过拟合判定**：训练误差极低而验证误差先降后升、两者差距拉大；K 折交叉验证下训练集显著优于验证集也是信号。对策：L2 正则化 + Dropout

**优化器演进**：固定步长只看当前梯度 → 引入历史梯度平滑加速（动量）→ 自适应步长 + 动量双保险（Adam），每一代都在解决前代的痛点
