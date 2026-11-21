# Project B 模型改进说明（my_model/）

原始 baseline 在 `../tested baseline/` 中，**一行未动**。`my_model/` 里的三个文件是它的副本：被替换的 baseline 代码保留为注释（标 `ORIGINAL`），新代码写在下面（标 `NEW`）。

第三版把 LR 的训练部分改写成了 hw3 的写法（见第 16 条）。改写前的训练脚本存在 `old_versions/train_baseline_v2.py`，可以拿来对比。

## 怎么运行

在 `my_model` 文件夹里运行（`url_train.csv` 和 `eval_project_b.py` 放在它的上一层）：

```powershell
cd my_model
python train_baseline.py
cd ..
python eval_project_b.py --model my_model/model.py --preprocess my_model/preprocess.py --weights my_model/model.pt --csv url_train.csv
```

训练约 3 分钟（CPU）。输出：
- `my_model/model.pt`：提交用的权重
- `my_model/results/`：`metrics.csv`、`lr_search.csv`、`learning_curves.png`、`model_comparison.png`、`final_model_summary.txt`，以及 early stopping 存下的 `best_lr_checkpoint.pt`（训练中间文件，不用提交）

提交排行榜时上传 `my_model/model.py`、`my_model/preprocess.py`、`my_model/model.pt` 这三个文件。

## 结果（url_train.csv 内部划分，80/10/10）

| 模型 | 验证集 | 测试集 |
|---|---|---|
| 多数类 | 50.4% | 50.4% |
| Baseline NN（原代码） | 95.4% | 95.8% |
| LR + AdamW（对比实验，未采用） | 99.2% | 99.6% |
| Logistic regression（SGD） | 97.3% | 97.9% |
| 剪枝决策树 | 96.9% | 97.7% |
| **LR + 决策树（最终）** | **96.9%** | **97.7%** |

- 最终模型在测试集上的混淆矩阵（NBC 为正类）：TP=227，FN=11，FP=0，TN=242；NBC 的 precision 是 1.00，recall 是 0.954。漏掉的 11 条是没有 ID 的 NBC 栏目页（如 `/politics/cory-booker`）。
- 选出的超参数（按验证集准确率）：LR 用 SGD，lr=1.0，weight_decay=0，训练 30 个 epoch；决策树 ccp_alpha≈0.00094（5 个节点，深度 2）。
- 用 `eval_project_b.py` 跑通：4,800 条用时约 0.4 秒，接口正常。
- **注意**：这里的验证集和测试集都来自 `url_train.csv`，和训练集一样，几乎没有 NBC 新闻链接。所以这些分数**不能**说明模型在排行榜上会怎样，也解释不了为什么 AdamW 分数更高却没被采用（见第 9 条）。真正的检验是排行榜分数，或者收集真实的 NBC 新闻链接。

### 开发过程中做过的检查（代码里已删除）
开发时我们曾造过两组"人造链接"做检查：把 Fox 的政治标题改成 NBC 的链接格式，再把 NBC 的购物标题改成 Fox 的链接格式，看模型能不能靠链接格式判断。按组员的决定，这部分代码已经从第五版删除，只在这里记录当时的结论：
- Baseline NN 在这两组上几乎全错（1.7% / 4.9%），说明它只看主题词。
- AdamW 训练的 LR 也几乎全错（14.5% / 0.4%），这就是改用 SGD 的原因。
- SGD 训练的 LR 单独用时，第二组只有 53.8%；剪枝决策树两组都是 100%；两者合起来以后是 100% / 98.2%。

## 修改说明：改了什么 → 为什么 → 对模型的影响

### preprocess.py
1. **新增 `url_to_text`：标题词 + 两个格式 token**
   - 改了什么：取路径最后一段（标题），转小写，去掉 NBC 式 ID 和所有数字，只保留字母。再加上 `segcountK`（路径段数）和 `nbcidtoken`（以 `-字母+6位以上数字` 结尾）。
   - 为什么：baseline 只看标题词，学到的是主题（购物→NBC，政治→Fox）。url_train.csv 里 94% 的 NBC 是购物文章，所以 NBC 政治新闻会被判成 Fox，这是排行榜只有 0.64 的主要原因。格式特征和主题无关。
   - 影响：开发时的人造链接检查里，模型对"NBC 格式的政治链接"的判断从 1.7% 正确提高到 100%（见上面"开发过程中做过的检查"）。
2. **不再跳过任何一行**（注释掉了原来的 `return`）
   - 为什么：接口规定必须每行返回一个输入、顺序不变。没有标签的行改在训练脚本里过滤。
   - 影响：更符合接口合约；排行榜数据本来就没有网站名，所以结果不变。
3. **读文件时加 `encoding="utf-8"`**
   - 为什么：Windows 默认编码是 GBK，加上后本地和后端读法一致。

### model.py
4. **新增 `texts_to_bucket_ids` / `bucket_ids_to_counts` / `texts_to_counts`**
   - 改了什么：沿用 baseline 的分词和 md5 哈希，生成词袋计数向量。
   - 为什么：两个子模型共用同一份特征；用哈希就不用保存词表，权重能全部存进一个 `model.pt`。
5. **新的 `NewsClassifier` = logistic regression + 剪枝决策树，概率取平均**（原来的 `NewsClassifier` 已注释掉）
   - LR 是一个 `nn.Linear(50000, 2)`。决策树的节点数组用 `register_buffer` 存成张量。`predict` 用一个 `for` 循环沿着树往下走，把两个模型的概率取平均后再 argmax。
   - 为什么：两个模型都很好解释，而且 ensemble 是项目说明里明确允许的写法。
6. **`predict` 直接返回 `"FoxNews"` / `"NBC"` 字符串**
   - 为什么：原来返回 0/1，要靠评测器去猜哪个数字对应哪个网站。
7. **原 `HashedTextClassifier` 保留不动**，训练脚本用它在同一划分上重训 baseline，作为对比。去掉了 3 个没用到的 sklearn import（已注释）。

### train_baseline.py
8. **数据按 80/10/10 分层划分**：train 用来学参数，val 用来选超参数，test 只在最后看一次。
9. **LR 训练改用 mini-batch SGD + L2 weight decay（bias 不加）+ early stopping**
   - 为什么不用 AdamW：Adam 会给每个权重单独调整步长，少见的标题词也能拿到很大的权重。AdamW 的验证集准确率（99.2%）其实比 SGD 高，但开发时的人造链接检查显示它几乎只看主题词（14.5% / 0.4%）。现在的验证集里几乎没有 NBC 新闻链接，看不出这个问题，所以这一条的依据只记录在上面的"开发过程中做过的检查"里。换成 SGD 后，偏向 NBC 的最大权重是格式 token（最终模型里 `segcount3` +4.19，`nbcidtoken` +3.04），不过 "after"、"trump"、"says" 这类主题词也有不小的权重（约 -2）。
   - AdamW 保留为对比实验（`optimizer_name="adamw"`），结果写在 `metrics.csv` 里。
10. **选参规则：选验证集准确率最高的那组**（第四版，和 hw2 第 72 个 cell 的写法一样：`if val_acc > max_score`）
    - 出处：L04 第 20 页的 holdout 流程。
    - 第三版曾用"验证集准确率和两个人造测试取平均"来选参，但这个公式课上没讲过，所以第四版改回标准做法；第五版把人造测试的代码也删掉了。
    - 以后收集到真实的 NBC 新闻链接、放进验证集以后，这个规则选出来的模型自然就会更稳。
11. **决策树**：用 5 折交叉验证选 `ccp_alpha`，写法和你 hw2 第 72 个 cell 一样。准确率相同时选更大的 alpha，也就是更简单的树。
12. **评估**：accuracy、混淆矩阵、两类各自的 precision/recall，并和多数类基线、原 baseline 对比。
13. **画图**：一张学习曲线（baseline vs LR），一张包含 baseline 在内的模型对比折线图。
14. **最终模型**：用选好的超参数在全部 4,800 条上重训，再保存。
15. **修 bug**：
    - 原来写死 `torch.save(..., "model.pt")`，`--output` 不起作用。
    - 默认 `--csv url_train16k.csv` 这个文件不存在，改成 `../url_train.csv`。
16. **第三版：把 LR 的训练部分改成 hw3 的写法**（改写前的版本在 `old_versions/train_baseline_v2.py`）
    - 新增 `UrlCountDataset`（和 hw3 第 75 个 cell 的 `STKDataset` 同一个模式），用 `DataLoader` 分批：训练集打乱，验证集和测试集不打乱（hw3 第 82 个 cell）。
    - 新增 `train_one_epoch` 和 `evaluate`，几乎逐行照搬 hw3 第 89 个 cell；损失函数改成 `criterion = nn.CrossEntropyLoss()`（hw3 第 88 个 cell）。
    - Early stopping 的最好 checkpoint 改用 `torch.save(model.state_dict(), ...)` 存文件，结束时再 `load_state_dict` 读回来，不再用 `.clone()` / `.copy_()`。
    - `predict_probs` 照 hw3 第 92 个 cell 的写法：用 list 收集结果，再转成 `np.array`，不再用 `torch.cat`。
    - 结果表改用 for 循环打印，不再用 `df.pivot`；嵌套的 `def take` 移到外面，改名为 `take_items`。
    - 影响（当时还用第三版的选参规则）：选出的超参数没变（lr=1.0，weight_decay=0.01）。第四版改了选参规则以后的结果见第 10 条和上面的结果表。early stopping 停在第 10 个 epoch（原来第 9 个），因为 `DataLoader` 打乱数据的随机顺序和原来不同。验证集和测试集的准确率和第二版完全相同。
17. **第五版：删除人造链接检查的全部代码**（`make_probe_sets` 函数，以及网格搜索、结果表和对比图里的相关部分）。选参规则和模型都没变，结果表现在只有验证集和测试集两列。

## 出处汇总表

| 概念/公式 | 用在代码哪里 | 课上文件及位置 |
|---|---|---|
| 手工构造的特征映射 φ(x)（文本） | `preprocess.py` `url_to_text` 的格式 token | L02_Linear_regression.pdf 第 58–63 页；L20_Natural Language Processing Part 2.pdf 第 3 页 |
| 词袋（bag-of-words）特征 | `model.py` `bucket_ids_to_counts` | L20 第 5–6 页 |
| BoW 前的预处理（小写、去符号） | `preprocess.py` `url_to_text` 第 2–4 步 | L20 第 7 页；Project_Intro 第 16 页 4.3.1 节 |
| 哈希分桶（不存词表） | `model.py` `_hash_token`、`texts_to_bucket_ids` | tested baseline/model.py `_hash_token`（课程提供的代码） |
| Logistic 模型 / logits | `model.py` `NewsClassifier.linear` | L07_Logistic Regression.pdf 第 13–14 页 |
| 二分类交叉熵 | `train_baseline.py` `F.cross_entropy` | L07 第 22 页 |
| Softmax + 交叉熵（多类形式） | `model.py` `lr_probs`；训练的损失 | L08_Logistic Regression+Logistic Regression.pdf 第 7 页 |
| PyTorch 网络返回 logits；`eval()` + softmax 预测 | `model.py` `forward`、`predict` | L10_Neural Networks (ctd).pdf 第 29–31 页 |
| 训练循环（zero_grad / backward / step）、`train_one_epoch` / `evaluate` | `train_one_epoch`、`evaluate` | L10 第 30 页；hw3_skeleton.ipynb 第 89 个 cell |
| `nn.CrossEntropyLoss` 作为 criterion | `train_logistic_regression` 第 2 步 | hw3_skeleton.ipynb 第 88 个 cell |
| `Dataset` 类 / `DataLoader` 分批 | `UrlCountDataset`；`train_logistic_regression` 第 1 步 | hw3_skeleton.ipynb 第 70、75、82 个 cell |
| 预测时 `model.eval()` + `no_grad`，用 list 收集结果 | `predict_probs` | hw3_skeleton.ipynb 第 92 个 cell；L10 第 31 页 |
| Mini-batch SGD | `train_logistic_regression`、`make_optimizer` | L10 第 36 页 |
| Adam / AdamW（对比实验，说明为什么不用） | `make_optimizer` 的 `"adamw"` 分支 | L10 第 42 页 |
| L2 正则 = weight decay；bias 不正则 | `make_optimizer` 的参数分组 | L11_Neural Networks (ctd)+K-Nearest Neighbors and Decision Trees.pdf 第 17 页；L06_Linear Regression (contd.).pdf 第 21 页；hw2_skeleton.ipynb 第 50 个 cell |
| L2 正则 = 高斯先验（解释用） | — | L07 第 38–39 页 |
| Early stopping（保留验证 loss 最低的 checkpoint） | `train_logistic_regression` 第 4–5 步（`torch.save` / `load_state_dict`） | L11 第 24 页 |
| 超参数网格、log 尺度、相同随机初始化 | `main` 第 4 步 | L04_Linear Regression (contd.).pdf 第 16 页；L11 第 28–29 页 |
| Train/val/test 划分（80/10/10） | `main` 第 2 步 | L04 第 18–20 页 |
| 选验证集表现最好的超参数（holdout） | `main` 第 4 步 `if val_acc > max_score` | L04 第 20 页；hw2_skeleton.ipynb 第 72 个 cell |
| 测试集不参与任何选择（validation hygiene） | 测试集只在第 6 步看一次 | L04 第 21 页 |
| k 折交叉验证；选好后在全部数据上重训 | `choose_ccp_alpha`；`main` 第 8 步 | L04 第 22–23 页 |
| 决策树：选特征让子节点更纯 | sklearn `DecisionTreeClassifier` | L13_K-Nearest Neighbors and Decision Trees (ctd.).pdf 第 41–47 页 |
| Gini 指数（sklearn 默认的分裂标准） | 同上 | L13 第 67 页；L14 第 14 页；hw2_skeleton.ipynb 第 57 个 cell |
| 剪枝 / 复杂度惩罚，用交叉验证选树的超参数 | `choose_ccp_alpha` | L13 第 79–81 页；L14 第 26–28 页；hw2_skeleton.ipynb 第 71–72 个 cell |
| 和简单基线比较（多数类） | `main` 第 3 步 | L04 第 29 页；L08 第 26 页 |
| 混淆矩阵、accuracy | `compute_metrics` | L08 第 12–14 页 |
| Precision / Recall | `compute_metrics` | L08 第 23–24 页 |
| 学习曲线画法 | `main` 第 7a 步 | hw3_skeleton.ipynb 第 90 个 cell |
| 包含 baseline 的折线图 | `main` 第 7b 步 | Project_Intro 第 15 页 4.1 节 |
| Ensemble 放在一个 Model 类里、只用一个 model.pt | `NewsClassifier.ensemble_probs` | Project_Intro 第 10 页 2.7 节、第 18 页 FAQ |
| 每行一个输入、顺序不变 | `preprocess.py` `_process` | Project_Intro 第 7 页 2.3 节 |

页码都是 PDF 的页码（也就是阅读器里显示的页数），不是幻灯片角落印的编号。
