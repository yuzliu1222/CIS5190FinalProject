# Project B 模型改进说明（my_model/）

原始 baseline 在 `../tested baseline/` 中，**一行未动**。`my_model/` 里的三个文件是它的副本：被替换的代码保留为注释（标 `ORIGINAL`），新代码写在下面（标 `NEW`）。

## 怎么运行

```powershell
cd "final_project/Proj B Template/my_model"
python train_baseline.py
cd ..
python eval_project_b.py --model my_model/model.py --preprocess my_model/preprocess.py --weights my_model/model.pt --csv url_train.csv
```

训练约 2–3 分钟（CPU）。输出：
- `my_model/model.pt`：提交用的权重
- `my_model/results/`：`metrics.csv`、`lr_search.csv`、`learning_curves.png`、`model_comparison.png`、`final_model_summary.txt`

提交排行榜时上传 `my_model/model.py`、`my_model/preprocess.py`、`my_model/model.pt` 这三个文件。

## 结果（url_train.csv 内部划分，80/10/10）

| 模型 | 验证集 | 测试集 | NBC 政治测试 | Fox 购物测试 |
|---|---|---|---|---|
| 多数类 | 50.4% | 50.4% | 0% | 100% |
| Baseline NN（原代码） | 95.4% | 95.8% | 1.7% | 4.9% |
| LR + AdamW（对比实验，未采用） | 99.4% | 99.6% | 15.4% | 0.4% |
| Logistic regression（SGD） | 96.9% | 97.7% | 100% | 100% |
| 剪枝决策树 | 96.9% | 97.7% | 100% | 100% |
| **LR + 决策树（最终）** | **96.9%** | **97.7%** | **100%** | **100%** |

- 两个"测试"是用测试集链接改造的**人造链接**，只检查模型有没有用上链接格式，**不能**当作排行榜分数的估计。
- 最终模型在测试集上的混淆矩阵（NBC 为正类）：TP=227，FN=11，FP=0，TN=242；NBC 的 precision 是 1.00，recall 是 0.954。漏掉的 11 条是没有 ID 的 NBC 栏目页（如 `/politics/cory-booker`）。
- 选出的超参数：LR 用 SGD，lr=1.0，weight_decay=0.01，训练 9 个 epoch；决策树 ccp_alpha≈0.00094（5 个节点，深度 2）。
- 用 `eval_project_b.py` 跑通：4,800 条用时 0.43 秒，接口正常。

## 修改说明：改了什么 → 为什么 → 对模型的影响

### preprocess.py
1. **新增 `url_to_text`：标题词 + 两个格式 token**
   - 改了什么：取路径最后一段（标题），转小写，去掉 NBC 式 ID 和所有数字，只保留字母。再加上 `segcountK`（路径段数）和 `nbcidtoken`（以 `-字母+6位以上数字` 结尾）。
   - 为什么：baseline 只看标题词，学到的是主题（购物→NBC，政治→Fox）。url_train.csv 里 94% 的 NBC 是购物文章，所以 NBC 政治新闻会被判成 Fox，这是排行榜只有 0.64 的主要原因。格式特征和主题无关。
   - 影响：NBC 政治测试从 1.7% 提高到 100%。
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
   - 为什么不用 AdamW：Adam 会给每个权重单独调整步长，少见的标题词也能拿到很大的权重。实测 AdamW 的验证集准确率 99.4%，但两个测试只有 15% 和 0.4%。换成 SGD 后，权重集中在格式 token 上（`segcount3` +2.06，`nbcidtoken` +1.62）。
   - AdamW 保留为对比实验（`optimizer_name="adamw"`），结果写在 `metrics.csv` 里。
10. **稳健的选参规则**：选参分数 = (验证集准确率 + 验证集 NBC 政治测试 + 验证集 Fox 购物测试) / 3。
    - 为什么：验证集和训练集一样偏，只看验证集准确率会选出依赖主题词的模型。
    - 两个测试**在验证集和测试集上各造一份**：验证集那份用来选参，测试集那份只用来汇报，所以测试集不会被污染。
11. **决策树**：用 5 折交叉验证选 `ccp_alpha`，写法和你 hw2 第 72 个 cell 一样。准确率相同时选更大的 alpha，也就是更简单的树。
12. **评估**：accuracy、混淆矩阵、两类各自的 precision/recall，并和多数类基线、原 baseline 对比。
13. **画图**：一张学习曲线（baseline vs LR），一张包含 baseline 在内的模型对比折线图。
14. **最终模型**：用选好的超参数在全部 4,800 条上重训，再保存。
15. **修 bug**：
    - 原来写死 `torch.save(..., "model.pt")`，`--output` 不起作用。
    - 默认 `--csv url_train16k.csv` 这个文件不存在，改成 `../url_train.csv`。

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
| 训练循环（zero_grad / backward / step） | `train_logistic_regression` | L10 第 30 页；hw3_skeleton.ipynb 第 89 个 cell |
| Mini-batch SGD | `train_logistic_regression`、`make_optimizer` | L10 第 36 页 |
| Adam / AdamW（对比实验，说明为什么不用） | `make_optimizer` 的 `"adamw"` 分支 | L10 第 42 页 |
| L2 正则 = weight decay；bias 不正则 | `make_optimizer` 的参数分组 | L11_Neural Networks (ctd)+K-Nearest Neighbors and Decision Trees.pdf 第 17 页；L06_Linear Regression (contd.).pdf 第 21 页；hw2_skeleton.ipynb 第 50 个 cell |
| L2 正则 = 高斯先验（解释用） | — | L07 第 38–39 页 |
| Early stopping（保留验证 loss 最低的 checkpoint） | `train_logistic_regression` 第 4–5 步 | L11 第 24 页 |
| 超参数网格、log 尺度、相同随机初始化 | `main` 第 4 步 | L04_Linear Regression (contd.).pdf 第 16 页；L11 第 28–29 页 |
| Train/val/test 划分（80/10/10） | `main` 第 2 步 | L04 第 18–20 页 |
| 验证集要像真正的考试 → 稳健选参规则 | `main` 第 4 步 selection score；`make_probe_sets` | L04 第 18 页 |
| 测试集不参与任何选择（validation hygiene） | 验证集/测试集各造一份测试 | L04 第 21 页 |
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
