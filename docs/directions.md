# 方向雷达（Direction Radar）使用指南

AI Infra Radar 支持按**研究方向**切换雷达。每个方向拥有独立的配置文件、数据库和 digest 目录，互不干扰。tag 设计为**研究子方向**（而非大方向名），用于洞察方向内部各主题的热度与进展。

## 文件清单

| 方向 | 配置文件 | 数据库 | digest 目录 | 子方向数 |
|------|----------|--------|------------|---------|
| AI Infra（默认） | `config.example.yaml` | `data/radar.db` | `reports/` | 7 |
| VLA | `config.vla.yaml` | `data/radar-vla.sqlite` | `reports/vla/` | 7 |
| VLM | `config.vlm.yaml` | `data/radar-vlm.sqlite` | `reports/vlm/` | 9 |
| WM | `config.wm.yaml` | `data/radar-wm.sqlite` | `reports/wm/` | 8 |

## 一键切换方向（Makefile 方式，推荐）

Makefile 已内置 `DIR` 变量支持。设了 `DIR` 就切换到对应方向，不设则走默认的 AI Infra，**完全向后兼容**。

```bash
cd /Users/didi/Documents/project/paper-radar
source .venv/bin/activate

# 完整流水线 + 启动仪表盘，一行切换方向：
make pipeline DIR=vlm      # vla / vlm / wm
make dashboard DIR=vlm
```

`make pipeline DIR=vlm` 等价于自动执行：

```bash
python -m radar.cli ingest-arxiv  --config config.vlm.yaml --db data/radar-vlm.sqlite
python -m radar.cli ingest-github --config config.vlm.yaml --db data/radar-vlm.sqlite
python -m radar.cli tag-papers    --config config.vlm.yaml --db data/radar-vlm.sqlite
python -m radar.cli match-repos   --db data/radar-vlm.sqlite
python -m radar.cli score         --config config.vlm.yaml --db data/radar-vlm.sqlite
python -m radar.cli digest        --config config.vlm.yaml --db data/radar-vlm.sqlite --date today
```

`make dashboard DIR=vlm` 会自动设 `RADAR_DB_PATH=data/radar-vlm.sqlite` 启动 Streamlit。
启动后，侧边栏 "Topic tag" 下拉框会自动列出该方向的所有子方向，可按子方向筛选。
切换方向只需改 `DIR` 的值并重跑。

| 命令 | 说明 |
|------|------|
| `make pipeline` | 默认 AI Infra 方向（`config.example.yaml`） |
| `make pipeline DIR=vlm` | VLM 方向 |
| `make pipeline DIR=wm` | WM 方向 |
| `make dashboard DIR=vlm` | 启动 VLM 方向仪表盘 |
| `make digest DIR=wm` | 仅刷新 WM 方向的 digest |

> 不设 `DIR` 时行为与原版完全一致：`make pipeline` → `config.example.yaml` + `data/radar.db`。

## 仅刷新 digest（不重新采集）

```bash
make digest DIR=vlm
```

digest 会写入该方向专属目录（如 `reports/vlm/2026-08-03.md`），不会覆盖其他方向。

## 手动方式（不用 Makefile）

如果不想用 Makefile，也可以直接用 `DIR` 变量手动执行（与 Makefile 内部逻辑一致）：

```bash
DIR=vla   # vla / vlm / wm

python -m radar.cli ingest-arxiv  --config config.$DIR.yaml --db data/radar-$DIR.sqlite
python -m radar.cli ingest-github --config config.$DIR.yaml --db data/radar-$DIR.sqlite
python -m radar.cli tag-papers    --config config.$DIR.yaml --db data/radar-$DIR.sqlite
python -m radar.cli match-repos   --db data/radar-$DIR.sqlite
python -m radar.cli score         --config config.$DIR.yaml --db data/radar-$DIR.sqlite
python -m radar.cli digest        --config config.$DIR.yaml --db data/radar-$DIR.sqlite --date today

RADAR_DB_PATH=data/radar-$DIR.sqlite streamlit run app/streamlit_app.py
```

## 子方向一览

子方向基于 arXiv 近期高发主题归纳，可在配置文件中自由增减。

### VLA（7 个子方向）

| 子方向 | 权重 | 关键词示例 |
|--------|------|-----------|
| action_generation | 1.3 | action generation, diffusion policy, policy learning, action chunking |
| reward_model | 1.2 | reward model, reward shaping, success detector, outcome reward |
| data_construction | 1.2 | data construction, demonstration, teleoperation, cross-embodiment |
| evaluation | 1.0 | benchmark, generalization, sim-to-real, OOD |
| pretraining | 1.1 | foundation model, representation learning, self-supervised |
| post_training | 1.2 | RLHF, DPO, reinforcement learning, alignment |
| world_model | 1.1 | world model, dynamics model, dreamer, latent dynamics |

### VLM（9 个子方向）

| 子方向 | 权重 | 关键词示例 |
|--------|------|-----------|
| hallucination | 1.3 | hallucination, faithfulness, object hallucination |
| safety_robustness | 1.2 | safety, adversarial, jailbreak, red-teaming |
| visual_token_efficiency | 1.2 | token pruning, visual compression, token merging |
| video_understanding | 1.2 | long video, streaming video, video question answering |
| visual_grounding | 1.1 | grounding, referring, salient object, segmentation |
| reasoning | 1.2 | chain-of-thought, compositional reasoning, visual reasoning |
| multimodal_agent | 1.3 | GUI agent, web agent, computer use, tool use |
| evaluation | 1.0 | benchmark, probing, leaderboard |
| pretraining_alignment | 1.1 | contrastive, alignment, fusion, prompt tuning |

### WM（8 个子方向）

| 子方向 | 权重 | 关键词示例 |
|--------|------|-----------|
| video_generation | 1.2 | video generation, text-to-video, long video generation |
| action_conditioned | 1.3 | action-conditioned, interactive, controllable generation |
| physical_dynamics | 1.3 | physical, physics, dynamics, physically-consistent |
| robot_learning | 1.2 | robot, manipulation, sim-to-real, autonomous driving |
| data_scaling | 1.1 | large-scale, dataset, data engine, synthetic data |
| representation_learning | 1.1 | latent, representation, variational, autoencoder |
| inference_efficiency | 1.1 | efficient, distillation, parallel decoding, quantization |
| evaluation | 1.0 | benchmark, protocol, metrics, leaderboard |

## 自动挖掘子方向（统计法 / LLM 法）

除了手动编写子方向，还可以从已采集的论文里**自动挖掘**子方向，再生成配置文件。

### 统计法（默认，无需 API key）

基于 TF-IDF 关键短语提取 + 共现聚类，纯本地运行，确定性：

```bash
make pipeline DIR=vlm              # 先采集某个方向的论文
python -m radar.cli discover-topics \
  --config config.vlm.yaml --db data/radar-vlm.sqlite \
  --directions 10 --out config.vlm-auto.yaml
```

### LLM 法（语义归纳，推荐）

统计法不懂数语义，容易混入 "improves"、"gains" 这类无主题意义的词。加 `--use-llm` 让 LLM 做语义分组，能挖出 "reward shaping"、"hallucination mitigation" 这样真正的研究主题：

```bash
export OPENAI_API_KEY="sk-..."                 # 或 LLM_API_KEY
# 可选：兼容服务（如 Azure、自建网关）
# export OPENAI_BASE_URL="https://your-gateway/v1"
# export OPENAI_MODEL="gpt-4o-mini"

python -m radar.cli discover-topics \
  --config config.vlm.yaml --db data/radar-vlm.sqlite \
  --use-llm --directions 10 --out config.vlm-llm.yaml
```

**工作机制**：先用统计法提取高频候选短语（压缩信号，避免塞全文），再把候选短语 + 论文样本交给 LLM 做语义分组。LLM 返回结构化 JSON（name / label / keywords），直接渲染成可用配置。

**无 key 自动回退**：`--use-llm` 但未设 API key 时，自动回退到统计法，不会报错。

### 然后用生成的配置跑雷达

```bash
python -m radar.cli tag-papers --config config.vlm-llm.yaml --db data/radar-vlm.sqlite
python -m radar.cli score      --config config.vlm-llm.yaml --db data/radar-vlm.sqlite
RADAR_DB_PATH=data/radar-vlm.sqlite streamlit run app/streamlit_app.py
```

> 生成的配置复用了原方向的 arXiv/GitHub 采集语句，所以**无需重新采集**，直接对已有数据库重新打标、评分即可。

## 自定义与扩展

**调整子方向关键词：** 直接编辑对应配置文件的 `topics.<子方向>.keywords` 列表。关键词出现在论文标题/摘要中即打上该 tag。

**调整子方向权重：** 修改 `topics.<子方向>.weight`。权重越高，该子方向在综合评分中影响力越大，仪表盘排名更靠前。

**新增子方向：** 在配置文件 `topics:` 下追加一个块即可，仪表盘会自动识别：

```yaml
topics:
  new_topic:
    weight: 1.2
    keywords:
      - 关键词1
      - 关键词2
    arxiv_queries: *vla_arxiv    # 复用大方向采集语句
    github_queries: *vla_github
```

**新增方向：** 复制任一配置文件，改 `database_path`、`reports_dir` 和采集语句即可。

## 设计要点

- **采集去重：** 子方向通过 YAML 锚点（`&vla_arxiv` / `*vla_arxiv`）共用大方向采集语句，论文只采集一次，但被各子方向 keywords 分别打标，实现多维度标注。
- **偏少子方向补强：** 某子方向命中过少时，可单独追加 `arxiv_queries`（不再用锚点引用），扩大该主题的采集范围。例如 VLM 的 `multimodal_agent` 额外加了 `cat:cs.HC` 查询。
- **digest 隔离：** 每个方向的 `reports_dir` 独立，避免多方向 digest 互相覆盖。
- **仪表盘自适应：** topic 下拉框从数据库动态读取，新增的子方向/方向无需改任何代码。
