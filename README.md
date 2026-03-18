# scisci_agent_disagree
This repository investigates how scientific disagreement patterns could be simulated by multi-agents collaboration

## 1) Environment Setup (Python 3.10 required)

> Python 3.10 works best. Other versions may cause dependency conflicts.
> 

### Create and activate conda environment

```powershell
conda create--name sciagent python=3.10-y
&"C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1"
conda activate sciagent

```

### Install baseline scientific stack first (prevents numpy ABI issues)

```powershell
conda install-c conda-forge numpy scipy-y

```

### Install Python dependencies

```powershell
python-m pip install-r requirements.txt

```

> Note: In the repo, yaml should be replaced with PyYAML (package name is PyYAML).
> 

---

## 2) AgentScope Version Pin (critical)

CiteAgent expects an **older AgentScope format**.

Use **agentscope==0.0.4**.

```powershell
pip install agentscope==0.0.4

```

### (Optional) Installing the latest AgentScope from source

Not recommended for this project because format changes can break the pipeline.

```powershell
git clone-b main https://github.com/agentscope-ai/agentscope.git
cd agentscope
pip install-e .

```

---

## 3) Dataset Placement

You must download the dataset files and place them under the corresponding config directory.

Example (citeseer):

```
LLMGraph/tasks/citeseer/configs/<your_config>/data/

```

Where generated outputs will go (from config):

```
generated_article_dir: data/generated_article

```

So you should expect outputs like:

```
LLMGraph/tasks/citeseer/configs/<your_config>/data/generated_article/paper_citeseer_XXXX.txt

```

---

## 4) Fixing Dependency Conflicts (common on Windows)

The original `requirements.txt` did not pin versions, causing conflicts.

If you hit runtime errors / ABI mismatch (common with numpy), run:

```powershell
pip uninstall numpy pandas scikit-learn sentence-transformers-y

pip install numpy
pip install pandas scikit-learn sentence-transformers

pip install langchain-huggingface

```

---

## 5) Run: Build Citation Graphs

Supported tasks include:

- `cora`
- `citeseer`
- `llm_agent`

Example: build + save for `citeseer`

```powershell
python main.py--task citeseer--config template_fast_gpt4-mini--build--save

```

This runs:

- `-build` → simulation/build process
- `-save` → writes saved artifacts (graph / metadata depending on Executor implementation)

---

## 6) Notes on Logs & Warnings

### A) “Relevance scores must be between 0 and 1”

You may see warnings like:

```
UserWarning: Relevance scores must be between 0 and 1, got [...]

```

This happens because the retriever returns similarity scores that aren’t normalized to `[0,1]`.

It usually **does not stop execution**.

### B) Terminal “agent discussions”

The long dialogue logs are **agent communication traces** during the Socialization / Creation stages.

They may appear even if the saved paper text files look short—because the `.txt` files often store a compact “paper content” field (title + abstract-like draft), while citation edges/metadata live in saved graph/meta artifacts.

---

## 7) Local Code Changes Made

- Fixed missing topic lookups
- Addressed multiple `self` attribute issues:
    - Observed keys during runtime:
        
        ```
        dict_keys([... 'article_write_configs','different_tag_author','different_tag_rate'])
        
        ```
        
- Added initialization of missing `self` attributes to prevent runtime errors
- Regenerated a working `requirements.txt` targeting **Python 3.10**

---

## 8) Outputs

### Generated draft texts

Path (example):

```
LLMGraph/tasks/citeseer/configs/template_fast_gpt4-mini/data/generated_article/

```

Files:

- `paper_citeseer_####.txt`

These may be short (abstract-like). That’s expected unless prompts/templates are modified to produce full multi-section papers.

---

## 9) Troubleshooting

### If nothing is saved

Make sure you ran with `--save`:

```powershell
python main.py--task citeseer--config template_fast_gpt4-mini--build--save

```

### If generated_article_dir exists but no papers appear

- Confirm dataset files are placed under the config `data/` directory
- Verify your config points to:
    - `article_dir`
    - `generated_article_dir`
- Check whether `Executor.save()` writes paper drafts vs only graphs/metadata

---

## 10) Recommended Next Improvements (optional)

- Normalize relevance scores to `[0,1]` to remove retriever warnings
- Modify the paper-writing prompt/template to output full structured papers (Intro/Related Work/Method/Experiments/Conclusion)
