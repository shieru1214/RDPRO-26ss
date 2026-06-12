"""Build the direct-run Colab notebook for the vision benchmark catalog."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
# Jiaozi: GPT-5.5 生成并训练 14 个视觉任务

这个 notebook 会删除 Colab 中旧的 `/content/Jiaozi`，拉取最新的
`codex/integration-update-colab` 分支，然后完成：

1. 选择一个比赛或公开数据集。
2. 使用配置的 `gpt-5.5` 生成 `model.py`。
3. 下载 Kaggle 数据或加载 Hugging Face 数据。
4. 在 GPU 上训练、评估并保存 checkpoint。

四个指定 Kaggle 比赛：

- `cassava`: Cassava Leaf Disease Classification
- `state_farm`: State Farm Distracted Driver Detection
- `siim_isic`: SIIM-ISIC Melanoma Classification
- `diabetic_retinopathy`: Diabetic Retinopathy Detection

另外十个可直接加载的数据集：

- `cifar10`, `cifar100`, `food101`, `beans`, `cats_vs_dogs`
- `stanford_cars`, `caltech101`, `eurosat`, `mnist`, `oxford_pets`

只需在下一格修改 `BENCHMARK_KEY`。Kaggle 比赛必须先在网页上接受比赛规则。
State Farm、SIIM-ISIC 和 Diabetic Retinopathy 数据较大，免费 Colab 可能需要较长下载时间和更多磁盘空间。
"""
    ),
    code(
        r"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# 只改这里即可切换任务。
BENCHMARK_KEY = "cassava"

# 快速模式仍使用真实数据，但只训练/评估一个子集。
FAST_MODE = True
EPOCHS = 2 if FAST_MODE else 10
MAX_TRAIN_SAMPLES = 2000 if FAST_MODE else 0
MAX_EVAL_SAMPLES = 500 if FAST_MODE else 0

REPO_URL = "https://github.com/Isso-W/Jiaozi.git"
REPO_REF = "codex/integration-update-colab"
REPO_DIR = Path("/content/Jiaozi")
DATA_ROOT = Path("/content/jiaozi_data")
OUTPUT_DIR = Path("/content/jiaozi_generated_training")


def normalize_repo_url(value: str) -> str:
    value = (value or "").strip()
    match = re.fullmatch(r"\[(.*?)\]\((https?://[^)]+)\)", value)
    if match:
        return match.group(2)
    match = re.search(r"https?://\S+", value)
    return match.group(0) if match else value


REPO_URL = normalize_repo_url(os.getenv("JIAOZI_REPO_URL", REPO_URL))
REPO_REF = os.getenv("JIAOZI_REPO_REF", REPO_REF)

for path in (REPO_DIR, OUTPUT_DIR):
    if path.exists():
        shutil.rmtree(path)

subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO_DIR)],
    check=True,
)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        "requirements.txt",
        "kaggle>=1.7.4.5",
    ],
    check=True,
)

from vision_benchmark_catalog import BENCHMARKS, get_benchmark

if BENCHMARK_KEY not in BENCHMARKS:
    raise KeyError(f"Unknown BENCHMARK_KEY={BENCHMARK_KEY!r}; choose from {sorted(BENCHMARKS)}")

benchmark = get_benchmark(BENCHMARK_KEY)
print("Repository:", REPO_DIR)
print("Branch:", REPO_REF)
print("Selected:", BENCHMARK_KEY, "-", benchmark["name"])
print("Source:", benchmark["source"])
print("Metric:", benchmark["metric"])

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Warning: select Runtime > Change runtime type > GPU before real training.")
except Exception as exc:
    print("Torch GPU check failed:", exc)

print("\nAvailable benchmark keys:")
for key, item in BENCHMARKS.items():
    print(f"  {key:22s} {item['source']:11s} {item['name']}")
"""
    ),
    markdown(
        """
## 配置密钥

在 Colab 左侧 **Secrets** 中添加：

- `OPENAI_API_KEY`
- `KAGGLE_API_TOKEN`（选择四个 Kaggle 比赛时需要）
- `OPENAI_BASE_URL`（可选；默认使用 `https://yybb.codes`）

密钥只写入当前 Colab 运行时的环境变量，不会写入 Git 仓库或 notebook。
之前在聊天中发送过的密钥应视为已经暴露，建议运行成功后立即撤销并重新生成。
"""
    ),
    code(
        r"""
from getpass import getpass

try:
    from google.colab import userdata
except Exception:
    userdata = None


def read_secret(name: str, required: bool = False) -> str:
    value = ""
    if userdata is not None:
        try:
            value = userdata.get(name) or ""
        except Exception:
            value = ""
    if not value and required:
        value = getpass(f"{name} (hidden input): ").strip()
    return value.strip()


openai_api_key = read_secret("OPENAI_API_KEY", required=True)
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is required.")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_BASE_URL"] = (
    read_secret("OPENAI_BASE_URL", required=False)
    or os.getenv("OPENAI_BASE_URL", "").strip()
    or "https://yybb.codes"
)
os.environ["M4_LLM_PROVIDER"] = "openai"
os.environ["M4_OPENAI_MODEL"] = "gpt-5.5"

if benchmark["source"] == "kaggle":
    kaggle_token = read_secret("KAGGLE_API_TOKEN", required=True)
    if not kaggle_token:
        raise RuntimeError("KAGGLE_API_TOKEN is required for Kaggle competitions.")
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "access_token"
    token_path.write_text(kaggle_token, encoding="utf-8")
    token_path.chmod(0o600)
    del kaggle_token

del openai_api_key
print("LLM provider: openai")
print("Configured model: gpt-5.5")
print("Custom OpenAI-compatible endpoint configured:", bool(os.environ.get("OPENAI_BASE_URL")))
print("Kaggle token configured:", bool(os.environ.get("KAGGLE_API_TOKEN")))
"""
    ),
    markdown(
        """
## 准备数据

Kaggle 路径会自动下载比赛文件并递归解压 ZIP。若出现 `403`，先打开比赛页面接受规则；
若出现磁盘不足，请换用 Colab Pro、连接 Google Drive，或选择较小的数据集。
"""
    ),
    code(
        r"""
import zipfile


def run_visible(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    completed.check_returncode()


def extract_zip_archives(root: Path) -> None:
    extracted: set[Path] = set()
    while True:
        archives = [
            path
            for path in root.rglob("*.zip")
            if path.resolve() not in extracted and zipfile.is_zipfile(path)
        ]
        if not archives:
            return
        for archive in archives:
            print("Extracting:", archive)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(archive.parent)
            extracted.add(archive.resolve())


def find_first(root: Path, patterns: list[str], expect_directory: bool) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir() == expect_directory:
                matches.append(path)
    if not matches:
        kind = "directory" if expect_directory else "file"
        raise FileNotFoundError(
            f"Could not find expected {kind} under {root}. Patterns: {patterns}"
        )
    return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0]


runtime_data: dict[str, object] = {}
if benchmark["source"] == "kaggle":
    competition_dir = DATA_ROOT / BENCHMARK_KEY
    if competition_dir.exists():
        shutil.rmtree(competition_dir)
    competition_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_visible(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                benchmark["competition"],
                "-p",
                str(competition_dir),
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. Confirm KAGGLE_API_TOKEN, accept the competition "
            f"rules, and check free disk space. Competition: {benchmark['competition']}"
        ) from exc

    extract_zip_archives(competition_dir)
    train_csv = find_first(competition_dir, benchmark["csv_globs"], False)
    image_dir = find_first(competition_dir, benchmark["image_dir_globs"], True)
    runtime_data.update(
        {
            "train_csv": str(train_csv),
            "image_dir": str(image_dir),
            "image_column": benchmark["image_column"],
            "label_column": benchmark["label_column"],
            "image_path_template": benchmark["image_path_template"],
            "image_extension": benchmark["image_extension"],
        }
    )
    print("Training CSV:", train_csv)
    print("Image directory:", image_dir)
    run_visible(["du", "-sh", str(competition_dir)])
else:
    runtime_data["dataset_id"] = benchmark["dataset_id"]
    print("Hugging Face dataset:", benchmark["dataset_id"])

print(json.dumps(runtime_data, indent=2, ensure_ascii=False))
"""
    ),
    markdown(
        """
## 使用 GPT-5.5 生成训练项目

这一步直接调用 Module 4。生成后会检查 `generation_info.json`：
若 API 调用失败并回退到模板，单元格会明确报错，不会假装 GPT-5.5 已经参与。
"""
    ),
    code(
        r"""
from module4_agent.workflow import run_workflow

candidate = {
    "rank": 1,
    "score": 1.0,
    "model_config": {
        "task_type": "classification",
        "backbone": benchmark["backbone"],
        "head": "classification_head",
        "loss": benchmark["loss"],
        "optimizer": "adamw",
        "learning_rate": 0.0003,
        "num_classes": benchmark["num_classes"],
        "image_size": 224,
        "offline_smoke": True,
        "use_pretrained": True,
        "finetune_strategy": "full",
        "freeze_backbone": False,
        "evaluation_metric": benchmark["metric"],
        "benchmark_name": benchmark["name"],
        **runtime_data,
    },
}

module3_input = Path("/content/jiaozi_selected_candidate.json")
module3_input.write_text(
    json.dumps([candidate], indent=2, ensure_ascii=False),
    encoding="utf-8",
)

result = run_workflow(
    module3_input,
    OUTPUT_DIR,
    max_iter=2,
    timeout=180,
    skip_smoke=False,
    llm_provider="openai",
)
if not result.is_approved:
    raise RuntimeError(
        "Generated project did not pass smoke/review checks:\n"
        + json.dumps(result.to_summary(), indent=2, ensure_ascii=False)
    )

generation_info_path = OUTPUT_DIR / "generation_info.json"
generation_info = json.loads(generation_info_path.read_text(encoding="utf-8"))
print(json.dumps(generation_info, indent=2, ensure_ascii=False))

if not generation_info.get("llm_used"):
    raise RuntimeError(
        "GPT code generation was not used. The API call fell back to the template. "
        "Check OPENAI_API_KEY, OPENAI_BASE_URL, and whether the endpoint provides gpt-5.5."
    )
if generation_info.get("llm_model") != "gpt-5.5":
    raise RuntimeError(f"Unexpected generation model: {generation_info.get('llm_model')!r}")

generated_configs = json.loads((OUTPUT_DIR / "configs.json").read_text(encoding="utf-8"))
runtime_config = dict(generated_configs[0])
runtime_config.update(runtime_config.pop("model_config", {}) or {})
runtime_config.update(
    {
        "offline_smoke": False,
        "use_pretrained": True,
        "finetune_strategy": "full",
        "freeze_backbone": False,
        "evaluation_metric": benchmark["metric"],
        "recommended_epochs": EPOCHS,
        "batch_size": 16 if benchmark["backbone"] == "efficientnet_b3" else 32,
        "num_workers": 2,
        "validation_fraction": 0.2,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        **runtime_data,
    }
)
(OUTPUT_DIR / "configs.json").write_text(
    json.dumps(runtime_config, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Generated project:", OUTPUT_DIR)
print("Real-training config:")
print(json.dumps(runtime_config, indent=2, ensure_ascii=False))
"""
    ),
    markdown(
        """
## 开始真实训练

快速模式默认训练最多 2,000 张、评估 500 张并运行 2 个 epoch。
设置 `FAST_MODE = False` 后重新从第一格运行，即可使用完整训练集和 10 个 epoch。
"""
    ),
    code(
        r"""
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    cwd=OUTPUT_DIR,
    check=True,
)

training_command = [
    sys.executable,
    "run.py",
    "--config",
    "configs.json",
    "--epochs",
    str(EPOCHS),
]
print("$", " ".join(training_command))

process = subprocess.Popen(
    training_command,
    cwd=OUTPUT_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
training_lines: list[str] = []
assert process.stdout is not None
for line in process.stdout:
    print(line, end="")
    training_lines.append(line)
return_code = process.wait()

training_log = OUTPUT_DIR / "training_output.txt"
training_log.write_text("".join(training_lines), encoding="utf-8")
if return_code != 0:
    raise subprocess.CalledProcessError(return_code, training_command)

print("\nTraining log:", training_log)
print("Checkpoints:")
for path in sorted((OUTPUT_DIR / "checkpoints").glob("*.pt")):
    print(" ", path, f"({path.stat().st_size / 1024 / 1024:.1f} MB)")
"""
    ),
    markdown(
        """
## 检查生成与训练结果

最后一格显示 Module 4 审核摘要、GPT 使用记录、实际配置和生成文件。
"""
    ),
    code(
        r"""
summary = json.loads((OUTPUT_DIR / "module4_summary.json").read_text(encoding="utf-8"))
generation_info = json.loads((OUTPUT_DIR / "generation_info.json").read_text(encoding="utf-8"))
training_config = json.loads((OUTPUT_DIR / "configs.json").read_text(encoding="utf-8"))

print("=== Module 4 summary ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\n=== GPT generation record ===")
print(json.dumps(generation_info, indent=2, ensure_ascii=False))
print("\n=== Training config ===")
print(json.dumps(training_config, indent=2, ensure_ascii=False))
print("\n=== Generated files ===")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "vision_benchmarks_colab.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path(__file__).with_name("vision_benchmarks_colab.ipynb")
output_path.write_text(
    json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(output_path)
