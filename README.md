# HELMET: radiology report language models

Companion to [HELMET_stroke_mass_effect_prediction](https://github.com/ethanp274/HELMET_stroke_mass_effect_prediction).
This repository fine-tunes [Clinical-Longformer](https://huggingface.co/yikuan8/Clinical-Longformer)
on radiology reports to predict midline-shift severity class at 8, 24 and 36
hours. The class probabilities from those classifiers become features for the
XGBoost models in the main repository.

> **Paper** — Phillips E, O'Donoghue O, Zhang Y, *et al.* Hybrid machine learning for real-time prediction of edema trajectory in large middle cerebral artery stroke. *npj Digital Medicine* **8**, 288 (2025). [doi:10.1038/s41746-025-01687-y](https://doi.org/10.1038/s41746-025-01687-y)

## Status

Research code released alongside a published paper. It is the code that
produced the published language models, consolidated from two working copies
and made installable. It is not a maintained library. **Nothing here is
validated for clinical use.**

## Data

**No data is included.** Radiology reports are identifiable patient records,
and every dataset derived from them — including the templated text narratives
built from structured EHR data — is patient data too. `.gitignore` excludes
`data/` and `models/` wholesale.

The fine-tuned classifiers are on the HuggingFace Hub as
`ethanp5/edema_prediction_LLM_8hr`, `_24hr` and `_36hr`, with access granted by
the authors on request.

## Installation

```bash
git clone https://github.com/ethanp274/HELMET_stroke_radiology_llm.git
cd HELMET_stroke_radiology_llm
uv sync
```

This installs the main `helmet-mass-effect-pred` package (pinned to its `v1.0.0`
tag), which provides the data pipeline and every pinned dependency. Training
realistically needs a CUDA GPU; the scripts use fp16.

### Where the environment lives

Keep the virtual environment **outside** the repository. uv's default is a `.venv` folder
inside the project, which goes badly in a synced folder such as OneDrive: it churns tens of
thousands of small files, locks them mid-install and turns them into cloud placeholders that
then fail at import time. Point uv at a local directory before your first `uv` command, in
every new shell (or your shell profile):

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\uv-envs\HELMET_stroke_radiology_llm"
```

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/uv-envs/HELMET_stroke_radiology_llm"   # macOS / Linux
```

Then run `uv sync` and `uv run` as normal, and confirm the interpreter lives there:

```bash
uv run python -c "import sys; print(sys.executable)"
```

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `scripts/merge_radiology_reports.py` | Joins report text onto the longitudinal EHR tables by patient and timestamp, dropping direct identifiers. |
| 2 | `scripts/finetune_classifier.py` | **Final method.** Fine-tunes Clinical-Longformer as a 4-class midline-shift classifier on reports, split by patient, and evaluates on the MGB test split and the BMC cohort. Run once per horizon (`--lookahead_hours 8/24/36`). |
| 3 | `scripts/evaluate_classifier.py` | Scores a fine-tuned classifier on held-out MGB patients and on BMC. |

Earlier approaches, kept for completeness:

| Script | Approach |
| --- | --- |
| `src/helmet_mass_effect_llm/text_generation.py` | Renders structured EHR rows as templated clinical narratives, then fine-tunes Clinical-Longformer and BioGPT on them. |
| `scripts/domain_adapt_longformer.py` | Masked-language-model domain adaptation of Clinical-Longformer on those narratives. |
| `scripts/finetune_classifier_shorttext.py` | An intermediate version of step 2 using truncated report text. |
| `exploratory/` | Zero-shot prompting smoke tests for BioGPT and Clinical-Longformer. |

## Provenance

The code was consolidated from two diverging working copies, `text_transformers`
and `text_transformers_2`. Where they differed:

- `text_generation.py`, `merge_radiology_reports.py` and `evaluate_classifier.py`
  come from the first copy, which kept GPU training and the full MGB and BMC
  report merge.
- `domain_adapt_longformer.py` and `finetune_classifier_shorttext.py` come from
  the second copy, which fixed the tokeniser call and model ID in the former.
- Hard-coded virtual-machine output paths (`Q:/…`) now point to `models/`.
- Nested-quote f-strings, which need Python 3.12, were rewritten for the pinned
  Python 3.11. Nothing else changed.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{phillips2025helmet,
  title   = {Hybrid machine learning for real-time prediction of edema trajectory in large middle cerebral artery stroke},
  author  = {Phillips, Ethan and O'Donoghue, Odhran and Zhang, Yumeng and Tsimpos, Panos and Mallinger, Leigh Ann and Chatzidakis, Stefanos and Pohlmann, Jack and Du, Yili and Kim, Ivy and Song, Jonathan and Brush, Benjamin and Smirnakis, Stelios and Ong, Charlene J. and Orfanoudaki, Agni},
  journal = {npj Digital Medicine},
  volume  = {8},
  pages   = {288},
  year    = {2025},
  doi     = {10.1038/s41746-025-01687-y}
}
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).
