# π₀.₅ (pi05)

This repository contains the Hugging Face port of **π₀.₅**, adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi) by the Physical Intelligence.
It is designed as a **Vision-Language-Action model with open-world generalization**.

---

## Model Overview

| Feature              | π₀                                                     | π₀.₅                                      |
| -------------------- | ------------------------------------------------------ | ----------------------------------------- |
| Time Conditioning    | Concatenates time with actions via `action_time_mlp_*` | Uses `time_mlp_*` for AdaRMS conditioning |
| AdaRMS               | Not used                                               | Used in action expert                     |
| Tokenizer Length     | 48 tokens                                              | 200 tokens                                |
| Discrete State Input | False (Uses `state_proj` layer)                        | True                                      |
| Parameter Count      | Higher (includes state embedding)                      | Lower (no state embedding)                |

---

## Local PaliGemma Tokenizer

π₀.₅ uses the PaliGemma tokenizer during preprocessing. By default it still uses
`google/paligemma-3b-pt-224`, but local paths are supported to avoid gated Hub downloads.

```bash
export LEROBOT_PALIGEMMA_TOKENIZER=/home/phl/workspace/models/paligemma-tokenizer
```

You can also set it directly in policy config or CLI overrides:

```bash
--policy.paligemma_tokenizer_name=/home/phl/workspace/models/paligemma-tokenizer
```

## Optional Tactile Input

π₀.₅ can optionally consume O10 tactile heatmaps. Keep `use_tactile=false` for normal image/state
training and inference. Set `use_tactile=true` only when the dataset contains 2D tactile maps.

Supported tactile feature keys:

```text
observation.tactile
observation.tactile.left
observation.tactile.right
```

When `tactile_features=null`, the config auto-detects whether the dataset is single left hand,
single right hand, or dual hand. Dual hand order is normalized to left then right. The expected
default map shape is `(12, 32)`, matching the O10 130D-to-heatmap conversion.

Example training overrides:

```bash
--policy.type=pi05 \
--policy.use_tactile=true \
--policy.tactile_input_shape='[12,32]' \
--policy.tactile_encoder_type=cnn \
--policy.paligemma_tokenizer_name=/home/phl/workspace/models/paligemma-tokenizer
```

For a non-tactile π₀.₅ run, leave `--policy.use_tactile=false`; tactile fields are ignored and the
model behaves like the regular π₀.₅ policy.

---

## Citation

If you use this work, please cite both **OpenPI** and the π₀.₅ paper:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
  title        = {π₀.₅: a Vision-Language-Action Model with Open-World Generalization},
  author       = {Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
  year         = {2025},
  eprint       = {2504.16054},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.16054},
}
```

---

## License

This port follows the **Apache 2.0 License**, consistent with the original [OpenPI repository](https://github.com/Physical-Intelligence/openpi).
