# Personalized Adapter for Large Meteorology Model on Devices: Towards Weather Foundation Models

### Official implementation for ``Personalized Adapter for Large Meteorology Model on Devices: Towards Weather Foundation Models``, NeurIPS 2024.

---
:triangular_flag_on_post:**News** (2024.12) The official implementation of our paper is posted, and we are still shaping our code and dataset for achieving a complete repo.
:triangular_flag_on_post:**News** (2024.09) Our paper has been accepted by **NeurIPS 2024** and we will be releasing the code for our paper ASAP.
---

**Abstract**

This paper demonstrates that pre-trained language models (PLMs) are strong foundation models for on-device meteorological variable modeling. We present LM-Weather, a generic approach to taming PLMs, that have learned massive sequential knowledge from the universe of natural language databases, to acquire an immediate capability to obtain highly customized models for heterogeneous meteorological data on devices while keeping high efficiency. Concretely, we introduce a lightweight personalized adapter into PLMs and endows it with weather pattern awareness. During communication between clients and the server, low-rank-based transmission is performed to effectively fuse the global knowledge among devices while maintaining high communication efficiency and ensuring privacy. Experiments on real-wold dataset show that LM-Weather outperforms the state-of-the-art results by a large margin across various tasks (e.g., forecasting and imputation at different scales). We provide extensive and in-depth analyses experiments, which verify that LM-Weather can (1) indeed leverage sequential knowledge from natural language to accurately handle meteorological sequence, (2) allows each devices obtain highly customized models under significant heterogeneity, and (3) generalize under data-limited and out-of-distribution (OOD) scenarios.

---

### Please cite our publication if you found our research to be helpful and insightful.

```bibtex
@inproceedings{
chen2024personalized,
title={Personalized Adapter for Large Meteorology Model on Devices: Towards Weather Foundation Models},
author={Shengchao Chen and Guodong Long and Jing Jiang and Chengqi Zhang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=llTroju97T}
}
