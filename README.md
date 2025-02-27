# UGBench

This repository contains the official implementation of the paper "Erasing Without Remembering: Safeguarding Knowledge Forgetting in Large Language Models".

**Requirements**

```
conda create -n unlearn python=3.8.19
conda activate unlearn
pip install -r requirements.txt
``` 

**Dataset**
Download the data from 

**Running the Code**

Unlearning on TOFU dataset: 

```
bash scripts/forget_tofu.sh
```

Unlearning on Harry Potter dataset: 

```
bash scripts/forget_harry.sh
```

Unlearning on ZSRE dataset: 

```
bash scripts/forget_zsre.sh
```