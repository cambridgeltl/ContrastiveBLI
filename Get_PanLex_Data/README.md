## Get PanLex-BLI Data

- Download bilingual dictionaries from the [PanLex-BLI](https://github.com/cambridgeltl/panlex-bli) repo:
    ```bash 
    wget https://github.com/cambridgeltl/panlex-bli/raw/master/lexicons/all-l1-l2.zip
    unzip all-l1-l2.zip
    ```
- Install fastText: 
    ```bash
    git clone https://github.com/facebookresearch/fastText.git
    cd fastText
    pip install .
    ```
- Get monolingual embeddings: 
    ```bash
    python get_panlex_embs.py
    ```
## Attention

In PanLex-BLI, source->target and target->source translation tasks have different training and test dictionaries. So, it is needed to train two BLI models respectively for source->target and target->source translations.
