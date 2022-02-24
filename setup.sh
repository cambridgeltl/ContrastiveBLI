apt-get update
python -m pip install -U pip
apt-get install language-pack-en
pip install pytorch-metric-learning --pre
pip install networkx==2.5
pip install pandas==1.1.3
pip install scikit_learn==0.24.1
pip install transformers==4.4.2
pip install wandb
pip install fasttext
conda install faiss-gpu -c pytorch