#Get XLING DATA from XLING repo https://github.com/codogogo/xling-eval.git
mkdir -p /media/data
cd /media/data # replace with [YOUR DIR]
git clone https://github.com/codogogo/xling-eval.git

mkdir -p /media/data/WES
cd /media/data/WES # replace with [YOUR DIR]
#Get XLING preprocessed Word Embeddings from https://tinyurl.com/y5shy5gt, which is also provided by the XLING repo
gdown https://drive.google.com/uc?id=1YeQFauynj3JRwV6svWdyy0fMBwYMKNNd # de
gdown https://drive.google.com/uc?id=1wy-OlRycb5_dEB0w52dkOSJqfEXDytDT # en
gdown https://drive.google.com/uc?id=1wHJJk8yKu0yWsCi7wRf0SoWb_YwyZbBi # fi
gdown https://drive.google.com/uc?id=1nK5kdpUvG8L3q9q2IqTFFZnqdzKCgHGo # fr
gdown https://drive.google.com/uc?id=1fz1R3vayiIHd3OIfDS3HJ1yChCOrQguj # hr
gdown https://drive.google.com/uc?id=1x6tL-Hh7LVQ0KbMiWe9DI4TO1uRBFghM # it
gdown https://drive.google.com/uc?id=12pIp9zsLmeF5O-512Q1OmohU3AMVlU7a # ru
gdown https://drive.google.com/uc?id=10rryBWx5KWA136utnrEF99MJNB4h2-Xj # tr
