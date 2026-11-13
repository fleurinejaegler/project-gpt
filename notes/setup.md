# # Mise en place de nanoGPT sur Google Colab (Shakespeare, niveau caractère)

## Environnement

Plateforme : Google Colab
GPU : Tesla T4 (15 360 MiB de mémoire)
Pilote NVIDIA : 580.82.07
Version CUDA (nvidia-smi) : 13.0
Version de PyTorch : 2.14.1+cu130
Version de Python : 3.13.16
Commit de nanoGPT utilisé : 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
## Étapes

Cloner le dépôt : !git clone https://github.com/karpathy/nanoGPT.git, puis %cd nanoGPT.
Préparer les données : !python data/shakespeare_char/prepare.py. Fichiers produits : input.txt (1,1 Mo), train.bin (2,0 Mo), val.bin (218 Ko), meta.pkl (703 octets). Taille du vocabulaire attendue : 65.
Entraînement court : !python train.py config/train_shakespeare_char.py --max_iters=500 --compile=False --eval_iters=20 --log_interval=10 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --device=cuda --dtype=float16
Génération de texte : !python sample.py --out_dir=out-shakespeare-char --device=cuda --dtype=float16 --num_samples=3 --max_new_tokens=300
## Résultats observés

Itération 500 : perte d'entraînement 2,41, perte de validation 2,43.
Perte initiale d'environ 4,2 (prédiction quasi aléatoire sur 65 caractères).
Texte généré : structure de pièce reconnaissable (noms en majuscules suivis de deux-points), mots souvent incomplets. Sous-apprentissage, sans surapprentissage.
Durée : environ 176 ms par itération en fin d'entraînement, mfu d'environ 0,1 %.
