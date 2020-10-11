#!/usr/bin/env bash

NAME="covid19"
OUT="temp/$NAME"
datadir="."
addndata="./data/covid19"

python extract_features.py

mkdir -p ${OUT}

python preprocess.py --corpus covid19 --output ${OUT}/data --vocab_size 50000 --save_data "demo"

echo "preprocessing done"

python w2v.py --input ${OUT}/data --save_data "demo" --embeddings "${datadir}/crawl-300d-2M.vec" --addndata ${addndata}

echo "Training"

python main.py --corpus covid19 --model LSTMEncoder --debug \
--multi_gpu --input ${OUT}/data --output_path ${OUT}/model --exp_name "covid19" \
--use_pretrained_embeddings --nepoch_no_imprv 20 --timedistributed \
--d_down_proj 256 --d_units 300 --d_hidden 512 --nepochs 20 \
--optim adam --beta1 0.0 --beta2 0.98 \
--inc_unlabeled_loss --wbatchsize 1024 --wbatchsize_unlabel 1024 \
--lambda_at 1.0 --lambda_vat 1.0 --lambda_entropy 0.0 --lambda_clf 1.0