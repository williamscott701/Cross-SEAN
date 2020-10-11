import json
import pickle
import os
import torch

from config import get_train_args
from training import Training
from general_utils import get_logger

import numpy as np

args = get_train_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
logger = get_logger(args.log_path)
logger.info(json.dumps(args.__dict__, indent=4))

train_data = torch.load(os.path.join(args.input, args.save_data + ".train.pth"))
ek_data = torch.load(os.path.join(args.input, args.save_data + ".external.pth"))
ekt_data = torch.load(os.path.join(args.input, args.save_data + ".externalT.pth"))
dev_data = torch.load(os.path.join(args.input, args.save_data + ".valid.pth"))
test_data = torch.load(os.path.join(args.input, args.save_data + ".test.pth"))
unlabel_data = torch.load(os.path.join(args.input, args.save_data + ".unlabel.pth"))

addn_data = torch.load(os.path.join(args.input, args.save_data + ".train_addn.pth"))
addn_data_t = torch.load(os.path.join(args.input, args.save_data + ".valid_addn.pth"))
addn_data_t = torch.load(os.path.join(args.input, args.save_data + ".test_addn.pth"))
addn_data_unlab = torch.load(os.path.join(args.input, args.save_data + ".unlabel_addn.pth"))

with open(os.path.join(args.input, args.save_data + '.vocab.pickle'),
          'rb') as f:
    id2w = pickle.load(f)

with open(os.path.join(args.input, args.save_data + '.label.pickle'),
          'rb') as f:
    id2label = pickle.load(f)

args.id2w = id2w
args.n_vocab = len(id2w)
args.id2label = id2label
args.num_classes = len(id2label)

object = Training(args, logger)

logger.info('Corpus: {}'.format(args.corpus))
logger.info('Pytorch Model')
logger.info(repr(object.embedder))
logger.info(repr(object.encoder))
logger.info(repr(object.clf))
logger.info(repr(object.clf_loss))
if args.lambda_ae:
    logger.info(repr(object.ae))

object(train_data, dev_data, test_data, unlabel_data, addn_data, addn_data_unlab, addn_data_t, ek_data, ekt_data)