import os
import argparse
import warnings
import model.dch as model
import data_provider.image as dataset

from pprint import pprint

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Triplet Hashing')
parser.add_argument('--num_similar_pairs', help="specify the number of similar image pairs. \
                                Default is 0, which doesn't mean 0 similar pairs. Rather, it means batches will be drawn randomly", \
                        type=int, default=0)
parser.add_argument('--class_size', help='specify the number of images in a class', type=int, default=4)
parser.add_argument('--num-class', help="specify the number of classes", type=int, default=200000)
parser.add_argument('--encode', help="specify the file path to save hash codes of test images")
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float)
parser.add_argument('--output-dim', default=64, type=int)   # 256, 128
parser.add_argument('--gamma', default=20, type=float)
parser.add_argument('--iter-num', default=2000, type=int)
parser.add_argument('--q-lambda', default=0, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--log-dir', default='tflog', type=str)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-vb', '--val-batch-size', default=16, type=int)
parser.add_argument('--decay-step', default=10000, type=int)
parser.add_argument('--decay-factor', default=0.1, type=float)

tanh_parser = parser.add_mutually_exclusive_group(required=False)
tanh_parser.add_argument('--with-tanh', dest='with_tanh', action='store_true')
tanh_parser.add_argument('--without-tanh', dest='with_tanh', action='store_false')
parser.set_defaults(with_tanh=True)

parser.add_argument('--img-model', default='alexnet', type=str)
parser.add_argument('--model-weights', type=str,
                    default='../../DeepHash/architecture/pretrained_model/reference_pretrain.npy')
parser.add_argument('--finetune-all', default=True, type=bool)
parser.add_argument('--save-dir', default="./models/", type=str)
parser.add_argument('--data-dir', default="~/data/", type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')

args = parser.parse_args()

assert args.num_similar_pairs*2 < args.batch_size, "Number of similar pairs cannot be bigger than batch_size"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 81, 'coco': 80, 'train': 1}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000, 'train': 5000}
args.R = Rs[args.dataset]
args.label_dim = label_dims[args.dataset]
args.use_custom_dataset = True
if args.dataset in ['cifar10', 'cub', 'nuswide_81', 'coco']:
    args.use_custom_dataset = False

args.img_tr = os.path.join(args.data_dir, args.dataset, "train.txt")
args.img_te = os.path.join(args.data_dir, args.dataset, "test.txt")
args.img_db = os.path.join(args.data_dir, args.dataset, "database.txt")

pprint(vars(args))

data_root = os.path.join(args.data_dir, args.dataset)
query_img, database_img = dataset.import_validation(data_root, args.img_te, args.img_db)

if not args.evaluate:
    train_img = dataset.import_train(data_root, args.img_tr)
    model_weights = model.train(train_img, database_img, query_img, args)
    args.model_weights = model_weights

maps = model.validation(database_img, query_img, args)
for key in maps:
    print(("{}\t{}".format(key, maps[key])))

pprint(vars(args))
