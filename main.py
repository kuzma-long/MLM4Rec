import os
import torch
import argparse
import numpy as np

from models import FMLPRecModel
from trainers import FMLPRecTrainer
from utils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_dataloader, get_rating_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Yelp", type=str,help="Beauty,ml-1m,Sports_and_Outdoors,Toys_and_Games,Yelp")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default="MLM4Rec-ml-1m-0", type=str)
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")

    # model args
    parser.add_argument("--model_name", default="MLM4Rec", type=str)
    parser.add_argument("--model_desc", default="0", type=str)
    parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--kernel", default=3, type=int, help="[1,3,5,7,9,11]")
    parser.add_argument("--shift_size", default=1, type=int)
    parser.add_argument("--cmlp_type", default=1, type=int, help="[1,2]")
    parser.add_argument('--noise_ratio', default=0.0, type=float,
                        help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument("--merge_type", default="weightavg", type=str, help="avg,concatenate")

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=9000, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--patience", default=40, type=int,
                        help="how long to wait after last time validation loss improved")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--variance", default=5, type=float)

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())

    seq_dic, max_item = get_seq_dic(args)

    args.item_size = max_item+1

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_desc}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    args.checkpoint_path = os.path.join(args.output_dir, args_str + '.pt')

    train_dataloader, eval_dataloader, test_dataloader = get_dataloader(args, seq_dic)

    model = FMLPRecModel(args=args)
    trainer = FMLPRecTrainer(model, train_dataloader, eval_dataloader,
                             test_dataloader, args)

    args.valid_rating_matrix, args.test_rating_matrix = seq_dic['valid_rating_matrix'],seq_dic['test_rating_matrix']

    if args.do_eval:
        if args.load_model is None:
            print(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            print(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0, full_sort=True)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=True)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("---------------Full sort results---------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

if __name__ == '__main__':
    main()
