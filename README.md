# MLM4Rec
The source code for our DASFFA Submission 65 Paper **"Learning Global and Multi-granularity Local
Representation with MLP for Sequential
Recommendation"**


## Requirements
* Install Python, Pytorch(>=1.8). We use Python 3.7, Pytorch 1.8.
* If you plan to use GPU computation, install CUDA.



## Quick-Start
If you have downloaded the source codes, you can train the model just with data_name input.
```
python main.py --data_name=[data_name]
```

If you want to change the parameters, just set the additional command parameters as you need. For example:
```
python main.py --data_name=Beauty --num_hidden_layers=4 --batch_size=512
```

We present the optimal models on the three datasets and their experimental results.
You can test the model has been saved by command line.

Beauty:
```
python main.py --data_name=Beauty --do_eval --load_model=MLM4Rec-Beauty-0 --hidden_size=192 --cmlp_type=3 --data_aug
```
Yelp:
```
python main.py --data_name=Yelp --do_eval --load_model=MLM4Rec-Yelp-0 --cmlp_type=2 --max_seq_length=30 --data_aug
```
ml-1m:
```
python main.py --data_name=ml-1m --do_eval --load_model=MLM4Rec-ml-1m-0 --cmlp_type=2 --max_seq_length=300 --hidden_dropout_prob=0.2
```

Additional hyper-parameters can be specified, and detailed information can be accessed by:

```
python main.py --help
```

