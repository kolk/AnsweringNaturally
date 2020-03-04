# Answering Naturally : Factoid to Full length Answer Generation 

Code base for paper Answering Naturally : Factoid to Full length Answer Generation (https://www.aclweb.org/anthology/D19-5401.pdf) .The dataset is contained in data directory. train.ques, train.ans, train.tgt contains data triplet (question, factoid answer, target full length answer) in each line respectively.

The codebase is built over [OpenNMT](https://github.com/OpenNMT/OpenNMT)

## Requirements
All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

### Step 1: Preprocess the data

## Add padding to all the data files ###
```bash
python add_padding.py -input data/train.ques -output data/train_padded.ques
python add_padding.py -input data/train.ans -output data/train_padded.ans
python add_padding.py -input data/train.tgt -output data/train_padded.tgt
```
## Preprocess the padded data
```
python preprocess.py -train_src data/train_padded.ques -train_tgt data/train_padded.tgt -train_ans data/train_padded.ans -valid_src data/val_padded.ques -valid_tgt data/val_padded.tgt -valid_ans data/val_padded.ans -save_data data/demo -share_vocab -dynamic_dict 
```
After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


### Step 2: Train the model

```bash
python train.py -word_vec_size 512 -encoder_type rnn -layers 3 -rnn_size 512 -data data/demo -save_model models/model -batch_size 32 -valid_steps 2500 -dropout 0.5  -start_decay_steps 10000 -valid_steps 2500 -coverage_attn -copy_attn 
```

### Step 3: Translate

```bash
python translate.py -src data/test.ques  -tgt data/test.tgt -model models/model.pt -output pred.txt -replace_unk -verbose -beam_size 50 -dynamic_dict 
```
The output predictions are in `pred.txt`.

