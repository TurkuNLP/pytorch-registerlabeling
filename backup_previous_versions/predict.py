import transformers
import torch
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gzip

torch.set_num_threads(2)
MODEL_NAME = 'xlm-roberta-base'

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=MODEL_NAME,
                    help='Pretrained model name')
    ap.add_argument('--text', metavar='FILE', required=True,
                    help='Text to be predicted') #could also be string?
    ap.add_argument('--file_type', choices=['tsv', 'txt'], default='txt')
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='Load model from file')
    ap.add_argument('--threshold', default=0.4, metavar='FLOAT', type=float,
                    help='threshold for calculating f-score')
    ap.add_argument('--labels', choices=['full', 'upper'], default='full')
    ap.add_argument('--output', default=None, metavar='FILE', help='Location to save predictions')
    return ap

options = argparser().parse_args(sys.argv[1:])

labels_full = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
labels_upper = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']

if options.labels == 'full':
    labels = labels_full
else:
    labels = labels_upper
num_labels = len(labels)
#print("Number of labels:", num_labels)

# tokenizer from the pretrained model
tokenizer = transformers.AutoTokenizer.from_pretrained(options.model_name)

# load our fine-tuned model
model = torch.load(options.load_model, map_location=torch.device('cpu'))
#model.to('cpu')

def predict_labels(string):
    prlist = []
    labellist = []
    #tokenized = tokenizer(string, return_tensors='pt') V
    tokenized = tokenizer(string, truncation=True, max_length=512, return_tensors='pt')
    pred = model(**tokenized)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.logits.detach().numpy()))
#    print("XXX probs", probs)

    pr = probs.flatten()
    preds = np.zeros(probs.shape)
    preds[np.where(probs >= options.threshold)] = 1
 #   print("PREDS", preds)
    for iix, l in enumerate(pr.tolist()):
        if l >= options.threshold:
           prlist.append(l)
           labellist.append(labels[iix])
    if len(prlist) == 0:
        prlist.append("No_labels")
    if len(labellist) == 0:
        labellist.append("No_labels")
    return [labellist,prlist]

# predict labels text at a time
#outf = open(options.text+'_preds.txt', 'w')

with open(options.text, 'r') as f:
    for line in f:
            text = line.split('\t')[1]
#            print(" ".join(predict_labels(text)[0]), flush=True)
            floats = predict_labels(text)[1]
            strs = [str(x) for x in floats]
            print(" ".join(predict_labels(text)[0]) + "\t" + " ".join(strs) + "\t" + line.split("\t")[1], flush=True)
f.close()
