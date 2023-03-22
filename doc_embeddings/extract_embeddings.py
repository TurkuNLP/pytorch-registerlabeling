import transformers
import sys
import numpy as np
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import glob
import gzip

from common import write
from common import write_l


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', required=True,
                    help='Pretrained model name')
    ap.add_argument('--data', required=True,
                    help='Path to data')
    ap.add_argument('--model', default=None, metavar='FILE',
                    help='Path to model')
    ap.add_argument('--lang', required=False,
                    help='language')
    return ap


options = argparser().parse_args(sys.argv[1:])


tokenizer = transformers.AutoTokenizer.from_pretrained(options.model_name) #e.g. TurkuNLP/bert-base-finnish-cased-v1")
model = torch.load(options.model, map_location=torch.device('cpu'))

#labels_total = []
#embeddings = []
counter = 0

#files = []

#dirs = options.data.split("-")
#for di in dirs:
#    files = files + glob.glob(di+"/*")

#print("files", files)
#f = glob.glob(options.data+"/*")

def return_embeddings(dir):
    embeddings = []
    labels_total = []
    ids = []
    for file in glob.glob(dir+"/data.txt_predsmt.txt.gz"):
        print("path", file)
        fin = gzip.open(file, "rt")
        for line in fin:
            if len(line) < 2:
                continue
            else:
                line=line.split("\t")
                labels=line[0].split(" ",1)[1]
                id = line[0].split(" ",1)[0]
                text=line[2]
                ls=labels.split(" ")
                labels_total.append(labels)
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                #input_enc = tokenizer(text, truncation=True, max_length=512)
                #print(input_enc)
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                #            print(last_hidden_states.shape)
                #   print(last_hidden_states)
                last_layer = last_hidden_states[0]
                #           print("last layer shape", last_layer.shape)
                #print(last_layer)
                embeddings.append(last_layer[0,:].detach().numpy()) # take the index 0 so CLS
                #          print("CLS shape", last_layer[0,:].shape)
    print("Total number of embeddings", len(embeddings), "for", dir)
    write(embeddings, options.data+"/"+options.lang+"_embeddings.txt")
    write_l(labels_total, options.data+"/"+options.lang+"_labels.txt")
    write_l(ids, options.data+"/"+options.lang+"_ids.txt")

    
#for di in options.data.split("-"):
return_embeddings(options.data)

    
