def write(list,name):
  fout = open(name, "w")
  for line in list:
    new = [str(a) for a in line]
    fout.write(" ".join(new))
    fout.write("\n")
  fout.close()

def write_l(list,name):
  fout = open(name, "w")
  for line in list:
    fout.write(line)
    fout.write("\n")
  fout.close()

def read(inp, name):
  inp = open(inp, "r")
  name = []
  for line in inp:
      line2=line.strip().split(" ")
      line3 = [float(a) for a in line2]
      name.append(line3)
  return name

def read_l(inp, name):
    inp = open(inp, "r")
    name = []
    for line in inp:
      line=line.strip()
      name.append(line)
    return name


def filter(label):
    #! cat ../oscar-data/fin/fin_labels.txt | sort | uniq -ci | sort -rn | head -20 | perl -pe 's/[0-9]//g' | perl -pe 's/^ +//g' > top_labels.tt
    top = ["NA nb", "IP ds", "NA ne", "No_labels", "ID", "HI", "IN dtp", "NA", "OP rv", "NA sr", "IN", "HI re", "OP ob", "IN lt", "IN ra", "IP", "nb", "OP rs", "IN en", "MT"]
    if label not in top:
        return None
    else:
        return label
    
from collections import Counter

def top20(embeddings, labels,max):
    i_to_keep = []
    l_to_keep = []
    e_to_keep=[]
    lbs = []
    top = []
    for ind, l in enumerate(labels):
        lbs.append(l)
    for key in Counter(lbs).most_common(max):
        top.append(key[0])
    for ix, l in enumerate(labels):
        if l in top:
            i_to_keep.append(ix)
            l_to_keep.append(l)
    for ix, e in enumerate(embeddings):
        if ix in i_to_keep:
            e_to_keep.append(e)
    return l_to_keep,e_to_keep
#print(len(i_to_keep), len(e_to_keep), len(l_to_keep))

def insert_lang(lang,labs):
    to_return = []
    for l in labs:
       # print(l)
        final=lang+"-"+l
        to_return.append(final)
    return to_return

def filt(embeddings, labels,ids,regs,lang):
    i_to_keep = []
    l_to_keep = []
    e_to_keep=[]
    r_to_keep=regs.split("-")
    ids_to_keep = []
    lbs = []
    for ix, l in enumerate(labels):
       # print("LL",l)
        if l in r_to_keep:
        #    print("L",l)
            i_to_keep.append(ix)
            l_to_keep.append(lang+"_"+l)
            e_to_keep.append(embeddings[ix])
            ids_to_keep.append(ids[ix])
    return l_to_keep,e_to_keep,ids_to_keep

