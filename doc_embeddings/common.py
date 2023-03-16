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
