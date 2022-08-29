# torch-transformers-multilabel

This codebase was built with PyTorch 1.11, Transformers 4.18.0 and Hugging Face datasets. 

See performance scores [here](https://docs.google.com/spreadsheets/d/1lzmULFhy9DzfQkxqB3hzAmcvplS3sgFLjFgEys4-6lE/edit?usp=sharing).

Existing models: puhti.csc.fi:/scratch/project_2002026/anna/extended_registerlabeling/models and biarc.utu.fi:/mnt/biarc_taltio/pytorch-registerlabeling/models

## Quickstart

To train a model, run

    python3 train.py --train [TRAIN_LANGUAGE] --dev [TRAIN_LANGUAGE] --test [TEST_LANGUAGE]
    
Use several languages in training or testing by connecting them with a hyphen. E.g., en-fi-sv.
  
With the slurm script, you can launch multiple instances:

    sbatch slurm_train.sh [TRAIN_LANGUAGE] [TEST_LANGUAGE] [LRs] [EPOCHSs] [INSTANCEs]
    
To predict labels for a text file (several texts when separated by linebreak):
    
    python3 predict.py --text text.txt --load_model model.pt
    
## Data

English (full), French and Swedish COREs: https://github.com/TurkuNLP/multilingual-register-data

FinCORE: https://github.com/TurkuNLP/FinCORE_full

**The following have been split (stratified) into 50/50 train and test (included in this repo)**:

Arabian, Catala, Persian, Hindi, Indonesian, Norwegian, Portuguese, Urdu: https://github.com/TurkuNLP/Massively-multilingual-CORE

Japanese: https://github.com/TurkuNLP/JpnCORE

Russian: https://github.com/TurkuNLP/RuCORE

Simplified Chinese: https://github.com/TurkuNLP/SimChiCORE

Spanish: https://github.com/TurkuNLP/SpaCORE

Turkish: https://github.com/TurkuNLP/TurCORE

data/multi/ contains all of the smaller COREs combined for training and testing (not English, Finnish, French or Swedish).

**List of registers and their abbreviations (used as labels)**:

| Register                           | Label         |
|------------------------------------|--------------|
| How-to                             | HI           |
| Interactive Discussion             | ID           |
| Informative                        | IN           |
| Informational Persuasion           | IP           |
| Lyrical                            | LY           |
| Narrative                          | NA           |
| Opinion                            | OP           |
| Spoken                             | SP           |
| Advice                             | av           |
| Description with intent to sell    | ds           |
| Description of a thing or a person | dtp          |
| News & opinion blog or editorial   | ed           |
| Encyclopedia article               | en           |
| FAQ                                | fi           |
| Interview                          | it           |
| Legal terms and conditions         | lt           |
| Narrative blog                     | nb           |
| News report                        | ne           |
| Opinion blog                       | ob           |
| Research article                   | ra           |
| Recipe                             | re           |
| Denominational religious blog / sermon | rs           |
| Review                             | rv           |
| Sports report                      | sr           |
