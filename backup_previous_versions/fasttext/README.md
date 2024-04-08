
## Fasttext models

Trained using 18466 documents from English, Finnish, French and Swedish CORE corpora, following the instructions at https://fasttext.cc/docs/en/supervised-tutorial.html.

Precision 0.704, recall 0.492, threshold 0.5. The predictions are better for longer documents, and precision was considered more important than recall during training.

The task is a multilabel task over 26 labels described here: https://turkunlp.org/register-annotation-docs/abbreviations.

The model is available at http://dl.turkunlp.org/register-labeling-model/fasttext_model.bin.

The model is not yet described in any research paper, but if you use it, please cite this page and this paper:

Samuel Rönnqvist, Valtteri Skantsi, Miika Oinonen, and Veronika Laippala. 2021. Multilingual and Zero-Shot is Closing in on Monolingual Web Register Classification. In Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa), pages 157–165, Reykjavik, Iceland (Online). Linköping University Electronic Press, Sweden.
