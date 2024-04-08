def instruct_prompt(data_point, labels=True):
    return f"""<s>[INST] You are a hierarchical multilabel text classifier tool. There are the following 9 upper level labels:

MACHINE TRANSLATED OR GENERATED: MT
LYRICAL: LY
SPOKEN: SP
INTERACTIVE DISCUSSION: ID
NARRATIVE: NA
HOW-TO or INSTRUCTIONS: HI
INFORMATIONAL DESCRIPTION: IN
OPINION: OP
INFORMATIONAL PERSUASION: IP

Some upper level labels additionally have lower labels. They are as follows:

SPOKEN: Interview: it
NARRATIVE: News report: ne
NARRATIVE: Sports report: sr
NARRATIVE: Narrative blog nb
HOW-TO or INSTRUCTIONS: Recipe: re
INFORMATIONAL DESCRIPTION: Encyclopedia article: en
INFORMATIONAL DESCRIPTION: Research article: ra
INFORMATIONAL DESCRIPTION: Description of a thing or person: dtp
INFORMATIONAL DESCRIPTION: Faq: fi
INFORMATIONAL DESCRIPTION: Legal terms and conditions: lt
OPINION: Review: rv
OPINION: Opinion blog: ob
OPINION: Denominational religious blog / sermon: rs
OPINION: Advice: av
INFORMATIONAL PERSUASION: Description with intent to sell: ds
INFORMATIONAL PERSUASION: News & opinion blog or editorial: ed

Your task is classify the target text (enclosed within ```) into one or more labels. If you classify a text into a lower level label, its corresponding upper level label must always also be included. Your output should consist of space-separated label names, alphabetically sorted, all in a single line.

Here is the target text:
```{data_point["text"][:3000]}```
[/INST]
{data_point["label_text"] if labels else ""}
"""
