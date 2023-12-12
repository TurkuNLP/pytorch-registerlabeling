def prompt(data_point):
    return f"""
        You are a multilabel text classifier tool. The labels are hierarchical. There are 9 upper level labels. They are listed below using the following format (enclosed within ```):

        ```
        [UPPER LEVEL LABEL]: [ABBREVIATION]
        ```

        Here are the 9 upper level labels (enclosed within ```):

        ```
        MACHINE TRANSLATED OR GENERATED: MT
        LYRICAL: LY
        SPOKEN: SP
        INTERACTIVE DISCUSSION: ID
        NARRATIVE: NA
        HOW-TO or INSTRUCTIONS: HI
        INFORMATIONAL DESCRIPTION: IN
        OPINION: OP
        INFORMATIONAL PERSUASION: IP
        ```

        In addition, some labels have lower level labels. I will list them using the following format (enclosed within ```):

        ```
        [UPPER LEVEL LABEL]: [ABBREVIATION]
        ##########
        [Lower level label]: [abbreviation]

        ```

        Here are the upper level labels with their respective lower level labels (enclosed within ```)

        ```
        SPOKEN: SP
        ##########
        Interview: it
        Other spoken: os

        NARRATIVE: NA
        ##########
        News report: ne
        Sports report: sr
        Narrative blog nb

        HOW-TO or INSTRUCTIONS: HI
        ##########
        Recipe: re

        INFORMATIONAL DESCRIPTION: IN
        ##########
        Encyclopedia article: en
        Research article: ra
        Description of a thing or person: dtp
        Faq: fi
        Legal terms and conditions: lt

        OPINION: OP
        ##########
        Review: rv
        Opinion blog: ob
        Denominational religious blog / sermon: rs
        Advice: av

        INFORMATIONAL PERSUASION: IP
        ##########
        Description with intent to sell: ds
        News & opinion blog or editorial: ed

        ```

        Your task is classify the below text (given after ### Target text) to one or more labels. The correct labels are given  If you classify a text into a lower level label, its corresponding upper level label should always also be included. Your output should consist of space-separated label names, alphabetically sorted. 

        ### Target text
        {data_point["target"]}


        ### Labels
        {data_point["labels"]}
    """
