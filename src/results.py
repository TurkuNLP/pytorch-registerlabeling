
import json
import matplotlib.colors as mcolors
from sklearn.metrics import average_precision_score, f1_score

import numpy as np
import seaborn as sns

from .labels import binarize_labels, label_schemes, normalize_labels
from .data import small_languages

s = lambda x: "\\tiny{" + f"({x:.2f})" + "}"

palette = sns.color_palette("Blues", n_colors=100)

main_languages = ["en", "fi", "fr", "sv", "tr"]
small_languages = [
    "ar",
    "ca",
    "es",
    "fa",
    "hi",
    "id",
    "jp",
    "no",
    "pt",
    "ur",
    "zh",
]

multi_str = "en-fi-fr-sv-tr"

experiments = {
    "mono": [
        ("xlm-roberta-large", "XLMR-L"),
        ("BAAI/bge-m3-retromae_512", "BGE-M3"),
        ("BAAI/bge-m3-retromae_2048","BGE-M3$^{\\text{2048}}$"),
    ],
    "multi": [
        ("xlm-roberta-large", "XLMR-L"),
        ("BAAI/bge-m3-retromae_512", "BGE-M3"),
        ("BAAI/bge-m3-retromae_2048","BGE-M3$^{\\text{2048}}$"),
        ("intfloat/multilingual-e5-large", "ME5-L"),
        ("facebook/xlm-roberta-xl", "XLMR-XL"),
        #("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B")
    ],
    "zero": [
        ("xlm-roberta-large", "XLMR-L"),
        ("BAAI/bge-m3-retromae_512", "BGE-M3"),
        ("BAAI/bge-m3-retromae_2048","BGE-M3$^{\\text{2048}}$"),
    ],
}

hybrids = ['any_single', 'any_hybrid', 'single_any', 'single_single', 'single_hybrid', 'hybrid_any', 'hybrid_single', 'hybrid_hybrid']


seeds = [42, 43, 44]

import numpy as np

def rgb_to_hex(rgb):
    # Convert from [0, 1] to [0, 255]
    r, g, b = [int(255 * x) for x in rgb]
    return '{:02X}{:02X}{:02X}'.format(r, g, b)

def normalize(value, min_value, max_value, new_min=1, new_max=100):
    return new_min + ((value - min_value) * (new_max - new_min) / (max_value - min_value))

def colorize(value):
    # Clamp the value between min_val and max_val

    index = int(normalize(value, 50, 100))

    text_color = "\\textcolor{white}{" if index >= 69 else ""

    
    # Pick the color from the colormap
    hex_color = rgb_to_hex(palette[index])

    return f"\\cellcolor[HTML]{{{hex_color}}}{text_color}", "}" if text_color else ""
    
    return f"\\cellcolor[HTML]{{{hex_color[1:]}}}"

exp_map = {
    'mono': 'Monolingual',
    'multi': 'Multilingual',
    'zero': 'Zero-shot',
}

def exp_header(experiment_type, predict_labels):
    if predict_labels == "upper":
        return "\\\\"
    else:
        return "\\textbf{" + exp_map[experiment_type] + "} \\\\"

def make_learning_curve():
    result = {k: {i: [0, 0, 0] for i in range(100, 1100, 100)} for k in main_languages}
        

    for lang in main_languages:
        for i in range(1, 11):
            for k, seed in enumerate(seeds):
                file_path = f"predictions/xlm-roberta-large/{lang}_{lang}/seed_{seed}/subset_{i}/all_all_{lang}_metrics.json"
                try:
                    data = json.load(
                        open(
                            file_path,
                            "r",
                            encoding="utf-8",
                        )
                    )
            
                except:
                    print(f'No file: {file_path}')

                result[lang][i*100][k] = data[f"f1"] 

    print(result)

def make_hybrids():

    for hybrid in hybrids:
        print(f'\n\n ---{hybrid}--- \n\n')
        test_lang_data = {k: {'micro': [], 'macro': [], 'support': 0} for k in main_languages}
        for lang in main_languages:
         
            for seed in seeds:
                file_path = f"predictions/xlm-roberta-large/en-fi-fr-sv-tr_en-fi-fr-sv-tr/seed_{seed}/all_all_{lang}_{hybrid}_metrics.json"
                try:
                    data = json.load(
                        open(
                            file_path,
                            "r",
                            encoding="utf-8",
                        )
                    )
            
                except:
                    print(f'No file: {file_path}')

                for avg in ["micro", "macro"]:
                    test_lang_data[lang][avg].append(data[f"f1{'' if avg == 'micro' else '_macro'}"] * 100)
                if seed == 42:
                    support = np.sum([x['support'] for x in data['label_scores'].values()])
                    test_lang_data[lang]['support'] = support

        results = {}
        micro_means = []
        macro_means = []
        micro_stds = []
        macro_stds = []
        support = np.sum([x['support'] for x in test_lang_data.values()])   
        print(support / 28807 * 100)
        for key in test_lang_data.keys():
            
            micro_values = test_lang_data[key]['micro']
            macro_values = test_lang_data[key]['macro']

            assert(len(micro_values) == 3)
            assert(len(macro_values) == 3)
            
            micro_mean = np.mean(micro_values)
            micro_std = np.std(micro_values)
            
            macro_mean = np.mean(macro_values)
            macro_std = np.std(macro_values)
            
            results[key] = {
                'micro_mean': micro_mean,
                'micro_std': micro_std,
                'macro_mean': macro_mean,
                'macro_std': macro_std
            }
            
            micro_means.append(micro_mean)
            macro_means.append(macro_mean)
            micro_stds.append(micro_std)
            macro_stds.append(macro_std)

        # Calculate overall means
        overall_micro_mean = np.mean(micro_means)
        overall_macro_mean = np.mean(macro_means)

        # Calculate mean of standard deviations
        overall_micro_std_mean = np.mean(micro_stds)
        overall_macro_std_mean = np.mean(macro_stds)

        # LaTeX formatting with color
        # LaTeX formatting with color
        # LaTeX formatting with color
        micro_results = []
        macro_results = []

        for i, key in enumerate(list(results.keys()) + ["overall"]):
            if key == "overall":
                micro_mean = overall_micro_mean
                micro_std = overall_micro_std_mean
                macro_mean = overall_macro_mean
                macro_std = overall_macro_std_mean
                
                micro_color = colorize(micro_mean)
                macro_color = colorize(macro_mean)
                
                # Add a thin border around the overall mean using \fcolorbox
                micro_result = f"{micro_color[0]}{micro_mean:.0f} \\tiny{{({micro_std:.2f})}}{micro_color[1]}"
                macro_result = f"{macro_color[0]}{macro_mean:.0f} \\tiny{{({macro_std:.2f})}}{macro_color[1]}"
            else:
                micro_mean = results[key]['micro_mean']
                micro_std = results[key]['micro_std']
                macro_mean = results[key]['macro_mean']
                macro_std = results[key]['macro_std']
                
                micro_color = colorize(micro_mean)
                macro_color = colorize(macro_mean)
                
                micro_result = f"{micro_color[0]}{micro_mean:.0f} \\tiny{{({micro_std:.2f})}}{micro_color[1]}"
                macro_result = f"{macro_color[0]}{macro_mean:.0f} \\tiny{{({macro_std:.2f})}}{macro_color[1]}"

            micro_results.append(micro_result)
            macro_results.append(macro_result)

        # Join the results with '&'
        micro_prefix = f"\indd $\mu$ & "
        macro_prefix = "\indd $M$ & "
        micro_results_line = micro_prefix + (" & ".join(micro_results)) + ' \\\\'
        macro_results_line = macro_prefix + (" & ".join(macro_results)) + ' \\\\'

        # Print the LaTeX formatted results
        print(micro_results_line)
        print(macro_results_line)

def make_all():

    for lang_group in ["main", "small"]:

        print(f'\n\n ---{lang_group}--- \n\n')

        for predict_labels in ["all", "upper", "xgenre"]:

            print(f'\n\n --- --- ---{predict_labels}--- \n\n')

            for experiment_type in experiments:
                if lang_group == "small" and experiment_type in ["multi", "zero"]:
                    continue
                print('\\addlinespace[4pt]')
                print(exp_header(experiment_type, predict_labels))
                print('\\addlinespace[4pt]')
                for i, (model_path, model_name) in enumerate(experiments[experiment_type]):

                    last_model = (i+1) == len(experiments[experiment_type])
          

                    test_langs = main_languages if lang_group == "main" else small_languages
                    test_lang_data = {k: {'micro': [], 'macro': []} for k in test_langs}
                    for lang in test_langs:
                        if experiment_type == "mono":
                            if lang_group == "main":
                                lang_path = lang
                            else:
                                lang_path = multi_str
                        elif experiment_type == "multi":
                            lang_path = multi_str
                        elif experiment_type == "zero":
                            lang_path = "-".join([x for x in multi_str.split("-") if x != lang])


                        seed_data = []
                        result = ""

                        for seed in seeds:
                            file_path = f"predictions/{model_path}/{lang_path}_{lang_path}/seed_{seed}/all_{predict_labels}_{lang}_metrics.json"
                            try:
                                data = json.load(
                                    open(
                                        file_path,
                                        "r",
                                        encoding="utf-8",
                                    )
                                )
                                seed_data.append(data)
                            except:
                                print(f'No file: {file_path}')

               

                            for avg in ["micro", "macro"]:
                                test_lang_data[lang][avg].append(data[f"f1{'' if avg == 'micro' else '_macro'}"] * 100)
                    
                    results = {}
                    micro_means = []
                    macro_means = []
                    micro_stds = []
                    macro_stds = []

                    for key in test_lang_data.keys():
                        micro_values = test_lang_data[key]['micro']
                        macro_values = test_lang_data[key]['macro']

                        assert(len(micro_values) == 3)
                        assert(len(macro_values) == 3)
                        
                        micro_mean = np.mean(micro_values)
                        micro_std = np.std(micro_values)
                        
                        macro_mean = np.mean(macro_values)
                        macro_std = np.std(macro_values)
                        
                        results[key] = {
                            'micro_mean': micro_mean,
                            'micro_std': micro_std,
                            'macro_mean': macro_mean,
                            'macro_std': macro_std
                        }
                        
                        micro_means.append(micro_mean)
                        macro_means.append(macro_mean)
                        micro_stds.append(micro_std)
                        macro_stds.append(macro_std)

                    # Calculate overall means
                    overall_micro_mean = np.mean(micro_means)
                    overall_macro_mean = np.mean(macro_means)

                    # Calculate mean of standard deviations
                    overall_micro_std_mean = np.mean(micro_stds)
                    overall_macro_std_mean = np.mean(macro_stds)

                    # LaTeX formatting with color
                   # LaTeX formatting with color
                    # LaTeX formatting with color
                    micro_results = []
                    macro_results = []

                    for i, key in enumerate(list(results.keys()) + ["overall"]):
                        if key == "overall":
                            micro_mean = overall_micro_mean
                            micro_std = overall_micro_std_mean
                            macro_mean = overall_macro_mean
                            macro_std = overall_macro_std_mean
                            
                            micro_color = colorize(micro_mean)
                            macro_color = colorize(macro_mean)
                            
                            # Add a thin border around the overall mean using \fcolorbox
                            micro_result = f"{micro_color[0]}{micro_mean:.0f} \\tiny{{({micro_std:.2f})}}{micro_color[1]}"
                            macro_result = f"{macro_color[0]}{macro_mean:.0f} \\tiny{{({macro_std:.2f})}}{macro_color[1]}"
                        else:
                            micro_mean = results[key]['micro_mean']
                            micro_std = results[key]['micro_std']
                            macro_mean = results[key]['macro_mean']
                            macro_std = results[key]['macro_std']
                            
                            micro_color = colorize(micro_mean)
                            macro_color = colorize(macro_mean)
                            
                            micro_result = f"{micro_color[0]}{micro_mean:.0f} \\tiny{{({micro_std:.2f})}}{micro_color[1]}"
                            macro_result = f"{macro_color[0]}{macro_mean:.0f} \\tiny{{({macro_std:.2f})}}{macro_color[1]}"

                        micro_results.append(micro_result)
                        macro_results.append(macro_result)

                    # Join the results with '&'
                    micro_prefix = f"\\ind {model_name} & $\\mu$ &" if predict_labels is not "upper" else ""
                    macro_prefix = " &$M$ & " if predict_labels is not "upper" else ""
                    micro_results_line = micro_prefix + (" & ".join(micro_results)) + ' \\\\'
                    macro_results_line = macro_prefix + (" & ".join(macro_results)) + ' \\\\'

                    # Print the LaTeX formatted results
                    print(micro_results_line)
                    print(macro_results_line)
                    if not last_model:
                        print("\\addlinespace[2pt]")
                   




def run(cfg):

    if cfg.all:
        make_all()
        exit()

    if cfg.hybrids:
        make_hybrids()
        exit()

    if cfg.learning_curve:
        make_learning_curve()
        exit()

    average = "" if cfg.average == "micro" else "_macro"

    result_str = ""
    all_f1s = []
    all_f1_stds = []
    for lang in main_languages if cfg.target == "main" else small_languages:
        f1s = []
        for seed in seeds:
            if cfg.test == "mono":
                lang_path = lang
            elif cfg.test == "zero":
                lang_path = "-".join([x for x in multi_str.split("-") if x != lang])
            elif cfg.test == "multi":
                lang_path = multi_str
            data = json.load(
                open(
                    f"predictions/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/{lang_path}_{lang_path}/seed_{seed}/{cfg.labels}_{cfg.predict_labels}_{lang}_metrics.json",
                    "r",
                    encoding="utf-8",
                )
            )
            f1s.append(data[f"f1{average}"] * 100)

        f1s = np.array(f1s)
        mean = np.mean(f1s)
        std = np.std(f1s)
        all_f1s.append(mean)
        all_f1_stds.append(std)
        print(lang)
        print(f1s)

        result_str += f"& {np.mean(f1s):.0f} {s(np.std(f1s))} "
    result_str += f"& {np.mean(all_f1s):.0f} {s(np.mean(all_f1_stds))} "
    print(result_str)
