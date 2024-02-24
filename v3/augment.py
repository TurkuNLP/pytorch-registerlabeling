from transformers import MarianMTModel, MarianTokenizer


class Augment:
    def __init__(self, cfg):
        self.cfg = cfg

        # Run
        getattr(self, cfg.method)()

    def get_model_and_tokenizer(self, model_name):
        target_tokenizer = MarianTokenizer.from_pretrained(model_name)
        target_model = MarianMTModel.from_pretrained(model_name)

        return target_tokenizer, target_model

    def back_translate(self):
        source_tokenizer, source_model = self.get_model_and_tokenizer(
            f"Helsinki-NLP/opus-mt-{self.cfg.source}-{self.cfg.target}"
        )
        target_tokenizer, target_model = self.get_model_and_tokenizer(
            f"Helsinki-NLP/opus-mt-{self.cfg.target}-{self.cfg.source}"
        )

        src_text = """
            Tailles de seaux supplémentaires pour le système de tri des déchets BLANCO SELECT Des solutions intelligentes pour la cuisine au quotidien Quiconque veille à produire le moins de déchets possible, a besoin d’une solution pratique pour séparer les matériaux recyclables des autres ordures ménagères. Dans ce cas, il est conseillé de monter un système de tri sélectif des déchets directement sous l’évier. Il permet de gagner du temps et d’économiser des allées et venues car c’est là que les restes de fruits et légumes, les emballages et autres déchets papiers sont jetés chaque jour. Blanco Select séduit d’ores et déjà par ses détails intelligents qui augmentent considérablement le confort dans la cuisine au quotidien. Sa conception bien pensée a d’ailleurs été récompensée par le prix de design international « red dot product design award 2013 ». Cinq modèles viennent désormais compléter le système de tri sélectif des déchets disponibles pour trois tailles de meubles sous évier. Les nouvelles variantes pour meuble sous évier de 50 cm et de 60 cm de large disposent de seaux supplémentaires respectivement de 6 litres et de 8 litres. Les seaux les plus petits conviennent idéalement à la collecte de déchets organiques et sont équipés d’un couvercle. Avec onze modèles au total et un vaste choix de tailles de seaux, Blanco Select apporte désormais des solutions optimales aux habitudes de tri les plus diverses. En outre, sur les nouvelles variantes, l’espace dans le meuble sous évier est utilisé au mieux, l’impasse ayant été faite sur les parois de séparation. C’est ainsi que Blanco Select XL 60/3 Orga propose, par exemple, un volume total non négligeable de 46 litres répartis entre trois seaux (de 30/8/8 litres) et Blanco Select 60/4 Orga à quatre seaux un volume total encore plus impressionnant de 42 litres (seaux de 15/15/6/6 litres). Avec des coloris et des formes modernes, Blanco Select s’intègre parfaitement à la cuisine d’aujourd’hui. Le bord biseauté de tous les seaux intégrant la poignée rabattable facilite la chute des ordures et évitent que les déchets et miettes restent accrochés. Le système de tri sélectif des déchets haut de gamme et très stable est équipé d’un plateau de recouvrement en métal. Ce dernier est très facile à nettoyer, comme tous les composants du système. Le système de guidage, moderne et résistant, garantit un fonctionnement souple et silencieux. Les quatre modèles équipés d’un tiroir combiné avec compartiments de rangement apportent un maximum de confort. Le tiroir contient, en outre, des boîtes universelles à usage variable pour ranger de manière ordonnée les ustensiles dans le meuble sous évier. Le tiroir avec compartiments de rangement peut d’ailleurs être monté ultérieurement sans problème, tout comme la réglette pratique pour ouverture de porte au pied Blanco Movex. Blanco Select est monté en un tour de main par un spécialiste sous n’importe quel évier Blanco. De plus amples informations sur l’ingénieux système de tri sélectif des déchets pour meuble sous évier de 45 cm, 50 cm et 60 cm de large sont disponibles dans les magasins de cuisines aménagées et meubles de cuisine et sur www.blanco-germany.com/select. Blanco Select séduit d’ores et déjà par des détails intelligents qui augmentent considérablement le confort dans la cuisine au quotidien. Cinq modèles viennent compléter désormais le système de tri sélectif des déchets haut de gamme. Les nouvelles variantes pour meuble sous évier de 50 cm et de 60 cm de large disposent de seaux supplémentaires respectivement de 6 litres et de 8 litres. Les seaux supplémentaires plus petits et pratiques conviennent idéalement à la collecte des déchets organiques. Le système de tri sélectif des déchets Blanco Select séduit par l’unité harmonieuse générée par le cadre, les seaux et le tiroir avec compartiments de rangement. Les nouveaux seaux plus petits s’intègrent parfaitement. Le bord biseauté des seaux évitent, comme d’habitude, que les déchets restent accrochés. Utilisation optimale de l’espace : le tiroir amovible avec compartiments de rangement et ses boîtes universelles pratiques garantissent ordre et propreté dans le meuble sous évier. Les cadres latéraux larges dissimulent élégamment les rails de guidage et contribuent à l’impression globale harmonieuse du système de tri des déchets haut de gamme. Vous êtes journaliste et souhaitez de plus amples informations ? Nous nous tenons volontiers à votre disposition. Manuel Koch BLANCO Suisse Rössliweg 48 4852 Rothrist Tel: 062 388 89 94 Fax: 062 388 89 98 E-Mail: manuel.koch@blanco.ch
        """

        translated = source_tokenizer.decode(
            source_model.generate(
                **source_tokenizer(
                    src_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
            )[0],
            skip_special_tokens=True,
        )
        print(translated)
        back_translated = target_tokenizer.decode(
            target_model.generate(
                **source_tokenizer(
                    translated,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
            )[0],
            skip_special_tokens=True,
        )

        print(back_translated)
