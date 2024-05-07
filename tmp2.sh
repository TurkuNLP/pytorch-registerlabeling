python3 train.py -c src/configs/xlmr-large/labels_all/fi.yaml --predict_labels upper --just_evaluate --test fi -s 42 --cachedir test_cache
python3 train.py -c src/configs/xlmr-large/labels_all/fi.yaml --predict_labels upper --just_evaluate --test fi -s 43 --cachedir test_cache
python3 train.py -c src/configs/xlmr-large/labels_all/fi.yaml --predict_labels upper --just_evaluate --test fi -s 44 --cachedir test_cache

python3 train.py -c src/configs/xlmr-large/labels_all/fr.yaml --predict_labels upper --just_evaluate --test fr -s 42 --cachedir test_cache
python3 train.py -c src/configs/xlmr-large/labels_all/fr.yaml --predict_labels upper --just_evaluate --test fr -s 43 --cachedir test_cache
python3 train.py -c src/configs/xlmr-large/labels_all/fr.yaml --predict_labels upper --just_evaluate --test fr -s 44 --cachedir test_cache

python3 train.py -c src/configs/xlmr-large/labels_all/sv.yaml --predict_labels upper --just_evaluate --test sv -s 42 --cachedir test_cache
python3 train.py -c src/configs/xlmr-large/labels_all/sv.yaml --predict_labels upper --just_evaluate --test sv -s 43 --cachedir test_cache
python3 train.py -c src/configs/xlmr-large/labels_all/sv.yaml --predict_labels upper --just_evaluate --test sv -s 44 --cachedir test_cache

python3 train.py -c src/configs/xlmr-large/labels_all/tr.yaml --predict_labels upper --just_evaluate --test tr -s 42 --cachedir test_cache
python3 train.py -c src/configs/xlmr-large/labels_all/tr.yaml --predict_labels upper --just_evaluate --test tr -s 43 --cachedir test_cache
python3 train.py -c src/configs/xlmr-large/labels_all/tr.yaml --predict_labels upper --just_evaluate --test tr -s 44 --cachedir test_cache
