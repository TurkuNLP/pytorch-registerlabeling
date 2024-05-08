python3 train.py -c src/configs/bge-m3-2048/labels_all/fi.yaml --predict_labels upper --just_evaluate --test fi -s 42 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/fi.yaml --predict_labels all   --just_evaluate --test fi -s 42 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/fi.yaml --predict_labels upper --just_evaluate --test fi -s 43 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/fi.yaml --predict_labels all   --just_evaluate --test fi -s 43 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/fi.yaml --predict_labels upper --just_evaluate --test fi -s 44 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/fi.yaml --predict_labels all   --just_evaluate --test fi -s 44 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/en.yaml --predict_labels upper --just_evaluate --test en -s 42 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/en.yaml --predict_labels all   --just_evaluate --test en -s 42 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/en.yaml --predict_labels upper --just_evaluate --test en -s 43 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/en.yaml --predict_labels all   --just_evaluate --test en -s 43 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/en.yaml --predict_labels upper --just_evaluate --test en -s 44 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/en.yaml --predict_labels all   --just_evaluate --test en -s 44 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/fr.yaml --predict_labels upper --just_evaluate --test fr -s 42 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/fr.yaml --predict_labels all   --just_evaluate --test fr -s 42 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/fr.yaml --predict_labels upper --just_evaluate --test fr -s 43 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/fr.yaml --predict_labels all   --just_evaluate --test fr -s 43 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/fr.yaml --predict_labels upper --just_evaluate --test fr -s 44 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/fr.yaml --predict_labels all   --just_evaluate --test fr -s 44 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/sv.yaml --predict_labels upper --just_evaluate --test sv -s 42 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/sv.yaml --predict_labels all   --just_evaluate --test sv -s 42 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/sv.yaml --predict_labels upper --just_evaluate --test sv -s 43 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/sv.yaml --predict_labels all   --just_evaluate --test sv -s 43 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/sv.yaml --predict_labels upper --just_evaluate --test sv -s 44 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/sv.yaml --predict_labels all   --just_evaluate --test sv -s 44 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/tr.yaml --predict_labels upper --just_evaluate --test tr -s 42 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/tr.yaml --predict_labels all   --just_evaluate --test tr -s 42 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/tr.yaml --predict_labels upper --just_evaluate --test tr -s 43 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/tr.yaml --predict_labels all   --just_evaluate --test tr -s 43 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/tr.yaml --predict_labels upper --just_evaluate --test tr -s 44 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/tr.yaml --predict_labels all   --just_evaluate --test tr -s 44 --cachedir test_cache




python3 train.py -c src/configs/bge-m3-2048/labels_all/fi-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test en -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/fi-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test en -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/fi-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test en -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test fi -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test fi -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test fi -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test fr -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test fr -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-sv-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test fr -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test sv -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test sv -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-tr.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test sv -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test tr -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test tr -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv.yaml --labels all --just_evaluate --predict_labels all --cachedir test_cache --test tr -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/fi-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test en -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/fi-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test en -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/fi-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test en -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test fi -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test fi -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fr-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test fi -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test fr -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test fr -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-sv-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test fr -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test sv -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test sv -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-tr.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test sv -s 44

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test tr -s 42
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test tr -s 43
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv.yaml --labels all --just_evaluate --predict_labels upper --cachedir test_cache --test tr -s 44


python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv-tr.yaml --predict_labels upper --just_evaluate --test en-fi-fr-sv-tr -s 42 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv-tr.yaml --predict_labels upper --just_evaluate --test en-fi-fr-sv-tr -s 43 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv-tr.yaml --predict_labels upper --just_evaluate --test en-fi-fr-sv-tr -s 44 --cachedir test_cache

python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv-tr.yaml --predict_labels all --just_evaluate --test en-fi-fr-sv-tr -s 42 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv-tr.yaml --predict_labels all --just_evaluate --test en-fi-fr-sv-tr -s 43 --cachedir test_cache
python3 train.py -c src/configs/bge-m3-2048/labels_all/en-fi-fr-sv-tr.yaml --predict_labels all --just_evaluate --test en-fi-fr-sv-tr -s 44 --cachedir test_cache
