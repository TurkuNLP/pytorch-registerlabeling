#!/bin/bash

# List of values to loop through
values=(42 43 44)

BASE_CMD="python3 train.py -c src/configs/xlmr-large/labels_all/"

# Loop through each value
for value in "${values[@]}"; do
  echo "Running seed $value"
  $BASE_CMD"en.yaml" --just_evaluate --predict_labels upper -s "$value" --test en
  $BASE_CMD"fi.yaml" --just_evaluate --predict_labels upper -s "$value" --test fi
  $BASE_CMD"fr.yaml" --just_evaluate --predict_labels upper -s "$value" --test fr
  $BASE_CMD"sv.yaml" --just_evaluate --predict_labels upper -s "$value" --test sv
  $BASE_CMD"tr.yaml" --just_evaluate --predict_labels upper -s "$value" --test tr
  $BASE_CMD"en-fi-fr-sv-tr.yaml" --just_evaluate --predict_labels upper -s "$value"
  $BASE_CMD"en-fi-fr-sv.yaml" --just_evaluate --predict_labels upper -s "$value" --test tr
  $BASE_CMD"en-fi-fr-tr.yaml" --just_evaluate --predict_labels upper -s "$value" --test sv
  $BASE_CMD"en-fi-sv-tr.yaml" --just_evaluate --predict_labels upper -s "$value" --test fr
  $BASE_CMD"en-fr-sv-tr.yaml" --just_evaluate --predict_labels upper -s "$value" --test fi
  $BASE_CMD"fi-fr-sv-tr.yaml" --just_evaluate --predict_labels upper -s "$value" --test en
done

