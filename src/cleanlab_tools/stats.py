import pandas as pd

pd.set_option("display.max_rows", 100)  # Display up to 100 rows at a time
pd.set_option("display.max_columns", None)  # Display all columns
pd.set_option(
    "display.max_colwidth", 150
)  # Increase the maximum width of the 'text' column to 200 characters
pd.set_option(
    "display.width", None
)  # Automatically adjust the display width to accommodate the data

import src.labels as labels

# Paths to the files
path_multi = "data/multi_cleaned/multi.tsv"
path_multi_bad = "data/multi_cleaned/multi_bad.tsv"

# Read the files into pandas dataframes
df_multi = pd.read_csv(
    path_multi,
    sep="\t",
    header=None,
    names=["label", "text", "language"],
    na_values="",  # Don't interpret NA as NaN!
    keep_default_na=False,
)

missing_languages = df_multi["language"].isnull().sum()
print("Number of rows with missing 'language':", missing_languages)

df_multi_bad = pd.read_csv(
    path_multi_bad,
    sep="\t",
    header=None,
    names=["label", "text", "language", "badness_order"],
    na_values="",  # Don't interpret NA as NaN!
    keep_default_na=False,
)

# Calculate the number of rows grouped by language for each dataframe
language_counts_multi = df_multi.groupby("language").size()
language_counts_multi_bad = df_multi_bad.groupby("language").size()

# Print the stats
print("Counts for multi.csv grouped by language:")
print(language_counts_multi)
print("\nCounts for multi_bad.csv grouped by language:")
print(language_counts_multi_bad)

sum_grouped_counts = language_counts_multi.sum()
print("Sum of grouped language counts:", sum_grouped_counts)

# Combine the language data
combined_counts = pd.concat(
    [language_counts_multi, language_counts_multi_bad],
    axis=1,
    keys=["multi", "multi_bad"],
)

# Fill missing values with 0 (in case some languages are not present in both dataframes)
combined_counts.fillna(0, inplace=True)

# Calculate the percentage of lines
combined_counts["percentage_multi"] = (
    combined_counts["multi"]
    / (combined_counts["multi"] + combined_counts["multi_bad"])
    * 100
)
combined_counts["percentage_multi_bad"] = (
    combined_counts["multi_bad"]
    / (combined_counts["multi"] + combined_counts["multi_bad"])
    * 100
)

# Print combined stats and percentages
print("\nCombined Language Data:")
print(combined_counts)


def process_label(label):
    print(label)
    # Example processing (define the actual processing you need here)
    return labels.normalize_labels(label, "all")


df_multi_bad["label"] = df_multi_bad["label"].apply(process_label)


# Define a function to get the top 10 rows ordered by 'badness_order'
def top_10_sorted(group):
    return group.sort_values(by="badness_order").head(100)


# Apply the function to each language group in 'df_multi_bad'
top_10_per_language = df_multi_bad.groupby("language").apply(top_10_sorted)

top_10_per_language["text"] = top_10_per_language["text"].apply(
    lambda x: (x[:497] + "...") if len(x) >= 500 else x
)

# Reset the index if necessary (groupby + apply often results in multi-index)
top_10_per_language.reset_index(drop=True, inplace=True)


# Convert DataFrame to a tab-separated string
tsv_output = top_10_per_language[["language", "label", "text"]].to_csv(
    sep="\t", index=False, header=False
)

# Print the TSV output
print(tsv_output)

print()
print()

top_100_rows = df_multi_bad.sort_values(by="badness_order").head(100)


# Apply the processing function to the 'label' column

top_100_rows["text"] = top_100_rows["text"].apply(
    lambda x: (x[:497] + "...") if len(x) >= 500 else x
)

# print(top_100_rows[["language", "label", "text"]].to_string(index=False))


# Convert DataFrame to a tab-separated string
tsv_output = top_100_rows[["language", "label", "text"]].to_csv(
    sep="\t", index=False, header=False
)

# Print the TSV output
print(tsv_output)
