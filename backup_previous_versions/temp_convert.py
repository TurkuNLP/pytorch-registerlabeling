import re

from v3 import labels

label_set = labels.labels_all


def process_documents(input_filename, output_filename):
    # Read the entire text file
    with open(input_filename, "r", encoding="utf-8") as file:
        content = file.read()

    # Split the file by empty newlines into documents
    documents = content.split("\n###C:DOCSTART")
    errors = 0
    error_labels = []
    no_label = 0

    # Prepare to write to the TSV file
    with open(output_filename, "w", encoding="utf-8") as outfile:
        print(len(documents[1:]))
        for document in documents[1:]:

            # Initialize variables
            labels = ""
            text = ""

            match = re.search(r"###C:tags=(.*)", document)

            if match:
                # Extract the content after the matching pattern
                label_list = match.group(1).strip().split(" ")
                label_list = [x for x in label_list if x]

                if label_list:

                    if not all([x in label_set for x in label_list]):
                        errors += 1
                        error_labels.append(" ".join(label_list))

                    else:
                        # print("good")
                        labels = " ".join(label_list)
                        # print(labels)
                        clean_doc = re.sub(
                            r"^###C:.*\n?", "", document, flags=re.MULTILINE
                        ).strip()
                        if clean_doc:
                            clean_doc = clean_doc.replace("\t", " ")
                            clean_doc = clean_doc.replace("\n", "\\n")
                            outfile.write(f"{labels}\t{clean_doc}\n")

                else:
                    no_label += 1
            else:
                print("Pattern not found.")
                print(document)

            """
            # Split document into lines for processing
            lines = document.split("\n")
            for line in lines:
                # Find the line starting with ###C:tags= to extract labels
                if line.startswith("###C:tags="):
                    labels = line.split("=", 1)[1]
                # Remove lines starting with ###C:
                elif not line.startswith("###C:"):
                    text += line + " "  # Add a space for readability

            # Remove the trailing space from text
            text = text.rstrip()

            # Save the "labels" and "text" to the TSV file
            outfile.write(f"{labels}\t{text}\n")
            """
    print(errors)
    print(error_labels)
    print(no_label)


# Define the input and output filenames
input_filename = "Register annotated web texts.txt"
output_filename = "tr.tsv"

# Call the function to process the documents
process_documents(input_filename, output_filename)
