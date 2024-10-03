import json

def process_multiline_string(multiline_str):
    """
    Processes a multiline string by splitting it into a list of non-empty, trimmed strings.

    Parameters:
    - multiline_str (str): The input multiline string.

    Returns:
    - list of str: A list containing non-empty, trimmed lines from the input string.
    """
    # Split the string into lines based on the newline character
    lines = multiline_str.split('\n')

    # Use list comprehension to strip whitespace and exclude empty lines
    processed_lines = [line.strip() for line in lines if line.strip()]

    return processed_lines

def extract_documents_based_on_distance(rag_object, high_threshold=0.7, lower_threshold=0.65, low_threshold=0.6):
    """
    Extracts documents from a RAG object based on distance thresholds.

    Parameters:
    - rag_object (dict): The RAG object containing keys like 'distances' and 'documents'.
    - high_threshold (float): The primary distance threshold (default is 0.7).
    - lower_threshold (float): The secondary distance threshold (default is 0.65).
    - low_threshold (float): The fallback distance threshold if no distances meet the high_threshold (default is 0.6).

    Returns:
    - list of str: A list of documents that meet the distance criteria.
    NOTE: Will assume number of docs in rag call n_results is set to 1.
    """
    # Validate that 'distances' and 'documents' exist in the RAG object
    if 'distances' not in rag_object or 'documents' not in rag_object:
        raise KeyError("The RAG object must contain 'distances' and 'documents' keys.")

    def flatten_comprehension(matrix):
      return [item for row in matrix for item in row]

    # Flatten list of lists
    distances = flatten_comprehension(rag_object['distances'])
    documents = flatten_comprehension(rag_object['documents'])


    def filter_docs(threshold):
        return [doc for doc, dist in zip(documents, distances) if dist >= threshold]

    # Check if any distance meets the high_threshold
    if any(dist >= high_threshold for dist in distances):
        selected_docs = filter_docs(high_threshold)
        print(f"Selected documents with distances >= {high_threshold}: {len(selected_docs)} found.")
    elif any(dist >= lower_threshold for dist in distances):
        selected_docs = filter_docs(lower_threshold)
        print(f"No distances >= {high_threshold}. Selected documents with distances >= {lower_threshold}: {len(selected_docs)} found.")
    else:
      selected_docs = filter_docs(low_threshold)
      print(f"No distances >= {lower_threshold}. Selected documents with distances >= {low_threshold}: {len(selected_docs)} found.")

    # Always return at least the document with the best distance
    if not selected_docs:
        max_index = distances.index(max(distances))
        selected_docs = [documents[max_index]]

    return selected_docs


def make_json_objects(documents):
  json_docs = []
  for doc in documents:
    json_doc = json.loads(doc)
    json_docs.append(json_doc)
  return json_docs

def filter_unique_parent_codes(json_docs):
    """
    Filters the input list of JSON objects, retaining only the first occurrence
    of each unique parent_code.

    Parameters:
    - json_docs (list of dict): The list of JSON objects to filter.

    Returns:
    - list of dict: A new list containing only the first occurrence of each parent_code.
    """
    seen_parent_codes = set()
    filtered_docs = []

    for doc in json_docs:
        parent_code = doc.get('parent_code')
        if parent_code not in seen_parent_codes:
            seen_parent_codes.add(parent_code)
            filtered_docs.append(doc)
    return filtered_docs