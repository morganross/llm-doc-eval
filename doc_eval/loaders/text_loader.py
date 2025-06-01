import os

def load_documents_from_folder(folder_path):
    """
    Scans a target folder for .txt and .md files and yields their content.

    Args:
        folder_path (str): The path to the folder containing documents.

    Yields:
        tuple: A tuple containing (doc_id, text_content, file_path).
               doc_id is the filename (without extension).
               text_content is the content of the file.
               file_path is the full path to the file.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(folder_path, filename)
            doc_id = os.path.splitext(filename)[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            yield doc_id, text_content, file_path