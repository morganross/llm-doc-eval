import os

def load_documents_from_folder(folder_path):
    """
    Scans a target folder for .txt and .md files and yields their content.

    Args:
        folder_path (str): The path to the folder containing documents.

    Yields:
        tuple: A tuple containing (doc_id, text_content).
               doc_id is the filename (without extension).
               text_content is the content of the file.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(folder_path, filename)
            doc_id = os.path.splitext(filename)[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            yield doc_id, text_content

if __name__ == '__main__':
    # Example usage:
    # Create a dummy folder and files for testing
    if not os.path.exists("test_docs"):
        os.makedirs("test_docs")
    with open("test_docs/doc1.txt", "w", encoding="utf-8") as f:
        f.write("This is the content of document 1.")
    with open("test_docs/doc2.md", "w", encoding="utf-8") as f:
        f.write("# Document 2\n\nThis is the content of document 2 in markdown.")
    with open("test_docs/image.jpg", "w", encoding="utf-8") as f:
        f.write("This is not a text file.")

    print("Loading documents from 'test_docs':")
    for doc_id, content in load_documents_from_folder("test_docs"):
        print(f"Doc ID: {doc_id}\nContent:\n{content}\n---")

    # Clean up dummy files
    os.remove("test_docs/doc1.txt")
    os.remove("test_docs/doc2.md")
    os.remove("test_docs/image.jpg")
    os.rmdir("test_docs")