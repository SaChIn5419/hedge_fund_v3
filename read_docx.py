
import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def read_docx(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        with zipfile.ZipFile(file_path) as document:
            xml_content = document.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # XML namespace for Word
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text = []
            for node in tree.iter():
                if node.tag.endswith('}p'): # paragraphs
                    para_text = ""
                    for child in node.iter():
                         if child.tag.endswith('}t'): # text
                             if child.text:
                                 para_text += child.text
                    if para_text:
                        text.append(para_text)
            
            print("\n".join(text))
    except Exception as e:
        print(f"Error reading docx: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_docx(sys.argv[1])
    else:
        print("Usage: python read_docx.py <path_to_docx>")
