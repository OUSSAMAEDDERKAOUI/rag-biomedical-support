from unstructured.partition.pdf import partition_pdf
from app.rag.vision import describe_image

def load_pdf(path):

    elements = partition_pdf(
        filename=path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True
    )

    docs = []

    for el in elements:

        text = el.text or ""
        category = el.category

        if category == "Table":
            text = f"TABLE CONTENT:\n{text}"

        if category == "Image":
            image_path = el.metadata.image_path

            if image_path:
                description = describe_image(image_path)
                text = f"IMAGE DESCRIPTION:\n{description}"

        docs.append({
            "text": text,
            "metadata": {
                "type": category,
                "page": el.metadata.page_number
            }
        })

    return docs
