Ocr is a optical character recognizer 
it is the process of converting image of text (printed, typed, or handwritten) into a machine-readable text format 
ocr was used in tool called as tesseract 
these was free and open source engine which helps in extracting the characters
to use these firstly we have to install tesseract engine from github repo https://github.com/tesseract-ocr/tessdata/blob/main/tel.traineddata
then we have install in terminal  pip install pytesseract Pillow
it firstly convert the pdf into the images and then extract each character from it 
the main problem here is that it only works with clean and good data 
pillow was used to resize and open the image 

Basic text extraction from an image 

from PIL import Image
import pytesseract

# Open an image file
img = Image.open('example_image.png')

# Use Pytesseract to convert the image to text
text = pytesseract.image_to_string(img)

# Print the extracted text
print(text)


Extracting text from a PDF

import pytesseract
from pdf2image import convert_from_path

pdf_path = 'document.pdf'
pages = convert_from_path(pdf_path, 300) # 300 is the resolution (dpi)

extracted_text = ""
for page_number, page_image in enumerate(pages, start=1):
    text = pytesseract.image_to_string(page_image)
    extracted_text += f"--- Page {page_number} ---\n"
    extracted_text += text + "\n"

print(extracted_text)


Specifying languages

import pytesseract
from PIL import Image

image = Image.open('french_document.jpg')
text = pytesseract.image_to_string(image, lang='fra')
print(text)


tesseract language selection github repo https://github.com/tesseract-ocr/tessdata/blob/main/tel.traineddata

