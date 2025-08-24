import pytesseract
from PIL import Image
from tkinter import Tk, filedialog

# Path to Tesseract engine (Windows only, skip for Linux/Mac)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Hide the main Tkinter window
Tk().withdraw()

# Open file explorer for user to select an image
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if image_path:  # if user selected a file
    # Open the image
    image = Image.open(image_path)

    # OCR with English + Telugu
    text = pytesseract.image_to_string(image, lang="eng+tel")

    print("\nExtracted Text:\n")
    print(text)
else:
    print("No file selected.")

