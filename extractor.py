import fitz 
from PIL import Image
import io
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse


#TODO: Reduce output file size
#TODO: Add error handling

app = FastAPI()

def change_signature_color(pdf_data):
    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
    processed_first_page_img = None  # Variable to store the final image of the first page

    # --- Process only the first page --- 
    if len(pdf_document) > 0:
        page_num = 0
        page = pdf_document.load_page(page_num)
        # Render page at higher DPI for quality
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) 
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Process the first page for signature
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 21, 4) 
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = gray.shape
        bottom_half = height // 2
        max_area = 0
        signature_contour = None
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if y > bottom_half:  
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    signature_contour = contour
        
        if signature_contour is not None:
            # Found signature, process it
            x, y, w, h = cv2.boundingRect(signature_contour)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [signature_contour], -1, 255, -1)
            
            signature_area = gray[y:y+h, x:x+w]
            # Use Otsu's method for potentially better thresholding
            _, dark_pixels = cv2.threshold(signature_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  
            
            modified_img = img_array.copy()
            
            for i in range(h):
                for j in range(w):
                    if dark_pixels[i, j] > 0:  
                        modified_img[y+i, x+j] = [0, 0, 255] # Blue color
            
            processed_first_page_img = Image.fromarray(modified_img)
        else:
            # No signature found on first page, use original image
            processed_first_page_img = img
    else:
        # Handle empty PDF case (optional)
        return fitz.open() # Return empty PDF

    # --- Create output PDF with only the first page --- 
    if processed_first_page_img is None:
         # Handle case where first page somehow wasn't processed (shouldn't happen with current logic)
        return fitz.open()
        
    output_pdf = fitz.open()
    
    # Convert the processed first page image to bytes (using PNG for better quality)
    img_byte_arr = io.BytesIO()
    processed_first_page_img.save(img_byte_arr, format='PNG') 
    img_byte_arr = img_byte_arr.getvalue()
    
    # Open the image bytes as a fitz document (correct filetype)
    img_doc = fitz.open(stream=img_byte_arr, filetype="png") 
    rect = img_doc[0].rect
    
    # Create a new page in the output PDF with the dimensions of the image
    pdf_page = output_pdf.new_page(width=rect.width, height=rect.height)
    
    # Insert the image onto the new page
    pdf_page.insert_image(
        rect,
        stream=img_byte_arr,
    )
    img_doc.close() # Close the temporary image document

    return output_pdf

@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    
    output_pdf = change_signature_color(contents)
    
    output_buffer = io.BytesIO()
    # Add PDF compression options
    output_pdf.save(
        output_buffer,
        garbage=4,  # Maximum garbage collection
        deflate=True,  # Use deflate compression
        clean=True  # Remove unused elements
    )
    output_pdf.close()
    
    output_buffer.seek(0)
    
    return StreamingResponse(
        output_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=modified_document.pdf"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)