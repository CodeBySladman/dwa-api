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
    modified_images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        # Reduce DPI from 4 to 2 for better file size while maintaining quality
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        if page_num == 0:  # Only process first page for signature
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
                x, y, w, h = cv2.boundingRect(signature_contour)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [signature_contour], -1, 255, -1)
                
                signature_area = gray[y:y+h, x:x+w]
                _, dark_pixels = cv2.threshold(signature_area, 180, 255, cv2.THRESH_BINARY_INV)  
                
                modified_img = img_array.copy()
                
                for i in range(h):
                    for j in range(w):
                        if dark_pixels[i, j] > 0:  
                            modified_img[y+i, x+j] = [0, 0, 255]  
                
                modified_img_pil = Image.fromarray(modified_img)
                modified_images.append(modified_img_pil)
            else:
                modified_images.append(img)
        else:
            modified_images.append(img)

    
    output_pdf = fitz.open()
    for img in modified_images:
    
        img_byte_arr = io.BytesIO()
    
        img.save(img_byte_arr, format='JPEG', quality=85, optimize=True) 
        img_byte_arr = img_byte_arr.getvalue()
        
        img_doc = fitz.open(stream=img_byte_arr, filetype="jpeg")
        rect = img_doc[0].rect
        pdf_page = output_pdf.new_page(width=rect.width, height=rect.height)
        
        pdf_page.insert_image(
            rect,
            stream=img_byte_arr,
            compression=fitz.PDF_FLATE_ENCODE  # Use Flate compression
        )
    
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