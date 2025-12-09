import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import os

# Initialize Engines
# GPU=False for compatibility, change to True if available
reader = easyocr.Reader(['en'], gpu=False) 
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def preprocess_image(image_path):
    """
    Reads and pre-processes the image for OCR.
    Steps: Grayscale -> Denoise -> Threshold -> Deskew (Optional)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive Thresholding to handle shadows/uneven lighting
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    return img, thresh

def extract_text(image):
    """
    Extracts text using EasyOCR.
    Returns full text and detailed results (bbox, text, conf).
    """
    results = reader.readtext(image)
    full_text = " ".join([res[1] for res in results])
    return full_text, results

def analyze_pii(text):
    """
    Detects PII in the text using Microsoft Presidio.
    """
    results = analyzer.analyze(text=text,
                               entities=["PERSON", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS"],
                               language='en')
    return results

def simple_redact_text(text, pii_results):
    """
    Redacts PII from the text string.
    """
    result = anonymizer.anonymize(text=text, analyzer_results=pii_results)
    return result.text

def redact_image_visual(image, ocr_results, pii_results):
    """
    Redacts PII from the image based on OCR bounding boxes.
    """
    img_copy = image.copy()
    
    # Extract detected PII values
    pii_texts = [res.entity_type for res in pii_results] # checking types not values? No, cannot access value directly in analyzer result often, need to extract it from text.
    # Actually, Presidio Result has (start, end, score, entity_type). 
    # We need to extract the substring from the original text.
    
    # But wait, ocr_results has the text segments.
    # Let's simple search: if an OCR segment is found in the PII ranges, redact it.
    
    # We need the full text to map ranges.
    # Reconstruct full text logic was: full_text = " ".join([res[1] for res in results])
    # This introduces spaces. Presidio indices are based on that joined string.
    
    current_idx = 0
    full_text = ""
    ocr_segments_with_indices = []
    
    for bbox, text, conf in ocr_results:
        start = current_idx
        end = current_idx + len(text)
        ocr_segments_with_indices.append( (start, end, bbox, text) )
        full_text += text + " "
        current_idx = end + 1

    # Loop through PII results
    for pii in pii_results:
        pii_start = pii.start
        pii_end = pii.end
        
        # Find intersecting OCR segments
        for seg_start, seg_end, bbox, seg_text in ocr_segments_with_indices:
            # Check overlap
            if max(pii_start, seg_start) < min(pii_end, seg_end):
                # Redact this bbox
                # bbox is list of 4 points [[x,y], [x,y], [x,y], [x,y]]
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(img_copy, top_left, bottom_right, (0, 0, 0), -1)
    
    return img_copy

def main():
    if not os.path.exists('output'):
        os.makedirs('output')
        
    image_files = [f for f in os.listdir('.') if f.startswith('uploaded_image') and f.endswith('.jpg')]
    
    for img_file in image_files:
        print(f"Processing {img_file}...")
        try:
            original, processed = preprocess_image(img_file)
            
            # Run OCR on processed image
            text, details = extract_text(processed) # Using processed for better OCR
            
            print(f"--- Extracted Text ({img_file}) ---")
            print(text[:200] + "...") 
            
            pii_results = analyze_pii(text)
            print(f"--- PII Detected ---")
            for res in pii_results:
                print(res)
                
            redacted_text = simple_redact_text(text, pii_results)
            print(f"--- Redacted Text ---")
            print(redacted_text[:200] + "...")
            
            # Visual Redaction
            redacted_img = redact_image_visual(original, details, pii_results)
            
            output_path = os.path.join('output', f"redacted_{img_file}")
            cv2.imwrite(output_path, redacted_img)
            print(f"Saved redacted image to {output_path}")

            # Generate Comparison Screenshot
            plt.figure(figsize=(15, 10))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Redacted (PII Hidden)")
            plt.imshow(cv2.cvtColor(redacted_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            comparison_path = os.path.join('output', f"comparison_{img_file}")
            plt.savefig(comparison_path)
            plt.close()
            print(f"Saved comparison to {comparison_path}")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

if __name__ == "__main__":
    main()
