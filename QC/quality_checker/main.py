import cv2
import numpy as np
import os
import fitz  # PyMuPDF
from fpdf import FPDF
from sklearn.cluster import KMeans

def align_images(img1, img2):
    """Align img2 to img1 using Enhanced Correlation Coefficient (ECC)."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize for faster ECC
    h, w = gray1.shape
    gray2 = cv2.resize(gray2, (w, h))
    img2 = cv2.resize(img2, (w, h))

    # ECC requires float32
    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)

    # Initialize warp matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        5000,
        1e-6
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(gray1, gray2, warp_matrix, cv2.MOTION_AFFINE, criteria)
        aligned_img2 = cv2.warpAffine(img2, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_img2
    except cv2.error:
        print("⚠️ ECC alignment failed — skipping alignment.")
        return img2

def highlight_differences(img1, img2, threshold=25):
    """
    Highlights differences between two engineering drawings with red (img1) and green (img2),
    handling various sizes, alignments, gaps, or missing template borders.
    """
    # Step 1: Align Drawing2 to Drawing1
    img2_aligned = align_images(img1, img2)

    # Step 2: Grayscale conversion
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)

    # Step 3: Adaptive Thresholding (dynamic for template size)
    max_dim = max(img1.shape[:2] + img2.shape[:2])
    block_size = 21 if max_dim < 3000 else 61
    c_value = 10 if max_dim < 3000 else 15

    thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, block_size, c_value)
    thresh2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, block_size, c_value)

    # Step 4: Absolute difference and Otsu's refinement
    diff = cv2.absdiff(thresh1, thresh2)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Step 5: Morphological Expansion
    kernel_size = (3, 3) if max_dim < 3000 else (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Step 6: Red and Green mask creation
    red_mask = np.zeros_like(img1)
    green_mask = np.zeros_like(img2_aligned)

    red_mask[np.bitwise_and(mask > 0, thresh1 > thresh2)] = [0, 0, 255]
    green_mask[np.bitwise_and(mask > 0, thresh2 > thresh1)] = [0, 255, 0]

    # Step 7: Overlay on original Drawing1 background
    highlighted = img1.copy()
    highlighted[np.where((red_mask == [0, 0, 255]).all(axis=2))] = [0, 0, 255]
    highlighted[np.where((green_mask == [0, 255, 0]).all(axis=2))] = [0, 255, 0]

    return highlighted

def pdf_to_image(pdf_path, page_number=0, dpi=300):
    """
    Convert a PDF page to an image.
    """
    image=[]
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    for page_number in range (doc.page_count):
        page=doc.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        image.append(img)
    return image

def segment_image(img, n_clusters=3):
    """
    Segment the image into regions using K-Means clustering.
    """
    # Reshape the image to a 2D array of pixels
    pixels = img.reshape((-1, 3))  # 3 channels (BGR)
    pixels = np.float32(pixels)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)

    # Get the labels and cluster centers
    labels = kmeans.labels_
    centers = np.uint8(kmeans.cluster_centers_)

    # Map each pixel to its cluster center
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img.shape)

    return segmented_image, labels.reshape(img.shape[:2])

def refine_alignment(img1, img2, labels1, labels2):
    """
    Refine alignment based on segmented regions.
    """
    # Find the largest region (assumed to be the diagram)
    unique1, counts1 = np.unique(labels1, return_counts=True)
    unique2, counts2 = np.unique(labels2, return_counts=True)
    diagram_label1 = unique1[np.argmax(counts1)]
    diagram_label2 = unique2[np.argmax(counts2)]

    # Create masks for the diagram regions
    mask1 = (labels1 == diagram_label1).astype(np.uint8) * 255
    mask2 = (labels2 == diagram_label2).astype(np.uint8) * 255

    # Use the masks to refine alignment
    aligned_img2 = align_images(img1, img2, mask1, mask2)

    return aligned_img2

pdf1_path = "data/drawing1.pdf"
pdf2_path = "data/drawing2.pdf"

# Convert PDFs to images
img1 = pdf_to_image(pdf1_path)
img2 = pdf_to_image(pdf2_path)
for idx,(image1,image2) in enumerate(zip(img1,img2)):

    # Align second image to the first one
    aligned_img2 = align_images(image1, image2)

    # Highlight differences (Now in a single output file)
    combined_highlighted = highlight_differences(image1, aligned_img2)
    output_dir="outputs"

    # Dynamically create a unique output file name based on the index
    output_path=os.path.join(output_dir,f"highlighted_combined{idx+1}.png")
    #Save result in a single file with the unique name
    cv2.imwrite(output_path, combined_highlighted)

    print(f"Highlighted differences saved successfully at {output_path}")

