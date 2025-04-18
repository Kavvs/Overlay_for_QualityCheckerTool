--

**✅ Project Title: Quality Checker Tool – AutoCAD Drawing Comparator**
**🛠️ Overview**
The Quality Checker Tool is a Python-based system designed to automatically detect and highlight differences between two engineering drawings in PDF format (such as AutoCAD or CREO outputs). It's ideal for use cases in manufacturing, QA, and engineering design audits, ensuring even minute design discrepancies are flagged visually in red and green overlays.

**📌 Key Features**
🧠 ECC-Based Image Alignment – Precisely aligns drawings even with scale or rotation differences.

🟥🟩 Color-Based Difference Highlighting – Red for removed/changed, Green for added/new components.

🧾 PDF to Image Conversion – Supports multipage engineering PDFs.

🎯 Clustering-Based Segmentation (KMeans) – Smartly identifies and isolates major drawing areas.

📂 Batch Processing & Output Naming – Automatically processes and saves side-by-side comparisons.

⚙️ OpenCV & PyMuPDF Powered – Efficient and scalable for industrial-grade quality assurance.

🤝 Ideal Use Cases
Manufacturing & Fabrication QC

Product Lifecycle Management (PLM)

Reverse Engineering Verification

Regulatory & Compliance Inspections

🤖 Built With
Python

OpenCV

PyMuPDF (fitz)

scikit-learn (for clustering)

Flask (optional REST API for Windchill)

