# utils/pdf_utils.py

from fpdf import FPDF
import os
import cv2
from tkinter import simpledialog


def save_results_as_pdf(original_image, segmented_image, hole_overlay, detections, conf_threshold, scale_factor):
    pdf_filename = simpledialog.askstring("Save PDF", "Enter the name for the PDF file (without extension):")
    if not pdf_filename:
        print("PDF filename not provided. Exiting...")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Detection Results", ln=True, align="C")

    # Adding the parameters
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Trust Factor: {conf_threshold * 100:.2f}%", ln=True, align="C")
    pdf.cell(200, 10, f"Scale Factor: {scale_factor * 100:.2f}%", ln=True, align="C")
    pdf.ln(10)

    # Save temporary images for the PDF
    original_path = "original_temp.jpg"
    segmented_path = "segmented_temp.jpg"
    overlay_path = "overlay_temp.jpg"
    cv2.imwrite(original_path, original_image)
    cv2.imwrite(segmented_path, segmented_image)
    cv2.imwrite(overlay_path, hole_overlay)

    # Add 'Original' text above the original image
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Original", ln=True, align="C")
    pdf.image(original_path, x=70, w=100)
    pdf.ln(10)

    # Add 'Segmented' text above the segmented image
    pdf.cell(200, 10, "Segmented", ln=True, align="C")
