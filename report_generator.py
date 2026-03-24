from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'QNose - Quantum Breath Disease Detector', 0, 1, 'C')

def main():
    print("--- Generating Report ---")
    pdf = PDF()
    
    # Page 1: Title Page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 60, '', 0, 1)
    pdf.cell(0, 20, 'QNose Results Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, 'Classical vs Quantum Machine Learning for Parkinson\'s Detection', 0, 1, 'C')
    
    # Page 2: Accuracy Table 
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Model Accuracies', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'Detailed evaluation metrics recorded during pipeline execution.', 0, 1)
    
    # Page 3: Confusion Matrices
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Confusion Matrices (Classical vs Quantum)', 0, 1)
    if os.path.exists('confusion_matrices.png'):
        pdf.image('confusion_matrices.png', x=10, w=190)
        
    # Page 4: Quantum Circuit diagram
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Quantum Circuit Architecture', 0, 1)
    if os.path.exists('quantum_circuit.png'):
        pdf.image('quantum_circuit.png', x=10, w=190)
        
    # Page 5: SHAP Explanation
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'SHAP Explainability (Why was it flagged?)', 0, 1)
    if os.path.exists('shap_explanation.png'):
        pdf.image('shap_explanation.png', x=15, w=170)
        
    # Page 6: Conclusion
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Conclusion & Quantum Advantage', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, "Quantum Machine Learning (QML) models like QNose leverage complex, high-dimensional Hilbert spaces. While Classical SVMs perform well, the Quantum Kernel estimates similarities between samples by evolving them via quantum entanglement and interference. This foundational shift can potentially unearth non-linear correlations in complex biomarkers that classical systems may miss, offering a promising avenue for early and non-invasive detection of Parkinson's Disease.")
    
    pdf.output('QNose_Results.pdf')
    print("Generated QNose_Results.pdf")

if __name__ == '__main__':
    main()