import io
from typing import Dict, Any
from fpdf import FPDF

class PDFGenerator(FPDF):
    def header(self):
        # We don't need a repeating header for resumes typically, but we can add one if desired.
        pass

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('helvetica', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

class DocumentGenerator:
    """Service for generating downloadable PDF documents for resumes and cover letters"""
    
    def __init__(self):
        pass

    def generate_cover_letter_pdf(self, cover_letter_text: str, candidate_name: str, company: str) -> bytes:
        """
        Generate a PDF cover letter from text.
        Returns the PDF as bytes.
        """
        pdf = PDFGenerator()
        pdf.add_page()
        pdf.set_margins(20, 20, 20)
        
        # Add basic header
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, candidate_name, ln=True, align='C')
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 10, f"Cover Letter for {company}", ln=True, align='C')
        pdf.ln(10)
        
        # Add body text
        pdf.set_font("helvetica", "", 11)
        # multi_cell handles line breaks
        # We encode and decode to replace non-latin-1 characters if present to avoid FPDF errors
        clean_text = cover_letter_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, clean_text)
        
        return pdf.output(dest='S').encode('latin-1')

    def generate_resume_pdf(self, customized_resume_dict: Dict[str, Any], candidate_name: str, original_resume_data: Any) -> bytes:
        """
        Generate a PDF resume from the customized JSON dictionary.
        Returns the PDF as bytes.
        """
        pdf = PDFGenerator()
        pdf.add_page()
        pdf.set_margins(20, 20, 20)
        
        # Contact Header
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(0, 10, candidate_name.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
        
        pdf.set_font("helvetica", "", 10)
        contact_info = f"{getattr(original_resume_data.profile, 'email', '')} | {getattr(original_resume_data.profile, 'phone', '')} | {getattr(original_resume_data.profile, 'location', '')}"
        pdf.cell(0, 5, contact_info.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='C')
        pdf.ln(5)
        
        # Summary
        summary = customized_resume_dict.get('customized_summary', getattr(original_resume_data, 'summary', ''))
        if summary:
            pdf.set_font("helvetica", "B", 14)
            pdf.cell(0, 8, "Professional Summary", ln=True)
            pdf.set_font("helvetica", "", 11)
            pdf.multi_cell(0, 6, summary.encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)
            
        # Skills
        all_skills = customized_resume_dict.get('emphasized_skills', [])
        if all_skills:
            pdf.set_font("helvetica", "B", 14)
            pdf.cell(0, 8, "Core Competencies", ln=True)
            pdf.set_font("helvetica", "", 11)
            skills_text = " • ".join(all_skills)
            pdf.multi_cell(0, 6, skills_text.encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(5)
            
        # Experience
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 8, "Professional Experience", ln=True)
        pdf.set_font("helvetica", "", 11)
        
        # We try to use the original experience logic but weave in the modifications
        mods = customized_resume_dict.get('experience_modifications', [])
        
        # Dump original experience, optionally appending the modified bullet points
        if original_resume_data.experience and original_resume_data.experience.roles:
            # Quick format: just listing the customized bullets since the structure is loose
            for mod in mods:
                pdf.set_font("helvetica", "B", 12)
                section_name = mod.get('section_or_role', 'Experience Segment')
                pdf.cell(0, 6, section_name.encode('latin-1', 'replace').decode('latin-1'), ln=True)
                
                pdf.set_font("helvetica", "", 11)
                for suggestion in mod.get('suggestions', []):
                    # add bullet
                    pdf.multi_cell(0, 6, f"- {suggestion}".encode('latin-1', 'replace').decode('latin-1'))
                pdf.ln(2)
        else:
            pdf.multi_cell(0, 6, "See full resume for complete experience history.")

        return pdf.output(dest='S').encode('latin-1')

# Global Instance
document_generator = DocumentGenerator()
