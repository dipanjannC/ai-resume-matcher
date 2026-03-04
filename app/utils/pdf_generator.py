"""PDF generation utility for customized resumes."""

import io
from typing import Dict, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


class PDFGenerator:
    """Generate professional PDF resumes from structured data."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E3440')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor('#5E81AC'),
            borderWidth=1,
            borderColor=colors.HexColor('#D8DEE9'),
            borderPadding=3
        ))
        
        self.styles.add(ParagraphStyle(
            name='ContactInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='JobTitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceBefore=6,
            spaceAfter=3,
            textColor=colors.HexColor('#4C566A'),
            fontName='Helvetica-Bold'
        ))
    
    def generate_resume_pdf(self, resume_data: Dict[str, Any]) -> bytes:
        """Generate a PDF resume from structured resume data."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Build the content
        story = []
        
        # Header with name and contact info
        self._add_header(story, resume_data)
        
        # Professional summary
        if resume_data.get('summary'):
            self._add_section(story, 'Professional Summary', resume_data['summary'])
        
        # Skills
        if resume_data.get('skills'):
            self._add_skills_section(story, resume_data['skills'])
        
        # Experience
        if resume_data.get('experience'):
            self._add_experience_section(story, resume_data['experience'])
        
        # Education
        if resume_data.get('education'):
            self._add_education_section(story, resume_data['education'])
        
        # Projects
        if resume_data.get('projects'):
            self._add_projects_section(story, resume_data['projects'])
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _add_header(self, story, resume_data):
        """Add header with name and contact information."""
        profile = resume_data.get('profile', {})
        
        # Name
        name = profile.get('name', 'Unknown')
        story.append(Paragraph(name, self.styles['CustomTitle']))
        
        # Contact info
        contact_parts = []
        if profile.get('phone'):
            contact_parts.append(profile['phone'])
        if profile.get('email'):
            contact_parts.append(profile['email'])
        if profile.get('linkedin'):
            contact_parts.append(f"LinkedIn: {profile['linkedin']}")
        if profile.get('location'):
            contact_parts.append(profile['location'])
        
        if contact_parts:
            contact_text = " | ".join(contact_parts)
            story.append(Paragraph(contact_text, self.styles['ContactInfo']))
        
        story.append(Spacer(1, 0.2*inch))
    
    def _add_section(self, story, title, content):
        """Add a generic section with title and content."""
        story.append(Paragraph(title, self.styles['SectionHeader']))
        if isinstance(content, str):
            story.append(Paragraph(content, self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    def _add_skills_section(self, story, skills):
        """Add skills section."""
        story.append(Paragraph('Technical Skills', self.styles['SectionHeader']))
        
        if isinstance(skills, dict):
            for category, skill_list in skills.items():
                if skill_list:
                    skills_text = f"<b>{category}:</b> {', '.join(skill_list)}"
                    story.append(Paragraph(skills_text, self.styles['Normal']))
        elif isinstance(skills, list):
            skills_text = ', '.join(skills)
            story.append(Paragraph(skills_text, self.styles['Normal']))
        
        story.append(Spacer(1, 0.1*inch))
    
    def _add_experience_section(self, story, experience):
        """Add work experience section."""
        story.append(Paragraph('Professional Experience', self.styles['SectionHeader']))
        
        for exp in experience:
            # Job title and company
            title_company = f"<b>{exp.get('title', 'Position')}</b> at {exp.get('company', 'Company')}"
            story.append(Paragraph(title_company, self.styles['JobTitle']))
            
            # Duration and location
            duration_location = []
            if exp.get('duration'):
                duration_location.append(exp['duration'])
            if exp.get('location'):
                duration_location.append(exp['location'])
            
            if duration_location:
                story.append(Paragraph(' | '.join(duration_location), self.styles['Normal']))
            
            # Responsibilities
            if exp.get('responsibilities'):
                for resp in exp['responsibilities']:
                    story.append(Paragraph(f"• {resp}", self.styles['Normal']))
            
            story.append(Spacer(1, 0.1*inch))
    
    def _add_education_section(self, story, education):
        """Add education section."""
        story.append(Paragraph('Education', self.styles['SectionHeader']))
        
        for edu in education:
            # Degree and institution
            degree_school = f"<b>{edu.get('degree', 'Degree')}</b>, {edu.get('institution', 'Institution')}"
            story.append(Paragraph(degree_school, self.styles['Normal']))
            
            # Year and GPA
            details = []
            if edu.get('year'):
                details.append(f"Graduated: {edu['year']}")
            if edu.get('gpa'):
                details.append(f"GPA: {edu['gpa']}")
            
            if details:
                story.append(Paragraph(' | '.join(details), self.styles['Normal']))
            
            story.append(Spacer(1, 0.1*inch))
    
    def _add_projects_section(self, story, projects):
        """Add projects section."""
        story.append(Paragraph('Projects', self.styles['SectionHeader']))
        
        for project in projects:
            # Project name
            project_name = f"<b>{project.get('name', 'Project')}</b>"
            story.append(Paragraph(project_name, self.styles['JobTitle']))
            
            # Description
            if project.get('description'):
                story.append(Paragraph(project['description'], self.styles['Normal']))
            
            # Technologies
            if project.get('technologies'):
                tech_text = f"Technologies: {', '.join(project['technologies'])}"
                story.append(Paragraph(tech_text, self.styles['Normal']))
            
            story.append(Spacer(1, 0.1*inch))


def generate_resume_pdf(resume_data: Dict[str, Any]) -> bytes:
    """Generate a PDF resume from structured resume data."""
    generator = PDFGenerator()
    return generator.generate_resume_pdf(resume_data)