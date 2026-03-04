#!/usr/bin/env python3
"""
CSV to TXT Resume Converter
Converts resume data from CSV format to individual .txt files for vector database ingestion.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional


def clean_field(value: Any) -> str:
    """Clean and format field values"""
    if not value or value in ['N/A', 'None', 'null', '']:
        return ""
    
    # Handle list-like strings
    if isinstance(value, str):
        if value.startswith('[') and value.endswith(']'):
            try:
                # Parse as JSON array
                parsed = json.loads(value.replace("'", '"'))
                if isinstance(parsed, list):
                    return ", ".join(str(item) for item in parsed if item and item != 'N/A')
            except:
                # Fallback: clean the string
                return value.strip("[]'\"").replace("', '", ", ").replace('", "', ", ")
    
    return str(value).strip()


def format_resume_text(row: Dict[str, str], resume_id: int) -> str:
    """Convert CSV row to formatted resume text"""
    
    # Extract and clean key fields
    career_objective = clean_field(row.get('career_objective', ''))
    skills = clean_field(row.get('skills', ''))
    education_institution = clean_field(row.get('educational_institution_name', ''))
    degree = clean_field(row.get('degree_names', ''))
    passing_year = clean_field(row.get('passing_years', ''))
    major_field = clean_field(row.get('major_field_of_studies', ''))
    company = clean_field(row.get('professional_company_names', ''))
    position = clean_field(row.get('positions', ''))
    start_date = clean_field(row.get('start_dates', ''))
    end_date = clean_field(row.get('end_dates', ''))
    responsibilities = clean_field(row.get('responsibilities', ''))
    job_skills = clean_field(row.get('related_skils_in_job', ''))
    languages = clean_field(row.get('languages', ''))
    certifications = clean_field(row.get('certification_skills', ''))
    
    # Format the resume text
    resume_text = f"""RESUME - Candidate {resume_id:06d}

PROFESSIONAL SUMMARY
{career_objective if career_objective else 'Experienced professional seeking new opportunities.'}

TECHNICAL SKILLS
{skills if skills else 'Various technical skills and competencies.'}

EDUCATION
Institution: {education_institution if education_institution else 'Educational background available'}
Degree: {degree if degree else 'Relevant degree'}
Year: {passing_year if passing_year else 'Recent graduate'}
Major: {major_field if major_field else 'Relevant field of study'}

PROFESSIONAL EXPERIENCE
Company: {company if company else 'Professional experience available'}
Position: {position if position else 'Relevant work experience'}
Duration: {start_date} - {end_date if end_date else 'Present'}

Key Responsibilities:
{responsibilities if responsibilities else 'Professional responsibilities and achievements.'}

Job-Related Skills:
{job_skills if job_skills else 'Relevant professional skills applied.'}

ADDITIONAL QUALIFICATIONS
Languages: {languages if languages else 'Communication skills available'}
Certifications: {certifications if certifications else 'Professional development and training'}

KEYWORDS: {skills}
"""
    
    return resume_text.strip()


def convert_csv_to_txt_files(csv_file_path: str, output_dir: str, max_records: int = 100):
    """
    Convert CSV resume data to individual .txt files
    
    Args:
        csv_file_path: Path to the CSV file
        output_dir: Directory to save .txt files
        max_records: Maximum number of records to process (default: 100)
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    print(f"🔄 Converting CSV to TXT files...")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Max records to process: {max_records}")
    print()
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for idx, row in enumerate(reader):
                if processed_count >= max_records:
                    break
                
                # Check if row has meaningful data
                career_objective = row.get('career_objective', '').strip()
                skills = row.get('skills', '').strip()
                
                if not career_objective and not skills:
                    skipped_count += 1
                    continue
                
                # Generate resume text
                resume_text = format_resume_text(row, idx + 1)
                
                # Save to file
                filename = f"resume_{idx+1:06d}.txt"
                file_path = output_path / filename
                
                with open(file_path, 'w', encoding='utf-8') as txtfile:
                    txtfile.write(resume_text)
                
                processed_count += 1
                
                # Progress indicator
                if processed_count % 10 == 0:
                    print(f"✅ Processed {processed_count} resumes...")
        
        print()
        print(f"🎉 Conversion completed!")
        print(f"📊 Successfully converted: {processed_count} resumes")
        print(f"⏭️  Skipped empty records: {skipped_count}")
        print(f"📁 Files saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")


def main():
    """Main function to run the conversion"""
    
    # Configuration
    CSV_FILE = "data/samples/resume_data.csv"
    OUTPUT_DIR = "data/processed_resumes"
    MAX_RECORDS = 50  # Adjust this number as needed
    
    print("📄 CSV to TXT Resume Converter")
    print("=" * 40)
    
    # Check if CSV file exists
    if not Path(CSV_FILE).exists():
        print(f"❌ CSV file not found: {CSV_FILE}")
        print("Please ensure the CSV file exists in the correct location.")
        return
    
    # Run conversion
    convert_csv_to_txt_files(CSV_FILE, OUTPUT_DIR, MAX_RECORDS)


if __name__ == "__main__":
    main()
