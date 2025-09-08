"""
AI Resume Matcher Demo Script

This demo showcases the LangChain-powered resume matching system.
No database required - uses file-based storage and ChromaDB for vectors.

Features demonstrated:
1. Smart resume parsing with LangChain agents
2. Intelligent job description analysis
3. AI-powered candidate matching
4. Structured data extraction and insights
"""

import asyncio
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.resume_processor import resume_processor
from app.core.config import settings
from app.core.logging import get_logger

console = Console()
logger = get_logger(__name__)


class ResumeMatcherDemo:
    """Interactive demo for the AI Resume Matcher"""
    
    def __init__(self):
        self.processor = resume_processor
        
    async def run(self):
        """Run the interactive demo"""
        console.print(Panel.fit("🤖 AI Resume Matcher Demo", style="bold blue"))
        console.print("Welcome to the LangChain-powered Resume Matching System!\n")
        
        while True:
            choice = self._show_menu()
            
            if choice == "1":
                await self._demo_resume_upload()
            elif choice == "2":
                await self._demo_job_matching()
            elif choice == "3":
                await self._demo_sample_data()
            elif choice == "4":
                await self._show_processed_resumes()
            elif choice == "5":
                await self._analyze_specific_resume()
            elif choice == "0":
                console.print("👋 Thanks for trying the AI Resume Matcher!")
                break
            else:
                console.print("❌ Invalid choice. Please try again.")
    
    def _show_menu(self) -> str:
        """Display the main menu and get user choice"""
        console.print("\n" + "=" * 50)
        console.print("📋 Main Menu", style="bold cyan")
        console.print("1. 📄 Upload and Process Resume")
        console.print("2. 🎯 Match Resumes to Job Description")
        console.print("3. 🧪 Load Sample Data")
        console.print("4. 📊 View Processed Resumes")
        console.print("5. 🔍 Analyze Specific Resume")
        console.print("0. 🚪 Exit")
        console.print("=" * 50)
        
        return Prompt.ask("Choose an option", choices=["0", "1", "2", "3", "4", "5"])
    
    async def _demo_resume_upload(self):
        """Demo resume upload and processing"""
        console.print("\n📄 Resume Upload Demo", style="bold green")
        
        # Check for sample resumes
        sample_dir = Path("data/resumes")
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.txt"))
            if sample_files:
                console.print("Available sample resumes:")
                for i, file in enumerate(sample_files, 1):
                    console.print(f"  {i}. {file.name}")
                
                file_choice = Prompt.ask("Choose a sample file (enter number) or 'custom' for custom path")
                
                if file_choice.isdigit() and 1 <= int(file_choice) <= len(sample_files):
                    file_path = str(sample_files[int(file_choice) - 1])
                elif file_choice.lower() == 'custom':
                    file_path = Prompt.ask("Enter the full path to your resume file")
                else:
                    console.print("❌ Invalid choice")
                    return
            else:
                file_path = Prompt.ask("Enter the full path to your resume file")
        else:
            file_path = Prompt.ask("Enter the full path to your resume file")
        
        if not Path(file_path).exists():
            console.print(f"❌ File not found: {file_path}")
            return
        
        # Process the resume
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🔄 Processing resume with LangChain agents...", total=None)
            
            try:
                resume_data = await self.processor.process_resume_file(file_path)
                progress.remove_task(task)
                
                # Display results
                await self._display_resume_analysis(resume_data)
                
            except Exception as e:
                progress.remove_task(task)
                console.print(f"❌ Error processing resume: {str(e)}")
    
    async def _demo_job_matching(self):
        """Demo job description matching"""
        console.print("\n🎯 Job Matching Demo", style="bold green")
        
        # Check if we have processed resumes
        resumes = await self.processor.list_processed_resumes()
        if not resumes:
            console.print("❌ No processed resumes found. Please upload some resumes first.")
            return
        
        console.print(f"📊 Found {len(resumes)} processed resumes")
        
        # Get job description
        console.print("\nEnter the job description:")
        console.print("(You can paste multiple lines. Type 'END' on a new line to finish)")
        
        job_lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            job_lines.append(line)
        
        job_text = "\n".join(job_lines)
        if not job_text.strip():
            console.print("❌ No job description provided")
            return
        
        job_title = Prompt.ask("Job title (optional)", default="")
        company = Prompt.ask("Company name (optional)", default="")
        
        # Process job and find matches
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task1 = progress.add_task("🔄 Analyzing job description...", total=None)
            
            try:
                job_data = await self.processor.process_job_description(job_text, job_title, company)
                progress.update(task1, description="✅ Job analysis complete")
                
                task2 = progress.add_task("🔄 Finding best matches...", total=None)
                matches = await self.processor.find_best_matches(job_data, top_k=5)
                progress.remove_task(task1)
                progress.remove_task(task2)
                
                # Display match results
                await self._display_match_results(job_data, matches)
                
            except Exception as e:
                progress.remove_task(task1)
                if 'task2' in locals():
                    progress.remove_task(task2)
                console.print(f"❌ Error during matching: {str(e)}")
    
    async def _demo_sample_data(self):
        """Load and process sample data"""
        console.print("\n🧪 Sample Data Demo", style="bold green")
        
        sample_resumes = {
            "John Doe - Senior Developer": """
John Doe
Senior Software Developer

Email: john.doe@email.com
Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johndoe
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Experienced senior software developer with 8+ years in full-stack development.
Expert in Python, React, and cloud technologies with proven leadership skills.

EXPERIENCE
Senior Software Developer | TechCorp | 2020-Present
- Lead a team of 5 developers building scalable web applications
- Developed microservices architecture using Python and Docker
- Implemented CI/CD pipelines reducing deployment time by 60%

Software Developer | StartupXYZ | 2018-2020
- Built responsive web applications using React and Node.js
- Integrated third-party APIs and payment systems
- Mentored junior developers and conducted code reviews

TECHNICAL SKILLS
Programming Languages: Python, JavaScript, TypeScript, Java
Frameworks: React, FastAPI, Django, Node.js
Databases: PostgreSQL, MongoDB, Redis
Cloud: AWS, Docker, Kubernetes
Tools: Git, Jenkins, Terraform

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2016
            """,
            
            "Sarah Chen - Data Scientist": """
Sarah Chen
Data Scientist & Machine Learning Engineer

Email: sarah.chen@email.com
Phone: (555) 987-6543
LinkedIn: linkedin.com/in/sarahchen
Location: New York, NY

PROFESSIONAL SUMMARY
Innovative data scientist with 6+ years of experience in machine learning,
data analysis, and AI model development. PhD in Statistics with expertise
in deep learning and natural language processing.

EXPERIENCE
Senior Data Scientist | DataTech Solutions | 2021-Present
- Developed and deployed ML models improving customer retention by 25%
- Led data science initiatives for recommendation systems
- Built real-time analytics dashboards using Python and Tableau

Data Scientist | AI Innovations | 2019-2021
- Created predictive models for financial risk assessment
- Implemented NLP solutions for text analysis and sentiment detection
- Collaborated with engineering teams to productionize ML models

Research Assistant | Stanford AI Lab | 2017-2019
- Published 5 research papers on deep learning applications
- Developed novel neural network architectures for computer vision

TECHNICAL SKILLS
Programming: Python, R, SQL, Scala
ML/AI: TensorFlow, PyTorch, Scikit-learn, Keras
Data Tools: Pandas, NumPy, Jupyter, Apache Spark
Visualization: Matplotlib, Seaborn, Tableau, D3.js
Cloud: AWS SageMaker, Google Cloud AI, Azure ML

EDUCATION
PhD in Statistics | Stanford University | 2019
MS in Computer Science | MIT | 2017
            """
        }
        
        # Process sample resumes
        for name, content in sample_resumes.items():
            console.print(f"📄 Processing: {name}")
            
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"Processing {name}..."),
                    console=console
                ) as progress:
                    task = progress.add_task("processing", total=None)
                    resume_data = await self.processor.process_resume_content(content, f"{name}.txt")
                    progress.remove_task(task)
                
                console.print(f"✅ Successfully processed: {name}")
                
            except Exception as e:
                console.print(f"❌ Error processing {name}: {str(e)}")
        
        console.print("\n🎉 Sample data processing complete!")
        
        # Offer to run a sample job match
        if Confirm.ask("Would you like to run a sample job matching demo?"):
            await self._demo_sample_job_match()
    
    async def _demo_sample_job_match(self):
        """Demo with a sample job description"""
        sample_job = """
Senior Full-Stack Developer
TechStartup Inc.

We are looking for a Senior Full-Stack Developer to join our growing team.

Requirements:
- 5+ years of experience in web development
- Strong proficiency in Python and JavaScript
- Experience with React and modern frontend frameworks
- Knowledge of databases (PostgreSQL, MongoDB)
- Experience with cloud platforms (AWS preferred)
- Strong problem-solving and communication skills
- Team leadership experience preferred

Responsibilities:
- Design and develop scalable web applications
- Lead technical decisions and mentor junior developers
- Collaborate with product team on feature development
- Implement best practices for code quality and testing
- Work with DevOps team on deployment and infrastructure

Nice to have:
- Experience with microservices architecture
- Knowledge of Docker and Kubernetes
- Previous startup experience
- Open source contributions
        """
        
        console.print("\n🎯 Running sample job match...")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("🔄 Processing job description...", total=None)
                job_data = await self.processor.process_job_description(
                    sample_job, "Senior Full-Stack Developer", "TechStartup Inc."
                )
                
                progress.update(task, description="🔄 Finding matches...")
                matches = await self.processor.find_best_matches(job_data, top_k=3)
                progress.remove_task(task)
            
            await self._display_match_results(job_data, matches)
            
        except Exception as e:
            console.print(f"❌ Error in sample job match: {str(e)}")
    
    async def _show_processed_resumes(self):
        """Show all processed resumes"""
        console.print("\n📊 Processed Resumes", style="bold green")
        
        resumes = await self.processor.list_processed_resumes()
        
        if not resumes:
            console.print("❌ No processed resumes found.")
            return
        
        table = Table(title="Processed Resumes")
        table.add_column("Name", style="cyan")
        table.add_column("Title", style="magenta")
        table.add_column("Experience", style="green")
        table.add_column("Processed", style="yellow")
        
        for resume in resumes:
            table.add_row(
                resume["name"],
                resume["title"],
                f"{resume['experience_years']} years",
                resume["processed_at"][:10]  # Just the date
            )
        
        console.print(table)
    
    async def _analyze_specific_resume(self):
        """Analyze a specific resume in detail"""
        console.print("\n🔍 Resume Analysis", style="bold green")
        
        resumes = await self.processor.list_processed_resumes()
        if not resumes:
            console.print("❌ No processed resumes found.")
            return
        
        # Show available resumes
        console.print("Available resumes:")
        for i, resume in enumerate(resumes, 1):
            console.print(f"  {i}. {resume['name']} - {resume['title']}")
        
        choice = Prompt.ask("Choose a resume to analyze (enter number)")
        
        try:
            resume_idx = int(choice) - 1
            if 0 <= resume_idx < len(resumes):
                resume_id = resumes[resume_idx]['id']
                summary = await self.processor.get_resume_summary(resume_id)
                await self._display_resume_analysis_from_summary(summary)
            else:
                console.print("❌ Invalid choice")
        except ValueError:
            console.print("❌ Invalid input")
    
    async def _display_resume_analysis(self, resume_data):
        """Display detailed resume analysis"""
        console.print("\n🎉 Resume Processing Complete!", style="bold green")
        
        # Profile information
        profile_text = Text()
        profile_text.append(f"Name: {resume_data.profile.name}\n", style="bold cyan")
        profile_text.append(f"Title: {resume_data.profile.title}\n", style="cyan")
        profile_text.append(f"Location: {resume_data.profile.location}\n", style="cyan")
        profile_text.append(f"Email: {resume_data.profile.email}\n", style="cyan")
        console.print(Panel(profile_text, title="👤 Profile", border_style="cyan"))
        
        # Experience summary
        exp_text = Text()
        exp_text.append(f"Total Experience: {resume_data.experience.total_years} years\n", style="bold green")
        exp_text.append(f"Recent Roles: {', '.join(resume_data.experience.roles[:3])}\n", style="green")
        exp_text.append(f"Companies: {', '.join(resume_data.experience.companies[:3])}\n", style="green")
        console.print(Panel(exp_text, title="💼 Experience", border_style="green"))
        
        # Skills
        skills_text = Text()
        skills_text.append(f"Technical Skills ({len(resume_data.skills.technical)}): ", style="bold yellow")
        skills_text.append(f"{', '.join(resume_data.skills.technical[:10])}\n", style="yellow")
        if len(resume_data.skills.technical) > 10:
            skills_text.append(f"... and {len(resume_data.skills.technical) - 10} more\n", style="dim yellow")
        
        if resume_data.skills.certifications:
            skills_text.append(f"Certifications: {', '.join(resume_data.skills.certifications)}\n", style="yellow")
        
        console.print(Panel(skills_text, title="🛠️ Skills", border_style="yellow"))
        
        # AI Summary
        if resume_data.summary:
            console.print(Panel(resume_data.summary, title="🤖 AI-Generated Summary", border_style="magenta"))
        
        # Key Strengths
        if resume_data.key_strengths:
            strengths_text = "\n".join(f"• {strength}" for strength in resume_data.key_strengths)
            console.print(Panel(strengths_text, title="💪 Key Strengths", border_style="red"))
    
    async def _display_resume_analysis_from_summary(self, summary):
        """Display resume analysis from summary data"""
        console.print(f"\n📋 Resume Analysis: {summary['filename']}", style="bold green")
        
        # Profile
        profile = summary['profile']
        profile_text = Text()
        profile_text.append(f"Name: {profile.get('name', 'N/A')}\n", style="bold cyan")
        profile_text.append(f"Title: {profile.get('title', 'N/A')}\n", style="cyan")
        profile_text.append(f"Location: {profile.get('location', 'N/A')}\n", style="cyan")
        console.print(Panel(profile_text, title="👤 Profile", border_style="cyan"))
        
        # Experience
        exp = summary['experience_summary']
        exp_text = Text()
        exp_text.append(f"Total Experience: {exp['total_years']} years\n", style="bold green")
        exp_text.append(f"Recent Roles: {', '.join(exp['recent_roles'])}\n", style="green")
        console.print(Panel(exp_text, title="💼 Experience", border_style="green"))
        
        # Skills
        skills = summary['skills_summary']
        skills_text = Text()
        skills_text.append(f"Technical Skills ({skills['technical_count']}): ", style="bold yellow")
        skills_text.append(f"{', '.join(skills['top_skills'])}\n", style="yellow")
        console.print(Panel(skills_text, title="🛠️ Skills", border_style="yellow"))
        
        # Summary
        if summary.get('summary'):
            console.print(Panel(summary['summary'], title="🤖 AI Summary", border_style="magenta"))
    
    async def _display_match_results(self, job_data, matches):
        """Display job matching results"""
        console.print(f"\n🎯 Match Results for: {job_data.title}", style="bold green")
        
        if not matches:
            console.print("❌ No suitable matches found.")
            return
        
        # Job requirements summary
        req_text = Text()
        req_text.append(f"Required Skills: {', '.join(job_data.required_skills[:5])}\n", style="red")
        req_text.append(f"Experience Required: {job_data.experience_years} years\n", style="red")
        console.print(Panel(req_text, title="📋 Job Requirements", border_style="red"))
        
        # Match results table
        table = Table(title=f"Top {len(matches)} Candidates")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Candidate", style="cyan")
        table.add_column("Overall Score", style="green")
        table.add_column("Skills Match", style="yellow")
        table.add_column("Experience", style="magenta")
        table.add_column("Recommendation", style="blue")
        
        for i, match in enumerate(matches, 1):
            # Get candidate info
            resume_data = await self.processor._get_resume_data(match.resume_id)
            candidate_name = "Unknown"
            if resume_data:
                candidate_name = resume_data.profile.name or resume_data.filename
            
            table.add_row(
                str(i),
                candidate_name,
                f"{match.overall_score:.2f}",
                f"{match.skills_match_score:.2f}",
                f"{match.experience_match_score:.2f}",
                match.recommendation[:50] + "..." if len(match.recommendation) > 50 else match.recommendation
            )
        
        console.print(table)
        
        # Detailed analysis for top candidate
        if matches and Confirm.ask("Show detailed analysis for top candidate?"):
            top_match = matches[0]
            await self._display_detailed_match(top_match)
    
    async def _display_detailed_match(self, match):
        """Display detailed match analysis"""
        console.print("\n🔍 Detailed Match Analysis", style="bold blue")
        
        # Scores breakdown
        scores_text = Text()
        scores_text.append(f"Overall Score: {match.overall_score:.2f}\n", style="bold green")
        scores_text.append(f"Skills Match: {match.skills_match_score:.2f}\n", style="yellow")
        scores_text.append(f"Experience Match: {match.experience_match_score:.2f}\n", style="magenta")
        scores_text.append(f"Semantic Similarity: {match.semantic_similarity_score:.2f}\n", style="cyan")
        console.print(Panel(scores_text, title="📊 Scores", border_style="green"))
        
        # Matching skills
        if match.matching_skills:
            matching_text = ", ".join(match.matching_skills)
            console.print(Panel(matching_text, title="✅ Matching Skills", border_style="green"))
        
        # Missing skills
        if match.missing_skills:
            missing_text = ", ".join(match.missing_skills)
            console.print(Panel(missing_text, title="❌ Missing Skills", border_style="red"))
        
        # Strengths and improvements
        if match.strength_areas:
            strengths_text = "\n".join(f"• {area}" for area in match.strength_areas)
            console.print(Panel(strengths_text, title="💪 Strengths", border_style="blue"))
        
        if match.improvement_areas:
            improvements_text = "\n".join(f"• {area}" for area in match.improvement_areas)
            console.print(Panel(improvements_text, title="📈 Areas for Improvement", border_style="yellow"))
        
        # Match summary
        if match.match_summary:
            console.print(Panel(match.match_summary, title="📝 Match Summary", border_style="magenta"))
        
        # Recommendation
        if match.recommendation:
            console.print(Panel(match.recommendation, title="💡 Hiring Recommendation", border_style="cyan"))


async def main():
    """Main entry point for the demo"""
    try:
        # Initialize demo
        demo = ResumeMatcherDemo()
        
        # Check if OpenAI API key is configured
        if not settings.OPENAI_API_KEY:
            console.print("⚠️  Warning: OPENAI_API_KEY not configured.", style="yellow")
            console.print("Some features may not work properly.")
            console.print("Please set your OpenAI API key in the .env file or environment variable.")
        
        # Run the demo
        await demo.run()
        
    except KeyboardInterrupt:
        console.print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        console.print(f"\n❌ Demo error: {str(e)}")
        logger.error(f"Demo error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
