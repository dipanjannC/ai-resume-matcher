#!/usr/bin/env python3
"""
AI Resume Matcher - Quick Feature Demo
A streamlined demo script to showcase core functionalities and verify vector search
"""

import asyncio
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from app.services.resume_processor import resume_processor
    from app.services.job_processor import job_processor
    from app.services.vector_store import vector_store
    from app.services.embeddings import embedding_service
    from app.core.logging import get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

console = Console()
logger = get_logger(__name__)


class QuickDemo:
    """Quick demonstration of AI Resume Matcher features"""
    
    def __init__(self):
        self.processor = resume_processor
        self.job_processor = job_processor
        
    async def run(self):
        """Run the quick demo"""
        console.print(Panel.fit("🚀 AI Resume Matcher - Quick Demo", style="bold blue"))
        console.print("Demonstrating key features with vector search verification\n")
        
        while True:
            choice = self._show_menu()
            
            if choice == "1":
                await self._test_vector_search()
            elif choice == "2":
                await self._demo_resume_processing()
            elif choice == "3":
                await self._demo_semantic_matching()
            elif choice == "4":
                await self._demo_bulk_features()
            elif choice == "5":
                await self._quick_start_demo()
            elif choice == "0":
                console.print("👋 Demo completed!")
                break
            else:
                console.print("❌ Invalid choice. Please try again.")
    
    def _show_menu(self) -> str:
        """Display menu options"""
        console.print("\n" + "=" * 60)
        console.print("📋 Quick Demo Menu", style="bold cyan")
        console.print("1. 🔍 Test Vector Search (Technical Verification)")
        console.print("2. 📄 Demo Resume Processing (LangChain AI)")
        console.print("3. 🧠 Demo Semantic Matching (Kong→API Gateway)")
        console.print("4. 🚀 Demo Bulk Processing (Enterprise Scale)")
        console.print("5. ⚡ Quick Start (All Features)")
        console.print("0. 🚪 Exit")
        console.print("=" * 60)
        
        return Prompt.ask("Choose demo", choices=["0", "1", "2", "3", "4", "5"])
    
    async def _test_vector_search(self):
        """Test and verify vector search functionality"""
        console.print("\n🔍 Vector Search Verification", style="bold green")
        console.print("━" * 50)
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]"),
                transient=True,
            ) as progress:
                
                # Test 1: Check vector store
                task1 = progress.add_task("Checking vector store...", total=None)
                candidates = vector_store.get_all_candidates()
                progress.update(task1, description=f"✅ Vector store active ({len(candidates)} candidates)")
                await asyncio.sleep(0.5)
                
                # Test 2: Test embeddings
                task2 = progress.add_task("Testing embeddings...", total=None)
                test_text = "API Gateway Kong Apigee microservices"
                embedding = embedding_service.generate_embedding(test_text)
                progress.update(task2, description=f"✅ Embeddings working (dim: {len(embedding)})")
                await asyncio.sleep(0.5)
                
                # Test 3: Vector search
                task3 = progress.add_task("Testing vector search...", total=None)
                results = []
                if len(candidates) > 0:
                    results = vector_store.search_similar(embedding, top_k=3)
                    progress.update(task3, description=f"✅ Search working ({len(results)} results)")
                else:
                    progress.update(task3, description="⚠️ No data - need to process resumes")
                await asyncio.sleep(0.5)
            
            # Display results
            if len(candidates) > 0:
                console.print(f"\n📊 Vector Store Status:")
                console.print(f"   • Total candidates: {len(candidates)}")
                console.print(f"   • Embedding dimension: {len(embedding)}")
                console.print(f"   • Search results: {len(results)}")
                
                if results:
                    console.print(f"\n🎯 Sample Search Results:")
                    for i, result in enumerate(results[:3], 1):
                        similarity = result.get('similarity', 0)
                        candidate_id = result.get('candidate_id', 'Unknown')[:12]
                        console.print(f"   {i}. {candidate_id}... (similarity: {similarity:.3f})")
            else:
                console.print("📝 No candidates in vector store yet.")
                if Confirm.ask("Process sample resumes now?"):
                    await self._process_sample_data()
            
            console.print("\n✅ Vector search verification complete!")
            
        except Exception as e:
            console.print(f"❌ Vector search test failed: {str(e)}")
    
    async def _demo_resume_processing(self):
        """Demonstrate resume processing with AI"""
        console.print("\n📄 Resume Processing Demo", style="bold green")
        console.print("━" * 50)
        
        # Check for sample resumes
        sample_dir = Path("data/resumes")
        if not sample_dir.exists() or not list(sample_dir.glob("*.txt")):
            console.print("❌ No sample resumes found in data/resumes/")
            console.print("Please add some .txt resume files to the data/resumes/ directory")
            return
        
        sample_files = list(sample_dir.glob("*.txt"))
        console.print(f"📁 Found {len(sample_files)} sample resume files")
        
        # Select a resume to process
        if len(sample_files) > 1:
            console.print("Available resumes:")
            for i, file in enumerate(sample_files[:5], 1):
                console.print(f"  {i}. {file.name}")
            
            choice = Prompt.ask("Choose resume to process (1-5)", default="1")
            try:
                selected_file = sample_files[int(choice) - 1]
            except (ValueError, IndexError):
                selected_file = sample_files[0]
        else:
            selected_file = sample_files[0]
        
        console.print(f"\n📋 Processing: {selected_file.name}")
        
        try:
            # Read and process the resume
            with open(selected_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Show preview
            console.print("\n📝 Resume Content Preview:")
            console.print(Panel(content[:300] + "...", style="dim"))
            
            # Process with AI
            console.print("\n🤖 Processing with LangChain AI...")
            start_time = time.time()
            
            result = await self.processor.process_resume_content(content, selected_file.name)
            
            end_time = time.time()
            console.print(f"⚡ Processed in {end_time - start_time:.2f} seconds")
            
            # Display results
            self._display_parsed_resume(result)
            
        except Exception as e:
            console.print(f"❌ Processing failed: {str(e)}")
    
    async def _demo_semantic_matching(self):
        """Demonstrate semantic matching capabilities"""
        console.print("\n🧠 Semantic Matching Demo", style="bold green")
        console.print("━" * 50)
        console.print("Showing how Kong/Apigee experience matches API Gateway jobs")
        
        try:
            # Ensure we have some data
            candidates = vector_store.get_all_candidates()
            if len(candidates) == 0:
                console.print("📝 No candidates found. Processing sample data...")
                await self._process_sample_data()
                candidates = vector_store.get_all_candidates()
            
            if len(candidates) == 0:
                console.print("❌ No candidates available for matching")
                return
            
            # Create or get API Gateway job
            console.print("🏢 Creating API Gateway job...")
            api_job_desc = """
            Senior API Gateway Architect
            
            We need an experienced engineer to design and implement our API infrastructure.
            
            Requirements:
            - API Gateway experience (Kong, Apigee, AWS API Gateway)
            - Microservices architecture
            - RESTful API design
            - Authentication and security (OAuth2, JWT)
            - Cloud platforms (AWS, Azure, GCP)
            - DevOps and CI/CD experience
            """
            
            job_data = await self.processor.process_job_description(
                api_job_desc,
                title="Senior API Gateway Architect",
                company="DemoTech Corp"
            )
            
            # Find matches
            console.print("🔍 Finding semantic matches...")
            matches = await self.processor.find_best_matches(job_data, top_k=5)
            
            if not matches:
                console.print("❌ No matches found")
                return
            
            # Display results
            console.print(f"\n🎯 Top {len(matches)} Semantic Matches:")
            
            table = Table()
            table.add_column("Rank", style="cyan", width=6)
            table.add_column("Candidate", style="green", width=20)
            table.add_column("Overall", style="yellow", width=8)
            table.add_column("Semantic", style="blue", width=8)
            table.add_column("Explanation", style="magenta")
            
            for i, match in enumerate(matches, 1):
                candidate_name = getattr(match, 'candidate_name', f"Candidate {i}")
                overall = f"{match.overall_score:.1%}"
                semantic = f"{getattr(match, 'semantic_similarity_score', 0):.1%}"
                explanation = match.match_summary[:40] + "..." if len(match.match_summary) > 40 else match.match_summary
                
                table.add_row(str(i), candidate_name, overall, semantic, explanation)
            
            console.print(table)
            
            # Show detailed analysis for top match
            top_match = matches[0]
            console.print(f"\n📊 Detailed Analysis - Top Match:")
            console.print(f"   🎯 Overall Score: {top_match.overall_score:.1%}")
            console.print(f"   🔧 Skills Match: {top_match.skills_match_score:.1%}")
            console.print(f"   💼 Experience Match: {top_match.experience_match_score:.1%}")
            console.print(f"   🧠 Semantic Similarity: {getattr(top_match, 'semantic_similarity_score', 0):.1%}")
            
            if hasattr(top_match, 'matching_skills') and top_match.matching_skills:
                console.print(f"\n✅ Matching Skills: {', '.join(top_match.matching_skills[:5])}")
            
            console.print(f"\n💡 AI Explanation: {top_match.match_summary}")
            
        except Exception as e:
            console.print(f"❌ Semantic matching failed: {str(e)}")
    
    async def _demo_bulk_features(self):
        """Demonstrate bulk processing capabilities"""
        console.print("\n🚀 Bulk Processing Demo", style="bold green")
        console.print("━" * 50)
        
        try:
            # Check current state
            candidates = vector_store.get_all_candidates()
            jobs = await self.job_processor.list_stored_jobs()
            
            console.print(f"📊 Current State:")
            console.print(f"   • Candidates: {len(candidates)}")
            console.print(f"   • Jobs: {len(jobs)}")
            
            # Simulate bulk processing
            if len(candidates) < 10:
                console.print("\n📁 Simulating bulk resume processing...")
                
                # Process sample resumes
                sample_dir = Path("data/resumes")
                if sample_dir.exists():
                    sample_files = list(sample_dir.glob("*.txt"))
                    
                    with Progress(
                        TextColumn("[progress.description]"),
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Processing resumes...", total=len(sample_files))
                        
                        for file in sample_files:
                            progress.update(task, description=f"Processing {file.name}...")
                            try:
                                with open(file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                await self.processor.process_resume_content(content, file.name)
                                progress.advance(task)
                                await asyncio.sleep(0.1)  # Brief pause for demo
                                
                            except Exception as e:
                                console.print(f"⚠️ Failed to process {file.name}: {str(e)}")
            
            # Test bulk search speed
            console.print("\n⚡ Testing search performance...")
            search_queries = [
                "API Gateway Kong Apigee",
                "Flutter React Native mobile",
                "Kubernetes Docker DevOps",
                "Python Django FastAPI",
                "AWS Azure cloud engineer"
            ]
            
            total_search_time = 0
            for query in search_queries:
                start_time = time.time()
                embedding = embedding_service.generate_embedding(query)
                results = vector_store.search_similar(embedding, top_k=10)
                end_time = time.time()
                
                search_time = end_time - start_time
                total_search_time += search_time
                console.print(f"   🔍 '{query[:20]}...': {len(results)} results in {search_time:.3f}s")
            
            avg_search_time = total_search_time / len(search_queries)
            console.print(f"\n📈 Performance Metrics:")
            console.print(f"   • Average search time: {avg_search_time:.3f} seconds")
            console.print(f"   • Total candidates: {len(vector_store.get_all_candidates())}")
            console.print(f"   • Searches per second: {1/avg_search_time:.1f}")
            
            console.print("\n✅ Bulk processing demo complete!")
            
        except Exception as e:
            console.print(f"❌ Bulk processing demo failed: {str(e)}")
    
    async def _quick_start_demo(self):
        """Quick start demo showing all features"""
        console.print("\n⚡ Quick Start - All Features Demo", style="bold green")
        console.print("━" * 50)
        
        try:
            # Step 1: Initialize data
            console.print("1️⃣ Checking system status...")
            candidates = vector_store.get_all_candidates()
            
            if len(candidates) == 0:
                console.print("   📝 Processing sample data...")
                await self._process_sample_data()
            
            # Step 2: Create a job
            console.print("2️⃣ Creating sample job...")
            job_desc = "Looking for a Flutter developer with 3+ years experience in mobile app development"
            job_data = await self.processor.process_job_description(
                job_desc,
                title="Flutter Developer",
                company="QuickDemo Inc"
            )
            
            # Step 3: Find matches
            console.print("3️⃣ Finding matches...")
            start_time = time.time()
            matches = await self.processor.find_best_matches(job_data, top_k=3)
            end_time = time.time()
            
            # Step 4: Display results
            console.print(f"4️⃣ Results found in {end_time - start_time:.2f} seconds")
            
            if matches:
                for i, match in enumerate(matches, 1):
                    candidate_name = getattr(match, 'candidate_name', f"Candidate {i}")
                    console.print(f"   {i}. {candidate_name} - Score: {match.overall_score:.1%}")
            else:
                console.print("   No matches found")
            
            console.print("\n🎉 Quick start demo complete!")
            console.print("✅ All core features working correctly")
            
        except Exception as e:
            console.print(f"❌ Quick start demo failed: {str(e)}")
    
    async def _process_sample_data(self):
        """Process sample resume data"""
        sample_dir = Path("data/resumes")
        if not sample_dir.exists():
            console.print("❌ No data/resumes directory found")
            return
        
        sample_files = list(sample_dir.glob("*.txt"))
        if not sample_files:
            console.print("❌ No .txt files found in data/resumes/")
            return
        
        console.print(f"📁 Processing {len(sample_files)} sample resumes...")
        
        for file in sample_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                await self.processor.process_resume_content(content, file.name)
                console.print(f"   ✅ Processed {file.name}")
                
            except Exception as e:
                console.print(f"   ❌ Failed {file.name}: {str(e)}")
    
    def _display_parsed_resume(self, resume_data):
        """Display parsed resume data"""
        console.print(f"\n👤 Parsed Resume Data:")
        console.print(f"   📛 Name: {resume_data.profile.name}")
        console.print(f"   💼 Title: {resume_data.profile.title}")
        console.print(f"   📍 Location: {resume_data.profile.location}")
        console.print(f"   📧 Email: {resume_data.profile.email}")
        console.print(f"   📞 Phone: {resume_data.profile.phone}")
        console.print(f"   🕒 Experience: {resume_data.experience.total_years} years")
        
        if resume_data.skills.technical:
            skills_preview = ', '.join(resume_data.skills.technical[:8])
            if len(resume_data.skills.technical) > 8:
                skills_preview += f" (+{len(resume_data.skills.technical) - 8} more)"
            console.print(f"   🔧 Top Skills: {skills_preview}")
        
        if resume_data.summary:
            console.print(f"\n📝 AI Summary:")
            console.print(Panel(resume_data.summary, style="blue"))


async def main():
    """Main demo entry point"""
    demo = QuickDemo()
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
