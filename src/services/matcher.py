from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MatchResult:
    candidate_id: str
    semantic_score: float
    keyword_score: float
    experience_score: float
    final_score: float
    matched_skills: List[str]
    explanation: str
    metadata: Dict[str, Any]


class ResumeMatcherService:
    def __init__(self):
        self.semantic_weight = 0.65
        self.keyword_weight = 0.30
        self.experience_weight = 0.05
    
    def match_resumes(self, 
                     job_description: str, 
                     job_title: Optional[str] = None,
                     required_skills: Optional[List[str]] = None,
                     required_experience_years: Optional[int] = None,
                     top_k: int = 10) -> List[MatchResult]:
        """
        Match resumes against a job description
        """
        try:
            logger.info("Starting resume matching", 
                       job_title=job_title, top_k=top_k)
            
            # Generate embedding for job description
            job_embedding = embedding_service.generate_embedding(job_description)
            
            # Extract job skills if not provided
            if not required_skills:
                required_skills = self._extract_job_skills(job_description)
            
            # Search for similar resumes
            similar_resumes = vector_store.search_similar(
                query_embedding=job_embedding,
                top_k=top_k * 2  # Get more candidates for filtering
            )
            
            # Score and rank candidates
            match_results = []
            for resume in similar_resumes:
                try:
                    match_result = self._calculate_match_score(
                        resume, job_description, required_skills, required_experience_years
                    )
                    match_results.append(match_result)
                except Exception as e:
                    logger.warning("Failed to score candidate", 
                                 candidate_id=resume.get('candidate_id'), error=str(e))
                    continue
            
            # Sort by final score and return top_k
            match_results.sort(key=lambda x: x.final_score, reverse=True)
            final_results = match_results[:top_k]
            
            logger.info("Resume matching completed", 
                       total_candidates=len(similar_resumes),
                       final_results=len(final_results))
            
            return final_results
            
        except Exception as e:
            logger.error("Failed to match resumes", error=str(e))
            raise
    
    def _calculate_match_score(self, 
                              resume: Dict[str, Any], 
                              job_description: str,
                              required_skills: List[str],
                              required_experience_years: Optional[int]) -> MatchResult:
        """Calculate comprehensive match score for a resume"""
        
        # Extract resume metadata
        metadata = resume.get('metadata', {})
        candidate_skills = metadata.get('skills', [])
        candidate_experience = metadata.get('experience_years', 0)
        
        # 1. Semantic score (from ChromaDB similarity)
        semantic_score = resume.get('similarity', 0.0)
        
        # 2. Keyword/Skills score
        keyword_score = self._calculate_keyword_score(required_skills, candidate_skills)
        
        # 3. Experience score
        experience_score = self._calculate_experience_score(
            required_experience_years, candidate_experience
        )
        
        # 4. Calculate final weighted score
        final_score = (
            self.semantic_weight * semantic_score +
            self.keyword_weight * keyword_score +
            self.experience_weight * experience_score
        )
        
        # Find matched skills
        matched_skills = self._find_matched_skills(required_skills, candidate_skills)
        
        # Generate explanation
        explanation = self._generate_explanation(
            semantic_score, keyword_score, experience_score, matched_skills
        )
        
        return MatchResult(
            candidate_id=resume['candidate_id'],
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            experience_score=experience_score,
            final_score=final_score,
            matched_skills=matched_skills,
            explanation=explanation,
            metadata=metadata
        )
    
    def _calculate_keyword_score(self, required_skills: List[str], 
                               candidate_skills: List[str]) -> float:
        """Calculate keyword matching score"""
        if not required_skills:
            return 1.0
        
        if not candidate_skills:
            return 0.0
        
        # Convert to lowercase for comparison
        required_lower = [skill.lower() for skill in required_skills]
        candidate_lower = [skill.lower() for skill in candidate_skills]
        
        # Count matches
        matches = sum(1 for skill in required_lower if skill in candidate_lower)
        
        # Calculate score
        score = matches / len(required_skills)
        return min(score, 1.0)
    
    def _calculate_experience_score(self, required_years: Optional[int], 
                                  candidate_years: Optional[int]) -> float:
        """Calculate experience matching score"""
        if required_years is None:
            return 1.0
        
        if candidate_years is None:
            return 0.5  # Neutral score if experience not specified
        
        if candidate_years >= required_years:
            return 1.0
        else:
            # Gradual scoring for candidates with less experience
            return max(0.0, candidate_years / required_years)
    
    def _find_matched_skills(self, required_skills: List[str], 
                           candidate_skills: List[str]) -> List[str]:
        """Find skills that match between job requirements and candidate"""
        if not required_skills or not candidate_skills:
            return []
        
        required_lower = {skill.lower(): skill for skill in required_skills}
        candidate_lower = [skill.lower() for skill in candidate_skills]
        
        matches = []
        for candidate_skill_lower in candidate_lower:
            if candidate_skill_lower in required_lower:
                matches.append(required_lower[candidate_skill_lower])
        
        return list(set(matches))  # Remove duplicates
    
    def _extract_job_skills(self, job_description: str) -> List[str]:
        """Extract technical skills from job description"""
        # Technical skills list
        TECHNICAL_SKILLS = [
            # Programming Languages
            'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift',
            'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell', 'bash', 'powershell',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express', 'django',
            'flask', 'fastapi', 'spring', 'laravel', 'rails', 'asp.net',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            'oracle', 'sqlite', 'dynamodb',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
            'jenkins', 'git', 'github', 'gitlab', 'ci/cd',
            
            # Data Science & ML
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            'matplotlib', 'seaborn', 'jupyter', 'spark', 'hadoop',
            
            # Others
            'linux', 'unix', 'rest api', 'graphql', 'microservices', 'agile', 'scrum'
        ]
        
        job_lower = job_description.lower()
        found_skills = []
        
        for skill in TECHNICAL_SKILLS:
            if skill.lower() in job_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _generate_explanation(self, semantic_score: float, keyword_score: float, 
                            experience_score: float, matched_skills: List[str]) -> str:
        """Generate human-readable explanation for the match"""
        explanation_parts = []
        
        # Semantic similarity
        if semantic_score >= 0.8:
            explanation_parts.append(f"High semantic similarity ({semantic_score:.2f})")
        elif semantic_score >= 0.6:
            explanation_parts.append(f"Good semantic similarity ({semantic_score:.2f})")
        else:
            explanation_parts.append(f"Moderate semantic similarity ({semantic_score:.2f})")
        
        # Keyword matching
        if matched_skills:
            skills_text = ", ".join(matched_skills[:5])  # Show top 5 skills
            if len(matched_skills) > 5:
                skills_text += f" and {len(matched_skills) - 5} more"
            explanation_parts.append(f"Matched skills: {skills_text}")
            explanation_parts.append(f"Keyword score: {keyword_score:.2f}")
        else:
            explanation_parts.append("No direct skill matches found")
        
        # Experience
        if experience_score >= 0.8:
            explanation_parts.append("Meets experience requirements")
        elif experience_score > 0:
            explanation_parts.append(f"Partial experience match ({experience_score:.2f})")
        
        return ". ".join(explanation_parts) + "."


# Global matcher service instance
matcher_service = ResumeMatcherService()
