"""Knowledge Graph Service for Job Application Tracking"""

import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict
import networkx as nx
from pyvis.network import Network


class KnowledgeGraphService:
    """Build and visualize temporal knowledge graph from job descriptions"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def extract_entities_from_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities from a job description.
        
        Returns:
            Dict with extracted entities: skills, topics, company, role, etc.
        """
        entities = {
            "job_id": job_data.get("id"),
            "title": job_data.get("title", ""),
            "company": job_data.get("company", ""),
            "location": job_data.get("location", ""),
            "created_at": job_data.get("created_at", ""),
            "skills": set(),
            "topics": set(),
            "experience_level": self._categorize_experience(job_data.get("experience_years", 0))
        }
        
        # Extract skills
        required_skills = job_data.get("required_skills", [])
        preferred_skills = job_data.get("preferred_skills", [])
        entities["skills"] = set(required_skills + preferred_skills)
        
        # Extract topics from title and responsibilities
        entities["topics"] = self._extract_topics(job_data)
        
        return entities
    
    def _categorize_experience(self, years: int) -> str:
        """Categorize experience level"""
        if years == 0:
            return "Entry Level"
        elif years <= 2:
            return "Junior"
        elif years <= 5:
            return "Mid-Level"
        elif years <= 10:
            return "Senior"
        else:
            return "Lead/Principal"
    
    def _extract_topics(self, job_data: Dict[str, Any]) -> Set[str]:
        """Extract topics from job data"""
        topics = set()
        
        title = job_data.get("title", "").lower()
        responsibilities = " ".join(job_data.get("responsibilities", [])).lower()
        
        # Topic keywords
        topic_keywords = {
            "Machine Learning": ["machine learning", "ml", "deep learning", "neural network", "ai"],
            "Data Science": ["data science", "analytics", "data analysis", "statistics"],
            "Web Development": ["web development", "frontend", "backend", "full stack", "web app"],
            "Mobile Development": ["mobile", "ios", "android", "react native", "flutter"],
            "DevOps": ["devops", "ci/cd", "kubernetes", "docker", "infrastructure"],
            "Cloud Computing": ["cloud", "aws", "azure", "gcp", "serverless"],
            "Security": ["security", "cybersecurity", "penetration testing", "encryption"],
            "Database": ["database", "sql", "nosql", "postgresql", "mongodb"],
            "API Development": ["api", "rest", "graphql", "microservices"],
            "UI/UX": ["ui", "ux", "design", "user experience", "user interface"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in title or keyword in responsibilities for keyword in keywords):
                topics.add(topic)
        
        return topics
    
    def build_graph_from_jobs(self, jobs: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build knowledge graph from list of jobs.
        
        Graph structure:
        - Job nodes (with temporal info)
        - Skill nodes
        - Topic nodes
        - Company nodes
        - Edges: job->requires->skill, job->in->topic, job->at->company
        """
        self.graph.clear()
        
        for job in jobs:
            entities = self.extract_entities_from_job(job)
            
            job_id = entities["job_id"]
            job_title = entities["title"]
            
            # Add job node
            self.graph.add_node(
                job_id,
                type="job",
                title=job_title,
                company=entities["company"],
                location=entities["location"],
                created_at=entities["created_at"],
                experience_level=entities["experience_level"],
                label=f"{job_title}\n{entities['company']}"
            )
            
            # Add company node and edge
            if entities["company"]:
                company_id = f"company_{entities['company']}"
                if not self.graph.has_node(company_id):
                    self.graph.add_node(
                        company_id,
                        type="company",
                        name=entities["company"],
                        label=entities["company"]
                    )
                self.graph.add_edge(job_id, company_id, relationship="at")
            
            # Add skill nodes and edges
            for skill in entities["skills"]:
                if skill and skill.strip():
                    skill_id = f"skill_{skill.lower().strip()}"
                    if not self.graph.has_node(skill_id):
                        self.graph.add_node(
                            skill_id,
                            type="skill",
                            name=skill,
                            label=skill
                        )
                    self.graph.add_edge(job_id, skill_id, relationship="requires")
            
            # Add topic nodes and edges
            for topic in entities["topics"]:
                topic_id = f"topic_{topic.lower().replace(' ', '_')}"
                if not self.graph.has_node(topic_id):
                    self.graph.add_node(
                        topic_id,
                        type="topic",
                        name=topic,
                        label=topic
                    )
                self.graph.add_edge(job_id, topic_id, relationship="in")
        
        return self.graph
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "jobs": sum(1 for _, data in self.graph.nodes(data=True) if data.get("type") == "job"),
            "skills": sum(1 for _, data in self.graph.nodes(data=True) if data.get("type") == "skill"),
            "topics": sum(1 for _, data in self.graph.nodes(data=True) if data.get("type") == "topic"),
            "companies": sum(1 for _, data in self.graph.nodes(data=True) if data.get("type") == "company")
        }
        
        return stats
    
    def get_top_skills(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most in-demand skills based on job connections"""
        skill_counts = defaultdict(int)
        
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "skill":
                # Count incoming edges from jobs
                skill_counts[data.get("name")] = self.graph.in_degree(node)
        
        return sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_topic_distribution(self) -> Dict[str, int]:
        """Get distribution of jobs across topics"""
        topic_counts = defaultdict(int)
        
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "topic":
                # Count incoming edges from jobs
                topic_counts[data.get("name")] = self.graph.in_degree(node)
        
        return dict(topic_counts)
    
    def visualize_graph(self, output_path: str = "knowledge_graph.html", 
                       height: str = "750px", width: str = "100%") -> str:
        """
        Create interactive visualization using PyVis.
        
        Returns:
            Path to generated HTML file
        """
        net = Network(height=height, width=width, directed=True, notebook=False)
        
        # Configure physics
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100
          }
        }
        """)
        
        # Add nodes with colors based on type
        node_colors = {
            "job": "#3498db",      # Blue
            "skill": "#2ecc71",    # Green
            "topic": "#e74c3c",    # Red
            "company": "#f39c12"   # Orange
        }
        
        node_sizes = {
            "job": 25,
            "skill": 15,
            "topic": 20,
            "company": 20
        }
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            color = node_colors.get(node_type, "#95a5a6")
            size = node_sizes.get(node_type, 10)
            
            # Create tooltip
            title = f"<b>{data.get('label', node)}</b><br>"
            title += f"Type: {node_type}<br>"
            
            if node_type == "job":
                title += f"Company: {data.get('company', 'N/A')}<br>"
                title += f"Location: {data.get('location', 'N/A')}<br>"
                title += f"Experience: {data.get('experience_level', 'N/A')}<br>"
                title += f"Posted: {data.get('created_at', 'N/A')[:10]}"
            elif node_type == "skill":
                connections = self.graph.in_degree(node)
                title += f"Required by {connections} job(s)"
            
            net.add_node(
                node,
                label=data.get("label", node),
                color=color,
                size=size,
                title=title
            )
        
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            net.add_edge(source, target, title=data.get("relationship", ""))
        
        # Save
        net.save_graph(output_path)
        return output_path


# Singleton instance
knowledge_graph_service = KnowledgeGraphService()
