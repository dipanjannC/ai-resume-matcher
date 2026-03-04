# AI Resume Matcher: Resume Customization Workflow

## Overview
This document outlines the workflow and architecture for the **Resume Customization Engine**. The objective is to correctly parse an existing resume and a target Job Description (JD), and then intelligently customize the resume. The customization involves rephrasing experiences, highlighting relevant tools, and adjusting technologies to align with the job role, all while adhering to strictly professional standards and ATS (Applicant Tracking System) best practices.

## High-Level Architecture

The below diagram illustrates the high-level data flow from user inputs to the generated ATS-optimized resume.

```mermaid
graph TD
    A[User Input: Existing Resume PDF/DOCX] --> C(Data Ingestion & Parsing)
    B[User Input: Job Description Link/Text] --> C
    C -->|Extracted Profile & Skills| D[Semantic Matching & Gap Analysis]
    C -->|Extracted JD Requirements| D
    D -->|Analysis Data| E[AI Customization Engine]
    E -->|Rephrase Experience| F[Draft Generation]
    E -->|Align Tools & Tech| F
    F --> G[ATS Formatting & Validation]
    G --> H[Final Customized Resume PDF/DOCX]
    
    style A fill:#e1f5fe,stroke:#01579b
    style B fill:#e1f5fe,stroke:#01579b
    style E fill:#fff3e0,stroke:#e65100
    style H fill:#e8f5e9,stroke:#1b5e20
```

## Detailed Process Flow

### 1. Data Ingestion & Parsing
*   **Resume Parsing:** Read and extract text from the user's existing resume. We utilize LangChain agents and NLP to break down the resume into discrete, structured JSON components: Profile, Professional Summary, Work Experience, Education, and Skills.
*   **Job Description Parsing:** Scrape the provided JD link or parse the raw text. Extract key entities such as required skills, expected tools/technologies, experience level, and core responsibilities.

### 2. Semantic Analysis & Gap Identification
*   **Vector Matching:** Compare the resume's parsed data against the JD's requirements using vector embeddings (e.g., ChromaDB). 
*   **Skill Gap Analysis:** Identify targeted keywords, tools, or skills required by the JD that are underrepresented or missing in the candidate's existing resume.
*   **Customization Strategy:** Create a targeted mapping of how existing resume bullets can be naturally adapted to include the targeted keywords without fabricating experience.

### 3. AI Customization Engine (Rephrasing)
This is the core of the customization process. The strategic mapping is passed to an LLM (e.g., Groq/Gemini/OpenAI) using specialized system prompts. The prompts enforce strict constraints:
*   **No Hallucination:** Absolutely no fabrication of experience or skills the candidate does not actually possess. The goal is re-framing, not inventing.
*   **Action-Oriented Bullet Points:** Ensure bullet points start with strong action verbs.
*   **Quantifiable Metrics:** Retain and emphasize any numbers, percentages, or metrics from the original resume.
*   **Seamless Keyword Integration:** Naturally weave in JD-specific tools and technologies into existing experience bullets.
*   **Summary & Skills Adjustment:** Rewrite the professional summary to directly address the target role and reorganize the skills section to prioritize the technologies mentioned in the JD.

### 4. ATS Optimization & Validation
*   **Formatting Constraints:** Ensure the output uses a standard, single-column layout with clean text. This prevents tracking systems from failing to parse the document. Complex tables, internal columns, graphics, or unusual fonts are stripped out.
*   **Keyword Density Validation:** Verify that the generated text has a balanced keyword presence. ATS penalize obvious "keyword stuffing." The text must remain highly professional and read naturally to human recruiters.

## Customization Sequence Diagram

The sequence diagram below shows the interactions between the services during the resume customization process.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Parser as Parsing Service
    participant Matcher as Semantic Matcher
    participant AI as Generative LLM
    participant ATS as ATS Validator
    
    User->>Parser: Submit Resume & JD Link
    
    rect rgb(240, 248, 255)
        Note over Parser: Phase 1: Ingestion
        Parser->>Parser: Extract Resume Sections (JSON)
        Parser->>Parser: Extract JD Requirements (JSON)
    end
    
    Parser->>Matcher: Send Structured Data
    
    rect rgb(255, 250, 240)
        Note over Matcher: Phase 2: Analysis
        Matcher->>Matcher: Vector Search (ChromaDB)
        Matcher->>Matcher: Identify Skill Gaps & Alignments
    end
    
    Matcher->>AI: Send Strategy & Resume Content
    
    rect rgb(240, 255, 240)
        Note over AI: Phase 3: Customization
        AI->>AI: Rephrase Experience & Summary
        AI->>AI: Highlight Relevant Tools
    end
    
    AI->>ATS: Send Draft Resume
    
    rect rgb(255, 240, 245)
        Note over ATS: Phase 4: Validation
        ATS->>ATS: Validate ATS Best Practices
        ATS->>ATS: Check Keyword Density (No 'Noise')
    end
    
    ATS-->>User: Return Professionally Customized Resume
```

## ATS Best Practices Adherence
To ensure maximum success rate through applicant tracking systems, the workflow adheres to the following rules:
*   **Standard Headings:** Conventional section titles are enforced (e.g., "Work Experience", "Education", "Skills") instead of creative variations like "My Journey".
*   **Clean Output Formatting:** The final output is generated into standard Markdown, which accurately translates into clean plain-text, PDF, or DOCX formats favoring ATS parsers. 
*   **Concise Language (No Noise):** Buzzwords that add no value are eliminated. The focus remains entirely on quantifiable achievements and relevant tool usage requested by the JD.
