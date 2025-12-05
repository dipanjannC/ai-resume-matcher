# AI Resume Matcher - Development Guidelines

## 🏗️ Development Principles

### SOLID Principles

#### 1. **Single Responsibility Principle (SRP)**
- Each class/module should have one reason to change
- **Example**: `ResumeProcessor` handles orchestration, `LangChainAgents` handles AI parsing
- Keep services focused on their domain (resumes, jobs, matching, memory)

#### 2. **Open/Closed Principle (OCP)**
- Open for extension, closed for modification
- **Example**: Memory providers (Mem0, Graphiti) extend `MemoryStore` base class
- Use abstract base classes for extensibility

#### 3. **Liskov Substitution Principle (LSP)**
- Derived classes must be substitutable for their base classes
- **Example**: Any `MemoryStore` implementation can replace another without breaking code
- Maintain consistent interfaces across implementations

#### 4. **Interface Segregation Principle (ISP)**
- Clients shouldn't depend on interfaces they don't use
- **Example**: Separate concerns between parsing, storage, and matching
- Keep service interfaces focused and minimal

#### 5. **Dependency Inversion Principle (DIP)**
- Depend on abstractions, not concretions
- **Example**: `ResumeProcessor` depends on `LangChainAgents` interface, not specific LLM implementations
- Use dependency injection where appropriate

### DRY (Don't Repeat Yourself)
- Avoid code duplication
- **Example**: `_execute_with_fallback` method centralizes LLM retry logic
- Extract common patterns into reusable functions/methods
- Use configuration files for repeated values

### KISS (Keep It Simple, Stupid)
- Favor simplicity over complexity
- **Example**: File-based storage instead of complex database setup
- Write clear, readable code over clever solutions
- Use straightforward algorithms unless performance requires optimization

## 🧪 Evaluation Driven Development (EDD) for LangChain Agents

### Overview
Evaluation Driven Development ensures LangChain agents produce consistent, high-quality outputs by continuously measuring and improving their performance.

### Core Principles

#### 1. **Define Success Metrics**
Before implementing an agent, define what "good" looks like:
- **Accuracy**: Does the parsed data match ground truth?
- **Completeness**: Are all required fields extracted?
- **Consistency**: Does the same input produce the same output?
- **Latency**: How fast does the agent respond?

#### 2. **Create Evaluation Datasets**
Build test datasets for each agent:

```python
# Example: Resume parsing evaluation dataset
resume_eval_dataset = [
    {
        "input": "John Doe resume text...",
        "expected_output": {
            "profile": {"name": "John Doe", "title": "Software Engineer"},
            "experience": {"total_years": 5},
            "skills": {"technical": ["Python", "JavaScript"]}
        }
    },
    # More examples...
]
```

#### 3. **Implement Evaluation Functions**
Create automated evaluation scripts:

```python
# app/evaluation/resume_parser_eval.py
async def evaluate_resume_parser(test_cases):
    """Evaluate resume parsing accuracy"""
    results = []
    for case in test_cases:
        parsed = await langchain_agents.parse_resume(case["input"])
        score = calculate_similarity(parsed, case["expected_output"])
        results.append({"input": case["input"], "score": score})
    return results
```

#### 4. **Continuous Evaluation**
- Run evaluations on every prompt change
- Track metrics over time
- Use LangSmith or similar tools for monitoring

#### 5. **Iterative Improvement**
Based on evaluation results:
1. **Identify failure patterns**
2. **Update prompts** to address failures
3. **Re-evaluate** to measure improvement
4. **Repeat** until metrics meet targets

### Implementation Guide

#### Step 1: Create Evaluation Directory
```
app/evaluation/
├── __init__.py
├── datasets/
│   ├── resume_parsing.json
│   ├── job_parsing.json
│   └── matching.json
├── evaluators/
│   ├── resume_parser_eval.py
│   ├── job_parser_eval.py
│   └── matching_eval.py
└── run_evaluations.py
```

#### Step 2: Define Evaluation Metrics

```python
# app/evaluation/metrics.py
from typing import Dict, Any

def field_accuracy(predicted: Dict, expected: Dict) -> float:
    """Calculate field-level accuracy"""
    correct = sum(1 for k in expected if predicted.get(k) == expected[k])
    return correct / len(expected)

def skill_extraction_recall(predicted_skills: list, expected_skills: list) -> float:
    """Calculate recall for skill extraction"""
    if not expected_skills:
        return 1.0
    found = sum(1 for skill in expected_skills if skill in predicted_skills)
    return found / len(expected_skills)
```

#### Step 3: Create Evaluation Runner

```python
# app/evaluation/run_evaluations.py
import asyncio
from app.services.langchain_agents import LangChainAgents
from app.evaluation.datasets import load_dataset
from app.evaluation.metrics import field_accuracy

async def run_resume_parser_evaluation():
    """Run comprehensive resume parser evaluation"""
    agents = LangChainAgents()
    test_cases = load_dataset("resume_parsing")
    
    results = []
    for case in test_cases:
        try:
            parsed = await agents.parse_resume(case["input"])
            accuracy = field_accuracy(parsed.dict(), case["expected"])
            results.append({
                "case_id": case["id"],
                "accuracy": accuracy,
                "passed": accuracy >= 0.8
            })
        except Exception as e:
            results.append({
                "case_id": case["id"],
                "error": str(e),
                "passed": False
            })
    
    # Generate report
    total_accuracy = sum(r.get("accuracy", 0) for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    
    print(f"Resume Parser Evaluation Results:")
    print(f"  Total Accuracy: {total_accuracy:.2%}")
    print(f"  Pass Rate: {pass_rate:.2%}")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_resume_parser_evaluation())
```

#### Step 4: Integrate with CI/CD

```yaml
# .github/workflows/evaluate.yml
name: LangChain Agent Evaluation

on:
  pull_request:
    paths:
      - 'app/services/langchain_agents.py'
      - 'app/evaluation/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: uv pip install -r requirements.txt
      - name: Run evaluations
        run: python app/evaluation/run_evaluations.py
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
      - name: Check thresholds
        run: |
          # Fail if accuracy < 80%
          python -c "import json; results = json.load(open('eval_results.json')); exit(0 if results['accuracy'] >= 0.8 else 1)"
```

### Best Practices

1. **Version Control Datasets**: Track evaluation datasets in git
2. **Diverse Test Cases**: Include edge cases, common patterns, and failure scenarios
3. **Automated Regression Testing**: Run evaluations on every agent change
4. **Monitor Production**: Use LangSmith to track real-world performance
5. **Iterative Refinement**: Continuously add failing cases to evaluation dataset
6. **Document Prompt Changes**: Track why prompts were changed and their impact

### Tools & Resources

- **LangSmith**: LangChain's evaluation and monitoring platform
- **LangChain Evaluators**: Built-in evaluation tools
- **Custom Metrics**: Domain-specific evaluation functions
- **A/B Testing**: Compare different prompt versions

## 📝 Code Style Guidelines

### General
- Use type hints for all function parameters and return values
- Write docstrings for all public methods
- Keep functions under 50 lines when possible
- Use meaningful variable names

### LangChain Specific
- Always handle LLM failures gracefully
- Implement retry logic with exponential backoff
- Log all LLM interactions for debugging
- Use structured output (Pydantic models) over raw text parsing

### Testing
- Write unit tests for business logic
- Create integration tests for LangChain agents
- Use evaluation datasets for agent testing
- Mock external API calls in tests

## 🔄 Development Workflow

1. **Define Requirements**: What should the agent do?
2. **Create Evaluation Dataset**: Build test cases
3. **Implement Agent**: Write the LangChain agent code
4. **Evaluate**: Run evaluation suite
5. **Iterate**: Improve based on results
6. **Deploy**: Ship when metrics meet targets
7. **Monitor**: Track production performance
