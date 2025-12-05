"""Resume parser evaluation"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from app.services.langchain_agents import LangChainAgents
from app.evaluation.metrics import (
    field_accuracy,
    skill_extraction_recall,
    skill_extraction_precision,
    completeness_score,
    f1_score
)


def load_test_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """Load test dataset from JSON file"""
    dataset_path = Path(__file__).parent.parent / "datasets" / f"{dataset_name}.json"
    
    with open(dataset_path, 'r') as f:
        return json.load(f)


async def evaluate_resume_parser(verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate resume parser on test dataset.
    
    Args:
        verbose: If True, print detailed results for each test case
        
    Returns:
        Evaluation results with metrics and per-case breakdown
    """
    print("=" * 80)
    print("RESUME PARSER EVALUATION")
    print("=" * 80)
    
    # Load test cases
    test_cases = load_test_dataset("resume_parsing")
    print(f"\nLoaded {len(test_cases)} test cases")
    
    # Initialize agents
    agents = LangChainAgents()
    
    # Required fields for completeness check
    required_fields = [
        "profile.name",
        "profile.email",
        "experience.total_years",
        "skills.technical"
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing: {case['description']}")
        
        try:
            # Parse resume
            parsed = await agents.parse_resume(case["input"])
            parsed_dict = {
                "profile": parsed.profile.__dict__ if parsed.profile else {},
                "experience": parsed.experience.__dict__ if parsed.experience else {},
                "skills": parsed.skills.__dict__ if parsed.skills else {}
            }
            
            # Calculate metrics
            profile_accuracy = field_accuracy(
                parsed_dict.get("profile", {}),
                case["expected"].get("profile", {})
            )
            
            exp_accuracy = field_accuracy(
                parsed_dict.get("experience", {}),
                case["expected"].get("experience", {})
            )
            
            predicted_skills = parsed_dict.get("skills", {}).get("technical", [])
            expected_skills = case["expected"].get("skills", {}).get("technical", [])
            
            skill_recall = skill_extraction_recall(predicted_skills, expected_skills)
            skill_precision = skill_extraction_precision(predicted_skills, expected_skills)
            skill_f1 = f1_score(skill_precision, skill_recall)
            
            completeness = completeness_score(parsed_dict, required_fields)
            
            # Overall score (weighted average)
            overall_score = (
                profile_accuracy * 0.3 +
                exp_accuracy * 0.2 +
                skill_f1 * 0.3 +
                completeness * 0.2
            )
            
            result = {
                "case_id": case["id"],
                "description": case["description"],
                "metrics": {
                    "profile_accuracy": profile_accuracy,
                    "experience_accuracy": exp_accuracy,
                    "skill_recall": skill_recall,
                    "skill_precision": skill_precision,
                    "skill_f1": skill_f1,
                    "completeness": completeness,
                    "overall_score": overall_score
                },
                "passed": overall_score >= 0.7,
                "parsed_data": parsed_dict
            }
            
            results.append(result)
            
            # Print results
            print(f"  ✓ Profile Accuracy: {profile_accuracy:.2%}")
            print(f"  ✓ Experience Accuracy: {exp_accuracy:.2%}")
            print(f"  ✓ Skill F1: {skill_f1:.2%} (P: {skill_precision:.2%}, R: {skill_recall:.2%})")
            print(f"  ✓ Completeness: {completeness:.2%}")
            print(f"  {'✅ PASS' if result['passed'] else '❌ FAIL'} Overall: {overall_score:.2%}")
            
            if verbose:
                print(f"\n  Parsed Skills: {predicted_skills}")
                print(f"  Expected Skills: {expected_skills}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            results.append({
                "case_id": case["id"],
                "description": case["description"],
                "error": str(e),
                "passed": False
            })
    
    # Calculate aggregate metrics
    successful_results = [r for r in results if "error" not in r]
    
    if successful_results:
        avg_metrics = {
            "profile_accuracy": sum(r["metrics"]["profile_accuracy"] for r in successful_results) / len(successful_results),
            "experience_accuracy": sum(r["metrics"]["experience_accuracy"] for r in successful_results) / len(successful_results),
            "skill_f1": sum(r["metrics"]["skill_f1"] for r in successful_results) / len(successful_results),
            "completeness": sum(r["metrics"]["completeness"] for r in successful_results) / len(successful_results),
            "overall_score": sum(r["metrics"]["overall_score"] for r in successful_results) / len(successful_results)
        }
    else:
        avg_metrics = {}
    
    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal Test Cases: {len(test_cases)}")
    print(f"Passed: {sum(1 for r in results if r['passed'])}")
    print(f"Failed: {sum(1 for r in results if not r['passed'])}")
    print(f"Pass Rate: {pass_rate:.2%}")
    
    if avg_metrics:
        print(f"\nAverage Metrics:")
        print(f"  Profile Accuracy: {avg_metrics['profile_accuracy']:.2%}")
        print(f"  Experience Accuracy: {avg_metrics['experience_accuracy']:.2%}")
        print(f"  Skill F1 Score: {avg_metrics['skill_f1']:.2%}")
        print(f"  Completeness: {avg_metrics['completeness']:.2%}")
        print(f"  Overall Score: {avg_metrics['overall_score']:.2%}")
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"resume_parser_eval_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_cases": len(test_cases),
            "pass_rate": pass_rate,
            "average_metrics": avg_metrics,
            "results": results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return {
        "pass_rate": pass_rate,
        "average_metrics": avg_metrics,
        "results": results
    }


if __name__ == "__main__":
    asyncio.run(evaluate_resume_parser(verbose=True))
