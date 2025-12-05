"""Main evaluation runner for all LangChain agents"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.evaluation.evaluators.resume_parser_eval import evaluate_resume_parser


async def run_all_evaluations():
    """Run all evaluation suites"""
    print("\n🧪 Running All Evaluations\n")
    
    results = {}
    
    # Resume Parser Evaluation
    print("\n" + "=" * 80)
    print("1. RESUME PARSER EVALUATION")
    print("=" * 80)
    resume_results = await evaluate_resume_parser(verbose=False)
    results["resume_parser"] = resume_results
    
    # Future evaluations can be added here
    # job_results = await evaluate_job_parser()
    # matching_results = await evaluate_matching()
    
    # Overall summary
    print("\n\n" + "=" * 80)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 80)
    
    for eval_name, eval_results in results.items():
        pass_rate = eval_results.get("pass_rate", 0)
        status = "✅ PASS" if pass_rate >= 0.7 else "❌ FAIL"
        print(f"\n{eval_name.replace('_', ' ').title()}: {status}")
        print(f"  Pass Rate: {pass_rate:.2%}")
        
        if "average_metrics" in eval_results and eval_results["average_metrics"]:
            avg_score = eval_results["average_metrics"].get("overall_score", 0)
            print(f"  Average Score: {avg_score:.2%}")
    
    # Determine overall pass/fail
    overall_pass = all(r.get("pass_rate", 0) >= 0.7 for r in results.values())
    
    print("\n" + "=" * 80)
    if overall_pass:
        print("✅ ALL EVALUATIONS PASSED")
    else:
        print("❌ SOME EVALUATIONS FAILED")
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_evaluations())
