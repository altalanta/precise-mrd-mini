#!/usr/bin/env python3
"""
Integration Test Runner for Precise MRD Pipeline

This script runs comprehensive integration tests for the Precise MRD pipeline,
including full pipeline tests, contamination analysis, stratified analysis,
and deterministic behavior validation.
"""

import argparse
import subprocess
import sys
import time


def run_command(cmd: str, description: str = "") -> bool:
    """Run a shell command and return success status."""
    print(f"🧪 {description or cmd}")
    print(f"   Running: {cmd}")

    try:
        start_time = time.time()
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        end_time = time.time()

        print(f"   ✅ Success in {end_time - start_time:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed in {time.time() - start_time:.2f}s")
        print(f"   Error: {e.stderr}")
        return False


def run_quick_integration_tests():
    """Run a quick subset of integration tests."""
    print("🚀 Running Quick Integration Tests...")
    print("=" * 50)

    tests = [
        (
            "Basic Pipeline Test",
            "pytest tests/integration/test_full_pipeline.py::TestFullPipelineIntegration::test_smoke_test_configuration -v",
        ),
        (
            "Configuration Validation",
            "pytest tests/integration/test_full_pipeline.py::TestPipelineRobustness::test_invalid_configurations -v",
        ),
        (
            "Determinism Test",
            "pytest tests/integration/test_deterministic_behavior.py::TestDeterministicBehavior::test_identical_results_same_seed -v",
        ),
    ]

    passed = 0
    total = len(tests)

    for description, cmd in tests:
        if run_command(cmd, description):
            passed += 1

    print(f"\n📊 Quick Tests Summary: {passed}/{total} passed")
    return passed == total


def run_contamination_tests():
    """Run contamination-specific integration tests."""
    print("🦠 Running Contamination Tests...")
    print("=" * 50)

    tests = [
        (
            "Index Hopping Test",
            "pytest tests/integration/test_contamination_scenarios.py::TestContaminationIntegration::test_index_hopping_simulation -v",
        ),
        (
            "Carryover Test",
            "pytest tests/integration/test_contamination_scenarios.py::TestContaminationIntegration::test_sample_carryover_contamination -v",
        ),
        (
            "UMI Deduplication Test",
            "pytest tests/integration/test_contamination_scenarios.py::TestContaminationMitigation::test_umi_deduplication_effectiveness -v",
        ),
    ]

    passed = 0
    total = len(tests)

    for description, cmd in tests:
        if run_command(cmd, description):
            passed += 1

    print(f"\n📊 Contamination Tests Summary: {passed}/{total} passed")
    return passed == total


def run_stratified_tests():
    """Run stratified analysis integration tests."""
    print("📊 Running Stratified Analysis Tests...")
    print("=" * 50)

    tests = [
        (
            "Context Stratification",
            "pytest tests/integration/test_stratified_analysis.py::TestStratifiedAnalysis::test_trinucleotide_context_stratification -v",
        ),
        (
            "Depth Stratification",
            "pytest tests/integration/test_stratified_analysis.py::TestStratifiedAnalysis::test_depth_stratification -v",
        ),
        (
            "Power Analysis",
            "pytest tests/integration/test_stratified_analysis.py::TestPowerAnalysisStratification::test_power_by_depth_stratum -v",
        ),
    ]

    passed = 0
    total = len(tests)

    for description, cmd in tests:
        if run_command(cmd, description):
            passed += 1

    print(f"\n📊 Stratified Tests Summary: {passed}/{total} passed")
    return passed == total


def run_deterministic_tests():
    """Run deterministic behavior validation tests."""
    print("🔄 Running Deterministic Behavior Tests...")
    print("=" * 50)

    tests = [
        (
            "Seed Reproducibility",
            "pytest tests/integration/test_deterministic_behavior.py::TestDeterministicBehavior::test_identical_results_same_seed -v",
        ),
        (
            "Different Seeds",
            "pytest tests/integration/test_deterministic_behavior.py::TestDeterministicBehavior::test_different_seeds_produce_different_results -v",
        ),
        (
            "Hash Stability",
            "pytest tests/integration/test_deterministic_behavior.py::TestDeterministicBehavior::test_artifact_hash_stability -v",
        ),
        (
            "Regression Test",
            "pytest tests/integration/test_deterministic_behavior.py::TestDeterminismRegression::test_known_good_configuration -v",
        ),
    ]

    passed = 0
    total = len(tests)

    for description, cmd in tests:
        if run_command(cmd, description):
            passed += 1

    print(f"\n📊 Deterministic Tests Summary: {passed}/{total} passed")
    return passed == total


def run_full_integration_suite():
    """Run the complete integration test suite."""
    print("🚀 Running Full Integration Test Suite...")
    print("=" * 50)

    test_suites = [
        ("Quick Tests", run_quick_integration_tests),
        ("Contamination Tests", run_contamination_tests),
        ("Stratified Tests", run_stratified_tests),
        ("Deterministic Tests", run_deterministic_tests),
    ]

    total_passed = 0
    total_tests = 0

    for suite_name, suite_func in test_suites:
        print(f"\n{'=' * 20} {suite_name} {'=' * 20}")
        if suite_func():
            total_passed += 1
        total_tests += 1

    print(f"\n{'=' * 50}")
    print(
        f"🎯 Integration Test Suite Summary: {total_passed}/{total_tests} test suites passed",
    )

    return total_passed == total_tests


def run_performance_tests():
    """Run performance regression tests."""
    print("⚡ Running Performance Tests...")
    print("=" * 50)

    tests = [
        (
            "Performance Matrix",
            "pytest tests/integration/test_full_pipeline.py::TestPerformanceRegression::test_configuration_performance_matrix -v",
        ),
        (
            "Parameter Sensitivity",
            "pytest tests/integration/test_deterministic_behavior.py::TestDeterminismRegression::test_parameter_sensitivity -v",
        ),
    ]

    passed = 0
    total = len(tests)

    for description, cmd in tests:
        if run_command(cmd, description):
            passed += 1

    print(f"\n📊 Performance Tests Summary: {passed}/{total} passed")
    return passed == total


def run_specific_test_category(category: str):
    """Run tests for a specific category."""
    categories = {
        "quick": run_quick_integration_tests,
        "contamination": run_contamination_tests,
        "stratified": run_stratified_tests,
        "deterministic": run_deterministic_tests,
        "performance": run_performance_tests,
        "full": run_full_integration_suite,
    }

    if category not in categories:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {list(categories.keys())}")
        return False

    return categories[category]()


def main():
    """Main CLI interface for integration tests."""
    parser = argparse.ArgumentParser(
        description="Run integration tests for Precise MRD",
    )
    parser.add_argument(
        "category",
        nargs="?",
        default="quick",
        choices=[
            "quick",
            "contamination",
            "stratified",
            "deterministic",
            "performance",
            "full",
        ],
        help="Test category to run",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test categories",
    )

    args = parser.parse_args()

    if args.list:
        print("Available integration test categories:")
        print("- quick: Fast subset of integration tests")
        print("- contamination: Contamination analysis tests")
        print("- stratified: Stratified analysis tests")
        print("- deterministic: Deterministic behavior tests")
        print("- performance: Performance regression tests")
        print("- full: Complete integration test suite")
        return 0

    print(f"Starting integration tests for category: {args.category}")
    print("Test files location: tests/integration/")

    success = run_specific_test_category(args.category)

    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
