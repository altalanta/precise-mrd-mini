"""Script to profile the Precise MRD pipeline."""

import cProfile
import pstats
from pathlib import Path

REPORTS_DIR = Path("reports")
PROFILING_DIR = REPORTS_DIR / "profiling"


def run_profiler():
    """Run the smoke pipeline under the cProfile profiler."""
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)
    profile_output = PROFILING_DIR / "pipeline.prof"

    # We use the cProfile runctx function to execute the Click command
    # This is a clean way to profile a command-line application
    command = "smoke --no-cache"
    cProfile.runctx(
        f"main('{command}'.split())", globals(), locals(), str(profile_output)
    )

    print(f"Profiling data saved to: {profile_output}")

    # Print out the top 20 functions by cumulative time
    print("\n--- Top 20 functions by cumulative time ---")
    p = pstats.Stats(str(profile_output))
    p.sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    run_profiler()
