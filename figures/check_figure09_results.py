"""Check the manuscript's regional sample and summer attribution claims."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import ATTRIBUTION_GROUPS, FOCUS_REGIONS, REGIONAL_ATTRIBUTION_PATH, regional_vector


def check_figure09_results(root: Path) -> None:
    data = pd.read_csv(root / REGIONAL_ATTRIBUTION_PATH)
    expected_counts = [38, 14, 41, 66, 38]
    temperature_index = ATTRIBUTION_GROUPS.index("Temperature / melt")
    failures = []
    for season in ("winter", "summer"):
        frame = data.loc[data["season"].eq(season)]
        counts = regional_vector(frame, "prediction", "n_glaciers")
        if not np.array_equal(counts, expected_counts):
            failures.append(f"{season.title()} glacier counts in RGI regions {FOCUS_REGIONS}: "
                            f"{counts.tolist()}, expected {expected_counts}")
        if season == "summer":
            for column, label in (("delta", "all glaciers"), ("leave_largest_out", "largest glacier excluded")):
                contributions = np.array([regional_vector(frame, group, column) for group in ATTRIBUTION_GROUPS])
                temperature = contributions[temperature_index]
                other_groups = np.delete(contributions, temperature_index, axis=0)
                failed_regions = (temperature >= 0) | (temperature >= other_groups.min(axis=0))
                if failed_regions.any():
                    regions = np.array(FOCUS_REGIONS)[failed_regions].tolist()
                    failures.append(f"Summer temperature / melt is not the most negative contribution "
                                    f"in RGI regions {regions} ({label}).")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Figure 9 claims match the 197-glacier cohort and the summer contributions, "
          "including after excluding each region's largest glacier.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project directory containing the regional attribution results.")
    check_figure09_results(parser.parse_args().root.expanduser().resolve())
