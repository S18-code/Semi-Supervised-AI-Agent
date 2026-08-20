"""
ATSC 3.0 Update Broadcast Agent — Data Generation (Steps 1-4), v2
================================================================
Same 4 steps as before, but instead of one bundled "profile" output, each
of the 6 transmission parameters is chosen INDEPENDENTLY, using its own
threshold function. Each output column can now be modeled/predicted
separately later (multi-output classification).

Outputs (per update):
  1. plp                   -- which Physical Layer Pipe
  2. modulation             -- QPSK / 16QAM / 64QAM / 256QAM / 1024QAM / 4096QAM
  3. code_rate               -- one of the 12 LDPC code rates
  4. ldpc_length             -- NORMAL_16200 / LONG_64800
  5. al_fec_redundancy_pct   -- application-layer FEC repair overhead
  6. carousel_repeat_sec     -- how often the file object re-broadcasts

NOTE: because these are chosen independently, it's possible to generate
combinations that don't make practical sense (e.g. robust modulation +
zero AL-FEC). The safety-floor guardrails below only protect the
critical-update MINIMUMS per variable -- they don't prevent every
possible weird combination on the non-critical side. Worth reviewing
the labeled output for sanity as you tune thresholds.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: FEATURE SCHEMA (inputs) -- unchanged
# ---------------------------------------------------------------------------
CRITICAL_ECUS = ["braking", "steering", "airbag", "powertrain_safety"]
NON_CRITICAL_ECUS = ["infotainment", "climate_control", "telematics", "seat_memory"]


# ---------------------------------------------------------------------------
# STEP 2: OUTPUT OPTIONS (ordered most -> least robust, where applicable)
# ---------------------------------------------------------------------------
PLP_OPTIONS = ["PLP_CRITICAL", "PLP_STANDARD", "PLP_BULK"]

# Ordered most -> least robust
MODULATION_OPTIONS = ["QPSK", "16QAM", "64QAM", "256QAM", "1024QAM", "4096QAM"]

# Ordered most -> least robust (subset of the 12 standard LDPC rates,
# expressed as k/15 per ATSC A/322 style notation)
CODE_RATE_OPTIONS = ["2/15", "5/15", "7/15", "9/15", "11/15", "13/15"]

LDPC_LENGTH_OPTIONS = ["LONG_64800", "NORMAL_16200"]  # long = more robust


# ---------------------------------------------------------------------------
# STEP 3: INDEPENDENT THRESHOLD FUNCTIONS -- ONE PER PARAMETER
# ---------------------------------------------------------------------------
# Edit the thresholds inside each function to match your actual robustness
# requirements. Each function only looks at (update_type, priority, size_mb)
# and decides ITS OWN parameter -- they don't reference each other.

def choose_plp(update_type: str, priority: int, size_mb: float) -> str:
    if update_type == "critical":
        return "PLP_CRITICAL"
    if priority >= 3 or size_mb > 100:
        return "PLP_STANDARD"
    return "PLP_BULK"


def choose_modulation(update_type: str, priority: int, size_mb: float) -> str:
    if update_type == "critical":
        return "QPSK" if (priority >= 4 or size_mb > 50) else "16QAM"
    # non_critical
    if priority >= 4:
        return "64QAM"
    if priority == 3:
        return "64QAM" if size_mb > 20 else "256QAM"
    if priority == 2:
        return "256QAM"
    return "1024QAM" if size_mb > 5 else "256QAM"  # priority == 1


def choose_code_rate(update_type: str, priority: int, size_mb: float) -> str:
    if update_type == "critical":
        return "2/15" if (priority >= 4 or size_mb > 50) else "5/15"
    if priority >= 4:
        return "7/15"
    if priority == 3:
        return "7/15" if size_mb > 20 else "9/15"
    if priority == 2:
        return "9/15"
    return "13/15" if size_mb > 5 else "11/15"  # priority == 1


def choose_ldpc_length(update_type: str, priority: int, size_mb: float) -> str:
    if update_type == "critical":
        return "LONG_64800"
    return "LONG_64800" if priority >= 4 else "NORMAL_16200"


def choose_al_fec_redundancy(update_type: str, priority: int, size_mb: float) -> int:
    if update_type == "critical":
        return 50 if (priority >= 4 or size_mb > 50) else 30
    if priority >= 4:
        return 15
    if priority == 3:
        return 15 if size_mb > 20 else 5
    if priority == 2:
        return 5
    return 0  # priority == 1


def choose_carousel_repeat(update_type: str, priority: int, size_mb: float) -> int:
    """Seconds between rebroadcasts of the file object. 0 = on-demand only."""
    if update_type == "critical":
        return 120 if (priority >= 4 or size_mb > 50) else 300
    if priority >= 4:
        return 900
    if priority == 3:
        return 900 if size_mb > 20 else 1800
    if priority == 2:
        return 1800
    return 0  # priority == 1, on-demand only


LABEL_FUNCTIONS = {
    "plp": choose_plp,
    "modulation": choose_modulation,
    "code_rate": choose_code_rate,
    "ldpc_length": choose_ldpc_length,
    "al_fec_redundancy_pct": choose_al_fec_redundancy,
    "carousel_repeat_sec": choose_carousel_repeat,
}


# ---------------------------------------------------------------------------
# SAFETY FLOOR GUARDRAILS (applied per-parameter, independent of the model)
# ---------------------------------------------------------------------------
# These are hard limits critical updates can never fall below, regardless of
# what any downstream model later predicts. Enforced as post-processing, not
# as something a model can override.

CRITICAL_FLOORS = {
    "modulation": "16QAM",          # never less robust than 16QAM
    "code_rate": "5/15",            # never a higher/less robust code rate than 5/15
    "ldpc_length": "LONG_64800",    # always the long, more robust frame
    "al_fec_redundancy_pct": 30,    # never less than 30% repair overhead
    "carousel_repeat_sec": 300,     # never repeats less often than every 5 min
}


def enforce_safety_floors(update_type: str, row: dict) -> dict:
    if update_type != "critical":
        return row
    row = dict(row)
    if MODULATION_OPTIONS.index(row["modulation"]) > MODULATION_OPTIONS.index(CRITICAL_FLOORS["modulation"]):
        row["modulation"] = CRITICAL_FLOORS["modulation"]
    if CODE_RATE_OPTIONS.index(row["code_rate"]) > CODE_RATE_OPTIONS.index(CRITICAL_FLOORS["code_rate"]):
        row["code_rate"] = CRITICAL_FLOORS["code_rate"]
    if row["ldpc_length"] != CRITICAL_FLOORS["ldpc_length"]:
        row["ldpc_length"] = CRITICAL_FLOORS["ldpc_length"]
    if row["al_fec_redundancy_pct"] < CRITICAL_FLOORS["al_fec_redundancy_pct"]:
        row["al_fec_redundancy_pct"] = CRITICAL_FLOORS["al_fec_redundancy_pct"]
    if row["carousel_repeat_sec"] == 0 or row["carousel_repeat_sec"] > CRITICAL_FLOORS["carousel_repeat_sec"]:
        row["carousel_repeat_sec"] = CRITICAL_FLOORS["carousel_repeat_sec"]
    return row


def label_update(update_type: str, priority: int, size_mb: float) -> dict:
    row = {name: fn(update_type, priority, size_mb) for name, fn in LABEL_FUNCTIONS.items()}
    return enforce_safety_floors(update_type, row)


# ---------------------------------------------------------------------------
# STEP 4: SYNTHETIC DATA GENERATION -- unchanged in approach
# ---------------------------------------------------------------------------

def sample_update(rng: np.random.Generator) -> dict:
    is_critical = rng.random() < 0.20
    update_type = "critical" if is_critical else "non_critical"
    target_ecu = rng.choice(CRITICAL_ECUS if is_critical else NON_CRITICAL_ECUS)

    if is_critical:
        priority = int(np.clip(rng.normal(loc=4, scale=1), 1, 5))
        size_mb = float(np.clip(rng.lognormal(mean=2.5, sigma=0.8), 0.5, 300))
    else:
        priority = int(np.clip(rng.normal(loc=2.5, scale=1.2), 1, 5))
        size_mb = float(np.clip(rng.lognormal(mean=3.5, sigma=1.0), 0.5, 500))

    return {
        "update_type": update_type,
        "target_ecu": target_ecu,
        "priority": priority,
        "size_mb": round(size_mb, 2),
    }


def generate_dataset(n_rows: int, labeled: bool, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for _ in range(n_rows):
        row = sample_update(rng)
        if labeled:
            labels = label_update(row["update_type"], row["priority"], row["size_mb"])
            row.update(labels)
        else:
            for col in LABEL_FUNCTIONS:
                row[col] = None
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    labeled_seed = generate_dataset(n_rows=150, labeled=True, rng=rng)
    unlabeled_pool = generate_dataset(n_rows=10_000, labeled=False, rng=rng)
    labeled_test = generate_dataset(n_rows=500, labeled=True, rng=rng)

    labeled_seed.to_csv(f"{OUTPUT_DIR}/labeled_seed.csv", index=False)
    unlabeled_pool.to_csv(f"{OUTPUT_DIR}/unlabeled_pool.csv", index=False)
    labeled_test.to_csv(f"{OUTPUT_DIR}/labeled_test.csv", index=False)

    print(f"Labeled seed set:      {len(labeled_seed)} rows -> {OUTPUT_DIR}/labeled_seed.csv")
    print(f"Unlabeled pool:        {len(unlabeled_pool)} rows -> {OUTPUT_DIR}/unlabeled_pool.csv")
    print(f"Held-out labeled test: {len(labeled_test)} rows -> {OUTPUT_DIR}/labeled_test.csv")

    print("\nPer-parameter label distribution (labeled seed set):")
    for col in LABEL_FUNCTIONS:
        print(f"\n-- {col} --")
        print(labeled_seed[col].value_counts())

    print("\nSample rows:")
    print(labeled_seed.head(8).to_string(index=False))

    print("\nSanity check -- any critical row violating its floor?")
    crit = labeled_seed[labeled_seed.update_type == "critical"]
    violations = crit[
        (crit.al_fec_redundancy_pct < CRITICAL_FLOORS["al_fec_redundancy_pct"])
        | (crit.ldpc_length != CRITICAL_FLOORS["ldpc_length"])
    ]
    print(f"Violations found: {len(violations)} (should be 0)")
