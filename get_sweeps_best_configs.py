import wandb

ENTITY = "EM_IMCR_BIOVSION"
PROJECT = "ForkSight-SAM"
METRIC = "validation/composite"
HIGHER_IS_BETTER = True  # flip to False if lower is better

api = wandb.Api()
project_path = f"{ENTITY}/{PROJECT}"
sweeps = api.project(name=PROJECT, entity=ENTITY).sweeps()

for sweep in sweeps:
    sweep = api.sweep(f"{project_path}/{sweep.id}")

    # Hyperparameters actually swept over, per the sweep config
    swept_params = set(sweep.config.get("parameters", {}).keys())

    finished_runs = [
        r for r in sweep.runs
        if r.state == "finished" and METRIC in r.summary
    ]

    if not finished_runs:
        print(f"\n=== Sweep: {sweep.name or sweep.id} ({sweep.id}) ===")
        print("  No finished runs with the target metric.")
        continue

    best_run = sorted(
        finished_runs,
        key=lambda r: r.summary[METRIC],
        reverse=HIGHER_IS_BETTER,
    )[0]

    # Keep only the swept hyperparameters
    tuned_config = {
        k: v for k, v in best_run.config.items()
        if k in swept_params
    }

    print(f"\n=== Sweep: {sweep.name or sweep.id} ({sweep.id}) ===")
    print(f"  Runs (finished / total): {len(finished_runs)} / {len(sweep.runs)}")
    print(f"  Best run: {best_run.name} ({best_run.id})")
    print(f"  {METRIC}: {best_run.summary[METRIC]:.4f}")
    print(f"  URL: {best_run.url}")
    print(f"  Tuned hyperparameters ({len(tuned_config)}):")
    for k, v in sorted(tuned_config.items()):
        print(f"    {k}: {v}")