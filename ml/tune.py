"""
tune.py — Optuna hyperparameter optimisation for FractureClassifier.

Runs N Bayesian trials of two-phase training with reduced epochs for speed.
Study is persisted to SQLite so interrupted runs can be resumed.
Best parameters are saved to logs/best_params.json when the study completes.

Usage:
    python tune.py                # run 20 trials (default)
    python tune.py --trials 10    # run a specific number of trials
    python tune.py --trials 0     # print results from an existing study (no training)

Recommended workflow:
    1. python tune.py --trials 20    # run overnight on M4 Mac (~12-16 hours)
    2. python tune.py --trials 0     # inspect results
    3. Copy best params from logs/best_params.json into config.py
    4. python train.py               # final training with validated hyperparameters
    5. python evaluate.py            # evaluation on test set
"""

import argparse
import json
import shutil

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .config import CHECKPOINT_DIR, LOG_DIR
from .train import train_model


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: train with sampled hyperparameters, return best val AUC."""

    # ── Learning rates (log-uniform: equal probability across orders of magnitude)
    p1_lr = trial.suggest_float("PHASE1_LR", 3e-4, 3e-3, log=True)
    p2_lr = trial.suggest_float("PHASE2_LR", 1e-5, 2e-4, log=True)

    # ── Weight decay for both phases (log-uniform: regularisation strength)
    p1_wd = trial.suggest_float("PHASE1_WEIGHT_DECAY", 1e-5, 1e-2, log=True)
    p2_wd = trial.suggest_float("PHASE2_WEIGHT_DECAY", 1e-5, 1e-2, log=True)

    # ── Label smoothing: 0.0 = off, >0.2 typically hurts calibration
    label_smoothing = trial.suggest_float("LABEL_SMOOTHING", 0.0, 0.2)

    # ── Number of backbone blocks to unfreeze in Phase 2 (EfficientNet-B0 has 9 blocks)
    unfreeze_n = trial.suggest_int("UNFREEZE_LAST_N", 1, 4)

    # ── Dropout rates in the classification head
    dropout1 = trial.suggest_float("DROPOUT1", 0.1, 0.5)
    dropout2 = trial.suggest_float("DROPOUT2", 0.05, 0.4)

    cfg_override = {
        "PHASE1_LR":           p1_lr,
        "PHASE2_LR":           p2_lr,
        "PHASE1_WEIGHT_DECAY": p1_wd,
        "PHASE2_WEIGHT_DECAY": p2_wd,
        "LABEL_SMOOTHING":     label_smoothing,
        "UNFREEZE_LAST_N":     unfreeze_n,
        "DROPOUT1":            dropout1,
        "DROPOUT2":            dropout2,
        # Reduced epochs for HPO speed (full training uses 20 / 40)
        "PHASE1_EPOCHS":       10,
        "PHASE2_EPOCHS":       20,
        "EARLY_STOP_PATIENCE": 5,
        # Optuna control keys (consumed by train_model, not passed to PyTorch)
        "_trial":              trial,
        "_trial_number":       trial.number,
        "_verbose":            False,
    }

    return train_model(cfg_override=cfg_override)


def run_study(n_trials: int) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    db_path = LOG_DIR / "optuna.db"
    storage = f"sqlite:///{db_path}"

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(
        n_startup_trials=5,   # first 5 trials are never pruned (build baseline distribution)
        n_warmup_steps=8,     # no pruning before step 8 (let training stabilise past early epochs)
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name="fracture_hpo",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,   # resume seamlessly if study already exists at db_path
    )

    print(f"Starting HPO study ({n_trials} new trials).")
    print(f"Study database : {db_path}")
    print(f"Existing trials: {len(study.trials)}\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),   # log failed trials as FAIL state without crashing the study
    )

    _save_best_params(study)
    _print_summary(study)


def _save_best_params(study: optuna.Study) -> None:
    best = study.best_trial
    output = {
        "best_trial_number": best.number,
        "best_val_auc":      round(best.value, 6),
        "params":            best.params,
    }
    out_path = LOG_DIR / "best_params.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nBest parameters saved → {out_path}")

    # Copy best trial's checkpoint to a canonical location for evaluate.py
    best_trial_ckpt = CHECKPOINT_DIR / f"trial_{best.number}" / "best_model.pth"
    canonical_ckpt  = CHECKPOINT_DIR / "best_hpo_model.pth"
    if best_trial_ckpt.exists():
        shutil.copy2(best_trial_ckpt, canonical_ckpt)
        print(f"Best checkpoint copied → {canonical_ckpt}")


def _print_summary(study: optuna.Study) -> None:
    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    pruned    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    failed    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)

    print("\n" + "=" * 60)
    print("OPTUNA STUDY SUMMARY")
    print("=" * 60)
    print(f"  Total trials   : {len(study.trials)}")
    print(f"  Completed      : {completed}")
    print(f"  Pruned         : {pruned}")
    print(f"  Failed         : {failed}")
    print(f"\n  Best trial     : #{study.best_trial.number}")
    print(f"  Best val AUC   : {study.best_value:.4f}")
    print("\n  Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {k:<25} = {v}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for FractureClassifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of new trials to run (default: 20). Use 0 to print existing results.",
    )
    args = parser.parse_args()

    if args.trials > 0:
        run_study(args.trials)
    else:
        db_path = LOG_DIR / "optuna.db"
        if not db_path.exists():
            print(f"No study found at {db_path}. Run 'python tune.py --trials N' first.")
            return
        study = optuna.load_study(
            study_name="fracture_hpo",
            storage=f"sqlite:///{db_path}",
        )
        _print_summary(study)


if __name__ == "__main__":
    main()
