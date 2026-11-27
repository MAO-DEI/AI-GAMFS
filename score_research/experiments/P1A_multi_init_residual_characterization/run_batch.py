"""P1-A staged batch CLI. GPU inference is restricted to Linux."""
import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P0 = HERE.with_name("P0_residual_characterization")
sys.path[:0] = [str(ROOT), str(P0)]

import pandas as pd
import yaml

from data_io import Inputs, iso, utc
from run_experiment import atomic_text, digest, environment, fingerprint, forecast_signature, identity, read_json, write_json
from storage import build, init_folder, init_times, inspect, single_config, verify_and_seed_p0

PHASES = ("inspect", "forecast", "build", "analyze", "aggregate", "report")


def files(folder, phase, names=()):
    if phase == "forecast":
        return [folder / "data/forecasts.nc", folder / "model_calls.json", folder / "metadata.json"]
    if phase == "build":
        return [folder / "residuals.nc"]
    return [folder / "statistics" / f"{name}.csv" for name in names]


def completed(status, folder, phase):
    entry = status.get(phase, {})
    artifacts = files(folder, phase, entry.get("tables", []))
    return (entry.get("state") == "complete" and bool(artifacts)
            and all(p.is_file() for p in artifacts)
            and entry.get("artifacts") == [identity(p) for p in artifacts])


@contextmanager
def lock_run(run):
    lock = run / ".running"
    try:
        with lock.open("x", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")
    except FileExistsError as error:
        raise RuntimeError(f"Run already locked: {lock}") from error
    try:
        yield
    finally:
        lock.unlink()


def forecast_inputs(root, config, env, hash_sources, hash_cache):
    from forecast import project_directory

    source = Inputs(root, config)
    full = dict(config, variables={name: {"model": name, "reference": name} for name in source.names})
    inputs = Inputs(root, full)
    # GEOS and weights may be hashed once per batch. Reference/analyzer revisions do not affect this key.
    signature = forecast_signature(inputs, env, hash_sources, hash_cache)
    signature["config"] = {"init_times": config["experiment"]["init_times"], "forecast": config["forecast"]}
    return inputs, signature, project_directory


def stage(root, run, stamp, config, phase, args, env, validation, hash_cache):
    folder = init_folder(run, stamp)
    (folder / "data").mkdir(parents=True, exist_ok=True)
    status_path = folder / "status.json"
    status = read_json(status_path)
    prior = status.get(phase, {})
    started = monotonic()
    tables = []
    single = single_config(config, stamp)
    logging.info("%s %s", phase, iso(stamp))
    try:
        if phase == "forecast":
            if validation.get(iso(stamp)) != "passed":
                raise ValueError("Initialization failed inspect")
            inputs, signature, project_directory = forecast_inputs(root, single, env, args.hash_sources, hash_cache)
            token = fingerprint(signature)
            key = token
            if args.resume and not args.refresh_forecast and completed(status, folder, phase) and prior.get("key") == key:
                return True
            provenance = dict(**env, experiment_config=single, signature=signature, fingerprint=token,
                              normalization_source="merra_norm.csv", model_weights=[
                                  str(inputs.weight(h)) for h in sorted({h for p in inputs.plans for h in p})])
            if args.p0_cache and not args.refresh_forecast and stamp == utc("2024-07-26T22:30:00Z"):
                try:
                    provenance["p0_cache"] = verify_and_seed_p0(args.p0_cache, inputs, folder, token, env)
                except (OSError, KeyError, ValueError, TypeError) as error:
                    provenance["p0_cache"] = {"rejected": str(error)}
                    logging.warning("P0 cache rejected, generating missing prefixes: %s", error)
        else:
            parent = "forecast" if phase == "build" else "build"
            if not completed(status, folder, parent):
                raise ValueError(f"Missing completed {parent} artifact")
            if phase == "build":
                source = Inputs(root, single, include_initial=False)
                provenance = {"forecast": status[parent]["artifacts"], "references": [
                    identity(p) for p in source.reference_files], "code": [
                        digest(HERE / "storage.py"), digest(P0 / "data_io.py"), digest(root / "utils/datasets.py")],
                    "experiment_config": single, **env}
            else:
                provenance = {"build": status[parent]["artifacts"], "analysis": config["analysis"],
                              "code": [digest(HERE / "diagnostics.py"), digest(P0 / "analysis.py")],
                              "normalization": digest(root / "merra_norm.csv")}
            key = fingerprint(provenance if phase == "analyze" else {
                k: v for k, v in provenance.items() if k in {"forecast", "references", "code", "experiment_config"}})
            if args.resume and completed(status, folder, phase) and prior.get("key") == key:
                return True

        for downstream in ("forecast", "build", "analyze")[("forecast", "build", "analyze").index(phase) + 1:]:
            if downstream in status:
                status[downstream]["state"] = "stale"
        status[phase] = {"state": "running"}
        write_json(status_path, status)

        if phase == "forecast":
            from forecast import generate

            with project_directory(root):
                generate(inputs, folder, token, json.dumps(provenance, ensure_ascii=False), args.refresh_forecast)
            write_json(folder / "metadata.json", provenance)
        elif phase == "build":
            build(root, single, folder, provenance)
        else:
            from diagnostics import analyze

            norm = pd.read_csv(root / "merra_norm.csv", index_col="index")
            floor = norm["std"] * config["analysis"]["denominator_rtol"]
            tables = analyze(folder, config["analysis"], floor)
        status[phase] = dict(state="complete", seconds=monotonic() - started, key=key, tables=tables,
                             artifacts=[identity(p) for p in files(folder, phase, tables)])
    except Exception as error:
        logging.exception("%s failed for %s", phase, iso(stamp))
        status[phase] = {"state": "failed", "seconds": monotonic() - started, "error": str(error)}
        for downstream in ("forecast", "build", "analyze")[("forecast", "build", "analyze").index(phase) + 1:]:
            if downstream in status:
                status[downstream]["state"] = "stale"
    write_json(status_path, status)
    return status[phase]["state"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*PHASES, "all"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--init", action="append", help="Select a configured UTC initialization (repeatable)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh-forecast", action="store_true", help="Ignore existing prefix caches")
    parser.add_argument("--p0-cache", type=Path, help="Optional verified P0 run directory")
    parser.add_argument("--hash-sources", action="store_true", help="Hash large GEOS and weight files")
    args = parser.parse_args()
    selected = PHASES if args.phase == "all" else (args.phase,)
    if "forecast" in selected and not sys.platform.startswith("linux"):
        parser.error("Run formal forecasting on the Linux GPU server")
    root = args.root.resolve()
    if args.p0_cache is not None:
        args.p0_cache = args.p0_cache.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    name = config["experiment"]["name"]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in (name, args.run_id)):
        parser.error("Invalid experiment name or run ID")
    configured = init_times(config)
    times = sorted({utc(t) for t in args.init}) if args.init else configured
    if not set(times).issubset(configured):
        parser.error("--init must select configured timestamps")
    run = (root / config["output"]["directory"] / name / args.run_id).resolve()
    if run.exists() and not args.resume:
        parser.error("Run already exists. Use --resume or a new run ID")
    run.mkdir(parents=True, exist_ok=True)

    with lock_run(run):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(run / "run.log", encoding="utf-8")])
        snapshot = run / "config_snapshot.yaml"
        if snapshot.exists():
            previous = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
            if previous != config:
                parser.error("Run configuration changed. Use a new run ID")
        else:
            atomic_text(snapshot, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        env = environment(root)
        write_json(run / "metadata.json", dict(**env, experiment_config=config, time_zone="UTC",
                                                configured_initializations=[iso(t) for t in configured]))
        failed = False
        validation = None
        hash_cache = read_json(run / "file_hashes.json")
        for phase in selected:
            status = {iso(t): read_json(init_folder(run, t) / "status.json") for t in configured}
            if phase == "inspect" or (phase == "forecast" and validation is None):
                # Validate all planned dates so the data inventory is complete even for a smoke test.
                result = inspect(root, run, config, configured)
                validation = {d["init_time"]: d["status"] for d in result["initializations"]}
                failed |= any(validation[iso(t)] != "passed" for t in times)
            if phase in {"forecast", "build", "analyze"}:
                if phase == "forecast" and "execution" not in env:
                    import torch

                    device = torch.device(config["forecast"]["device"])
                    if device.type != "cuda":
                        parser.error("Forecast device must be CUDA")
                    gpu = torch.cuda.get_device_properties(device)
                    env["execution"] = {"device": str(device), "cuda_runtime": torch.version.cuda,
                                        "cudnn": torch.backends.cudnn.version(), "gpu": gpu.name,
                                        "capability": [gpu.major, gpu.minor]}
                for stamp in times:
                    failed |= not stage(root, run, stamp, config, phase, args, env, validation, hash_cache)
                if phase == "forecast" and args.hash_sources:
                    write_json(run / "file_hashes.json", hash_cache)
            elif phase == "aggregate":
                from aggregate import aggregate_fields, collect

                valid = [init_folder(run, t) for t in times if completed(status[iso(t)], init_folder(run, t), "analyze")]
                if not valid:
                    logging.error("No completed analysis to aggregate")
                    failed = True
                    continue
                (run / "aggregate.json").unlink(missing_ok=True)
                stats = collect(valid, run)
                aggregate_fields(valid, run, config["analysis"], stats)
                write_json(run / "aggregate.json", {"included": [p.name for p in valid], "config": config["analysis"],
                    "analyses": [identity(p / "status.json") for p in valid]})
            elif phase == "report":
                from generate_report import report

                aggregate = read_json(run / "aggregate.json")
                if not aggregate or any(not Path(x["path"]).is_file() or identity(Path(x["path"])) != x
                                        for x in aggregate["analyses"]):
                    logging.error("Aggregate results are missing or stale")
                    failed = True
                    continue
                report(run, config, status, aggregate["included"])
            write_json(run / "status.json", {iso(t): read_json(init_folder(run, t) / "status.json") for t in configured})
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
