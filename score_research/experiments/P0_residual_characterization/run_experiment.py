"""P0 command-line entry point. Inspect never loads a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import yaml

HERE = Path(__file__).resolve().parent
PHASES = ("inspect", "forecast", "build", "analyze", "report")
VERSIONS = ("torch", "numpy", "pandas", "xarray", "netCDF4", "scipy", "matplotlib", "PyYAML")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def identity(path, content=False):
    path = path.resolve()
    stat = path.stat()
    item = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if content:
        item["sha256"] = digest(path)
    return item


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def environment(root):
    versions = {"python": sys.version}
    for name in VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(software=versions, git_commit=git(root, "rev-parse", "HEAD"),
                git_branch=git(root, "branch", "--show-current"),
                git_status=git(root, "status", "--porcelain", "--untracked-files=all"))


def forecast_signature(inputs, env, strict_hash, hash_cache):
    source_files = {p for files in inputs.input_files.values() for p in files.values()}
    weights = {inputs.weight(step) for path in inputs.plans for step in path}
    code = [HERE / p for p in ("forecast.py", "data_io.py")] + [
        inputs.root / p for p in ("utils/init.py", "utils/handle.py", "utils/scheduler.py",
                                 "merra_norm.csv", "info.pkl")]
    sources = []
    for path in sorted(source_files | weights):
        item = identity(path)
        if strict_hash:
            previous = hash_cache.get(item["path"], {})
            if any(previous.get(k) != value for k, value in item.items()):
                previous = identity(path, content=True)
                hash_cache[item["path"]] = previous
            item = previous
        sources.append(item)
    return dict(config=inputs.config, files=sources, full_file_hashes=strict_hash,
                code=[identity(p, content=True) for p in code],
                software=env["software"], execution=env["execution"], commit=env["git_commit"])


def fingerprint(value):
    raw = json.dumps(value, sort_keys=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def inspect_data(root, config, run):
    from data_io import Inputs
    from forecast import preflight

    report = {"status": "checking", "software": environment(root)["software"]}
    try:
        preflight(Inputs(root, config), report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(run / "data_validation.json", report)


def require(run, status, phase, name):
    current = status.get(phase, {})
    path = run / name
    if not completed(current, path):
        raise RuntimeError(f"{phase} has no matching completed artifact: {path}")
    return path


def completed(previous, path):
    return (previous.get("state") == "complete" and path.is_file()
            and previous.get("artifact") == identity(path))


def execute(phase, root, config, run, status, args, prior):
    env = environment(root)
    match phase:
        case "inspect":
            inspect_data(root, config, run)
            return run / "data_validation.json"
        case "forecast":
            import torch
            from data_io import Inputs
            from forecast import generate, project_directory

            inspect_data(root, config, run)
            inputs = Inputs(root, config)
            device = torch.device(config["forecast"]["device"])
            env["execution"] = {"device": str(device), "cuda_runtime": torch.version.cuda,
                                "cudnn": torch.backends.cudnn.version()}
            if device.type == "cuda":
                properties = torch.cuda.get_device_properties(device)
                env["execution"].update(gpu=properties.name, capability=[properties.major, properties.minor])
            cache_path = run / "file_hashes.json"
            hashes = read_json(cache_path)
            signature = forecast_signature(inputs, env, args.hash_sources, hashes)
            if args.hash_sources:
                write_json(cache_path, hashes)
            token = fingerprint(signature)
            path = run / "data" / "forecasts.nc"
            previous = read_json(run / "metadata.json")
            if (args.resume and not args.refresh_cache and completed(prior, path)
                    and previous.get("forecast_fingerprint") == token
                    and previous.get("model_calls") == identity_if_present(run / "model_calls.json")):
                logging.info("Reusing matching forecasts")
                return path
            importance = dict(**env, forecast_fingerprint=token, signature=signature,
                              reference_files=[str(p) for p in inputs.reference_files])
            with project_directory(root):
                generate(inputs, run, token, json.dumps(importance), args.overwrite or args.refresh_cache)
            importance["model_calls"] = identity(run / "model_calls.json")
            write_json(run / "metadata.json", importance)
            return path
        case "build":
            from data_io import build

            forecast = require(run, status, "forecast", "data/forecasts.nc")
            reference = [root / config["data"]["reference_dir"] / p
                         for p in config["data"]["reference_files"]]
            dependencies = dict(forecast=identity(forecast), reference=[identity(p) for p in reference],
                                code=digest(HERE / "data_io.py"))
            manifest = run / "build_inputs.json"
            target = run / "data" / "residuals.nc"
            comparison = run / "data" / "path_comparison.nc"
            previous = read_json(manifest)
            if (args.resume and completed(prior, target) and previous.get("inputs") == dependencies
                    and previous.get("comparison") == identity_if_present(comparison)):
                logging.info("Reusing matching residual dataset")
                return target
            provenance = dict(**env, **dependencies)
            build(config, root, run, json.dumps(provenance))
            write_json(manifest, {"inputs": dependencies, "comparison": identity_if_present(comparison)})
            return target
        case "analyze":
            from analysis import analyze

            require(run, status, "build", "data/residuals.nc")
            folder = run / f"analysis_{uuid.uuid4().hex[:12]}"
            analyze(run, config, folder)
            marker = folder / "complete.json"
            write_json(marker, {"residuals": status["build"]["artifact"], "config": config["analysis"],
                                "code_sha256": digest(HERE / "analysis.py")})
            return marker
        case "report":
            from report import report

            require(run, status, "build", "data/residuals.nc")
            analysis = status.get("analyze", {}).get("artifact", {}).get("path")
            if not analysis:
                raise RuntimeError("Complete analyze before report")
            marker = require(run, status, "analyze", Path(analysis).relative_to(run))
            report(run, marker.parent, config, HERE)
            return run / "report.html"


def identity_if_present(path):
    return identity(path) if path.is_file() else None


def invalidate(status, phase):
    if phase == "inspect":
        return
    for downstream in PHASES[PHASES.index(phase) + 1:]:
        if downstream in status:
            status[downstream]["state"] = "stale"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(*PHASES, "all"))
    parser.add_argument("--root", type=Path, default=HERE.parents[2])
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--run-id", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse the run and any valid prediction prefixes")
    mode.add_argument("--overwrite", action="store_true", help="Rerun selected stages in the same run")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute cached prefixes if forecast runs")
    parser.add_argument("--hash-sources", action="store_true", help="Also compute SHA-256 on large forecast inputs and weights")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = args.run_id or now
    experiment = config["experiment"]["name"]
    for value in (experiment, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            parser.error(f"Unsafe path component: {value}")
    run = (root / config["output"]["directory"] / experiment / run_id).resolve()
    existed = run.exists()
    if existed and not (args.resume or args.overwrite):
        parser.error(f"Run exists. Use --resume or --overwrite: {run}")
    run.mkdir(parents=True, exist_ok=True)
    lock = run / ".running"
    try:
        with lock.open("x", encoding="utf-8") as file:
            file.write(f"{os.getpid()}\n")
    except FileExistsError:
        parser.error(f"Run is locked: {lock}")

    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(run / "run.log", encoding="utf-8")])
        logging.info("Run directory: %s", run)
        snapshot = run / "config_snapshot.yaml"
        config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if snapshot.is_file() and snapshot.read_text(encoding="utf-8") != config_text:
            raise ValueError("Run configuration changed. Use a new --run-id.")
        if not snapshot.is_file():
            atomic_text(snapshot, config_text)
        (run / "data").mkdir(exist_ok=True)
        status_path = run / "status.json"
        status = read_json(status_path)
        selected = PHASES if args.phase == "all" else (args.phase,)
        for phase in selected:
            if phase == "report" and args.phase == "all" and not config["output"]["generate_html_report"]:
                continue
            prior = status.get(phase, {})
            artifact = prior.get("artifact", {}).get("path")
            if args.phase == "all" and args.resume and artifact and completed(prior, Path(artifact)):
                if phase == "analyze" and read_json(Path(artifact)).get("code_sha256") == digest(HERE / "analysis.py"):
                    continue
            status[phase] = {"state": "running"}
            write_json(status_path, status)
            logging.info("Starting %s", phase)
            try:
                result = execute(phase, root, config, run, status, args, prior)
            except Exception as error:
                status[phase] = {"state": "failed", "error": str(error)}
                invalidate(status, phase)
                write_json(status_path, status)
                logging.exception("Stage %s failed", phase)
                return 1
            status[phase] = {"state": "complete", "artifact": identity(result)}
            if prior != status[phase]:
                invalidate(status, phase)
            write_json(status_path, status)
            logging.info("Finished %s", phase)
    finally:
        lock.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
