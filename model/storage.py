from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import importlib.util
import pickle
import hashlib

from tools.utils import stable_id, adc_id, signal_id, safe_name


RUN_FORMAT_VERSION = 14


def _definition_path(root, kind, name):
    path = Path(root) / kind / "definitions" / f"{name}.py"
    if not path.exists():
        available = sorted(p.stem for p in path.parent.glob("*.py") if p.name != "__init__.py")
        raise FileNotFoundError(f"Unknown {kind[:-1]} definition '{name}'. Available: {available}")
    return path


def _load_definition_module(path):
    source = Path(path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    module_name = f"lc_matrix_definition_{Path(path).stem}_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, digest


def load_adc_definition(root, name):
    path = _definition_path(root, "adc", name)
    module, source_hash = _load_definition_module(path)
    maker = getattr(module, "make_adc", None) or getattr(module, "get_adc", None) or getattr(module, "get_definition", None)
    if maker is None:
        raise AttributeError(f"{path} must define make_adc(), get_adc(), or get_definition().")
    obj = maker()
    # A definition-file edit must create a distinct ADC even if numerical fields happen to be unchanged.
    obj.name = name
    definition_hash = stable_id({"definition_name": name, "source_sha256": source_hash, "adc_parameters": getattr(obj, "dp", obj)})
    return obj, definition_hash, path



class RunStore:
    """Flat persistent history: one pickle per exact ADC definition/realization."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _files(self):
        return sorted(self.runs_dir.glob("*.pkl"))

    @staticmethod
    def _load(path):
        with Path(path).open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _dump(path, record):
        record["updated_at"] = datetime.now(ZoneInfo("Europe/Zurich")).isoformat(timespec="seconds")
        tmp = Path(str(path) + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def find_adc_run(self, adc_definition_hash):
        for path in reversed(self._files()):
            try:
                record = self._load(path)
            except Exception:
                continue
            if record.get("adc_definition_hash") == adc_definition_hash:
                return path, record
        return None, None

    def get_adc_run(self, adc_name, create=True):
        adc, definition_hash, definition_path = load_adc_definition(self.root, adc_name)
        path, record = self.find_adc_run(definition_hash)
        if record is not None:
            return path, record
        if not create:
            raise FileNotFoundError(f"ADC '{adc_name}' [{definition_hash[:10]}] has no saved run yet.")

        stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%d%m%y%H%M")
        filename = self.runs_dir / f"{stamp}_{definition_hash[:6]}.pkl"
        suffix = 1
        while filename.exists():
            filename = self.runs_dir / f"{stamp}_{definition_hash[:6]}_{suffix}.pkl"
            suffix += 1
        now = datetime.now(ZoneInfo("Europe/Zurich")).isoformat(timespec="seconds")
        record = {
            "format_version": RUN_FORMAT_VERSION,
            "created_at": now,
            "updated_at": now,
            "adc_name": adc_name,
            "adc_definition_file": str(definition_path.relative_to(self.root)),
            "adc_definition_hash": definition_hash,
            "adc_definition_source": definition_path.read_text(),
            "adc_id": adc_id(adc),
            "adc": adc,
            "characterizations": {},
            "signals": {},
        }
        self._dump(filename, record)
        return filename, record

    def save(self, path, record):
        self._dump(path, record)

    def get_signal_entry(self, record, signal, create=True):
        """Find/create a cached signal entry from an actual Timeseries object.

        Signal identity is based on the returned Timeseries data/time/params. Loader or
        generator provenance stored in ``series.params`` therefore participates in the hash.
        """
        sid = signal_id(signal)
        signal_name = safe_name(getattr(signal, "name", "signal"))
        key = f"{signal_name}_{sid[:8]}"
        if key not in record["signals"]:
            if not create:
                raise FileNotFoundError(f"Signal '{signal_name}' [{sid[:10]}] has not been run on ADC '{record['adc_name']}'.")
            record["signals"][key] = {
                "name": getattr(signal, "name", signal_name),
                "signal_id": sid,
                "source": dict(getattr(signal, "params", {})).get("LC-matrix signal source"),
                "parameters": dict(getattr(signal, "params", {})),
                "signal": signal.copy() if hasattr(signal, "copy") else signal,
                "signal_statistics": {},
                "empirical_runs": {},
                "statistical_runs": {},
                "comparisons": {},
            }
        return key, record["signals"][key], record["signals"][key]["signal"]

    def list_runs(self):
        rows = []
        for path in self._files():
            try:
                r = self._load(path)
            except Exception as exc:
                rows.append({"file": path.name, "error": str(exc)})
                continue
            signal_names = [v.get("name", k) for k, v in r.get("signals", {}).items()]
            n_emp = sum(len(v.get("empirical_runs", {})) for v in r.get("signals", {}).values())
            n_stat = sum(len(v.get("statistical_runs", {})) for v in r.get("signals", {}).values())
            rows.append({
                "file": path.name,
                "adc": r.get("adc_name"),
                "adc_hash": r.get("adc_definition_hash", "")[:10],
                "characterizations": len(r.get("characterizations", {})),
                "signals": ", ".join(signal_names) if signal_names else "-",
                "empirical": n_emp,
                "statistical": n_stat,
                "levels": getattr(getattr(r.get("adc"), "dp", None), "lsb_n", "-"),
                "dnl_seed": getattr(getattr(r.get("adc"), "dp", None), "dac_dnl_seed", "-"),
                "noise_seed": getattr(getattr(r.get("adc"), "dp", None), "noise_seed", "-"),
                "updated": r.get("updated_at", "-"),
            })
        return rows

    def print_runs(self):
        rows = self.list_runs()
        if not rows:
            print("No saved ADC runs.")
            return
        headers = ["file", "adc", "adc_hash", "levels", "dnl_seed", "noise_seed", "characterizations", "signals", "empirical", "statistical", "updated"]
        widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
        print("\nSAVED ADC RUNS")
        print(" | ".join(h.ljust(widths[h]) for h in headers))
        print("-+-".join("-" * widths[h] for h in headers))
        for r in rows:
            print(" | ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))


def print_saved_runs(root="."):
    RunStore(root).print_runs()


if __name__ == "__main__":
    print_saved_runs(Path(__file__).resolve().parents[1])
