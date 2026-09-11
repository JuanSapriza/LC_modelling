from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


def ratio_to_db(ratio):
    ratio = float(ratio)
    if not np.isfinite(ratio):
        return np.nan
    if ratio == 0:
        return -np.inf
    return 20 * np.log10(abs(ratio))


def _fmt_rate(value):
    return "-" if value is None or not np.isfinite(value) else f"{value:.0f} Xps"


def _fmt_signed_mV(value):
    return "-" if value is None or not np.isfinite(value) else f"{value*1e3:+.1f} mV"


def _fmt_ratio(value):
    return "-" if value is None or not np.isfinite(value) else f"{100*value:.1f}% / {ratio_to_db(value):.1f} dB"


def _empirical_mode(metrics, suppressed=False):
    if suppressed and hasattr(metrics, "crossing_rate_reversals_suppressed"):
        return metrics.crossing_rate_reversals_suppressed
    if (not suppressed) and hasattr(metrics, "crossing_rate_reversals_reported"):
        return metrics.crossing_rate_reversals_reported
    rates = np.asarray(getattr(metrics, "crossing_rate_Hz", []), dtype=float)
    rates = rates[np.isfinite(rates)]
    return SimpleNamespace(
        mean_rate_Hz=float(getattr(metrics, "average_crossing_rate_Hz", np.nan)),
        sigma_rate_Hz=float(getattr(metrics, "sigma_crossing_rate_Hz", np.nan)),
        min_rate_Hz=float(np.min(rates)) if len(rates) else np.nan,
        max_rate_Hz=float(np.max(rates)) if len(rates) else np.nan,
    )


def _empirical_drift(metrics, suppressed=False):
    if suppressed and hasattr(metrics, "drift_reversals_suppressed"):
        return metrics.drift_reversals_suppressed
    if (not suppressed) and hasattr(metrics, "drift_reversals_reported"):
        return metrics.drift_reversals_reported
    return metrics.drift


def _d2_mode(metrics, suppressed=False):
    return metrics.crossing_rates.reversals_suppressed if suppressed else metrics.crossing_rates.reversals_reported


def _d2_drift(metrics, suppressed=False):
    return metrics.drift.reversals_suppressed if suppressed else metrics.drift.reversals_reported


@dataclass
class ModelComparison:
    empirical: object
    second_order: object | None = None
    first_order: object | None = None
    zeroth_order: object | None = None
    characterization_only: object | None = None
    selected_order: str = "D2"

    def summary(self):
        out = {"empirical": self.empirical.summary(), "selected_order": self.selected_order}
        if self.characterization_only is not None:
            out["W_static"] = self.characterization_only.summary()
        if self.zeroth_order is not None:
            out["D0_zeroth_order"] = self.zeroth_order.summary()
        if self.first_order is not None:
            out["D1_first_order"] = self.first_order.summary()
        if self.second_order is not None:
            out["D2_second_order"] = self.second_order.summary()
        return out

    def _selected(self):
        return {
            "W": self.characterization_only,
            "D0": self.zeroth_order,
            "D1": self.first_order,
            "D2": self.second_order,
        }.get(str(self.selected_order).upper())

    def rows(self):
        e = self.empirical
        w0, d0, d1, d2 = self.characterization_only, self.zeroth_order, self.first_order, self.second_order
        selected = self._selected()
        selected_name = str(self.selected_order).upper()
        rows = []

        def add(metric, w="-", z="-", f="-", s="-", emp="-", diff="-"):
            rows.append((metric, w, z, f, s, emp, diff))

        def width_stats(obj):
            if obj is None:
                return None
            if obj is d2:
                return d2.level_width.crossing_statistics
            return getattr(obj, "width_statistics", None)

        def global_ratio(obj):
            if obj is None:
                return None
            if obj is d2:
                return d2.distortion.global_ratio_crossings
            return getattr(obj, "global_distortion_ratio", None)

        def dr(obj):
            return None if obj is None else getattr(obj, "dynamic_range", None)

        def fmt_dr(obj):
            value = dr(obj)
            return "-" if value is None else f"{value.dynamic_range:.1f}x / {value.dynamic_range_dB:.1f} dB"

        def fmt_range(obj):
            value = dr(obj)
            return "-" if value is None else f"[{value.min_x:.2f}, {value.max_x:.2f}] V"

        # Transition classes are only defined once derivative direction is known.
        e_same = 100 * e.same_direction_fraction
        e_rev = 100 * e.reversal_fraction
        add(
            "Same-direction transitions",
            f="100.0%" if d1 is not None else "-",
            s=f"{100*d2.crossing_rates.same_direction_fraction:.1f}%" if d2 is not None else "-",
            emp=f"{e_same:.1f}%",
            diff=(f"{100*d2.crossing_rates.same_direction_fraction-e_same:+.1f} pp" if selected_name == "D2" and d2 is not None else (f"{100-e_same:+.1f} pp" if selected_name == "D1" and d1 is not None else "-")),
        )
        add(
            "Reversal transitions",
            f="0.0%" if d1 is not None else "-",
            s=f"{100*d2.crossing_rates.reversal_fraction:.1f}%" if d2 is not None else "-",
            emp=f"{e_rev:.1f}%",
            diff=(f"{100*d2.crossing_rates.reversal_fraction-e_rev:+.1f} pp" if selected_name == "D2" and d2 is not None else (f"{-e_rev:+.1f} pp" if selected_name == "D1" and d1 is not None else "-")),
        )

        # Width progression.
        values = []
        for obj in (w0, d0, d1, d2):
            st = width_stats(obj)
            values.append("-" if st is None else f"{1e3*st.mean:.1f} mV")
        emp_mean = 1e3*e.width_statistics.mean
        sel_st = width_stats(selected)
        diff = "-" if sel_st is None else f"{1e3*(sel_st.mean-e.width_statistics.mean):+.1f} mV"
        add("Average width (same)", *values, emp=f"{emp_mean:.1f} mV", diff=diff)

        values = []
        for obj in (w0, d0, d1, d2):
            st = width_stats(obj)
            values.append("-" if st is None else f"{1e3*st.sigma:.1f} mV")
        sel_st = width_stats(selected)
        diff = "-" if sel_st is None else f"{1e3*(sel_st.sigma-e.width_statistics.sigma):+.1f} mV"
        add("Sigma width (same)", *values, emp=f"{1e3*e.width_statistics.sigma:.1f} mV", diff=diff)

        vals = [_fmt_ratio(global_ratio(obj)) for obj in (w0, d0, d1, d2)]
        sel_ratio = global_ratio(selected)
        diff = "-" if sel_ratio is None else f"{100*(sel_ratio-e.global_distortion_ratio):+.1f} pp / {ratio_to_db(sel_ratio)-ratio_to_db(e.global_distortion_ratio):+.1f} dB"
        add("Global distortion (same)", *vals, emp=_fmt_ratio(e.global_distortion_ratio), diff=diff)

        d1_local = None if d1 is None else d1.local_distortion_ratio
        d2_local = None if d2 is None else d2.distortion.local_sigma_ratio
        sel_local = d1_local if selected_name == "D1" else d2_local if selected_name == "D2" else None
        diff = "-" if sel_local is None else f"{100*(sel_local-e.local_sigma_ratio):+.1f} pp / {ratio_to_db(sel_local)-ratio_to_db(e.local_sigma_ratio):+.1f} dB"
        add("Local distortion (same)", "-", "-", _fmt_ratio(d1_local), _fmt_ratio(d2_local), _fmt_ratio(e.local_sigma_ratio), diff)

        # Reversal metrics require D2.
        if d2 is not None:
            rev = d2.local_distortion.reversal_recapture_statistics
            add("Reversal recapture mean", "-", "-", "N/A", f"{1e3*rev.mean:+.1f} mV", f"{1e3*e.reversal_recapture_statistics.mean:+.1f} mV", f"{1e3*(rev.mean-e.reversal_recapture_statistics.mean):+.1f} mV" if selected_name == "D2" else "-")
            add("Reversal recapture sigma", "-", "-", "N/A", f"{1e3*rev.sigma:.1f} mV", f"{1e3*e.reversal_recapture_statistics.sigma:.1f} mV", f"{1e3*(rev.sigma-e.reversal_recapture_statistics.sigma):+.1f} mV" if selected_name == "D2" else "-")
        else:
            add("Reversal recapture mean", "-", "-", "N/A" if d1 is not None else "-", "-", f"{1e3*e.reversal_recapture_statistics.mean:+.1f} mV", "-")
            add("Reversal recapture sigma", "-", "-", "N/A" if d1 is not None else "-", "-", f"{1e3*e.reversal_recapture_statistics.sigma:.1f} mV", "-")

        # Crossing-rate distributions. D0/W cannot predict crossing timing.
        for title, suppressed in (("Rate [reversals reported]", False), ("Rate [reversals suppressed]", True)):
            em = _empirical_mode(e, suppressed)
            d1m = None if d1 is None else d1.crossing_rate
            d2m = None if d2 is None else _d2_mode(d2, suppressed)
            selected_rate = d1m if selected_name == "D1" else d2m if selected_name == "D2" else None
            for suffix, field in (("min", "min_rate_Hz"), ("avg", "mean_rate_Hz"), ("sigma", "sigma_rate_Hz"), ("max", "max_rate_Hz")):
                ev = getattr(em, field)
                sv = None if selected_rate is None else getattr(selected_rate, field)
                add(f"{title} {suffix}", "-", "-", "-" if d1m is None else _fmt_rate(getattr(d1m, field)), "-" if d2m is None else _fmt_rate(getattr(d2m, field)), _fmt_rate(ev), "-" if sv is None else _fmt_rate(sv-ev))

        # Dynamic range.
        selected_dr = dr(selected)
        diff_dr = "-" if selected_dr is None or e.dynamic_range is None else f"{selected_dr.dynamic_range-e.dynamic_range.dynamic_range:+.1f}x / {selected_dr.dynamic_range_dB-e.dynamic_range.dynamic_range_dB:+.1f} dB"
        diff_range = "-" if selected_dr is None or e.dynamic_range is None else f"Δmin {selected_dr.min_x-e.dynamic_range.min_x:+.2f} V, Δmax {selected_dr.max_x-e.dynamic_range.max_x:+.2f} V"
        add("Max dynamic range", fmt_dr(w0), fmt_dr(d0), fmt_dr(d1), fmt_dr(d2), fmt_dr(e), diff_dr)
        add("Input range @ max DR", fmt_range(w0), fmt_range(d0), fmt_range(d1), fmt_range(d2), fmt_range(e), diff_range)

        # Drift reference and signed error. W can estimate drift/crossing but has no rate.
        wdr = None if w0 is None else w0.drift
        d1dr = None if d1 is None else d1.drift
        d2dr = None if d2 is None else _d2_drift(d2, False)
        edr = _empirical_drift(e, False)
        add(
            "Drift reference width",
            "-" if wdr is None else f"{1e3*wdr.reference_width_V:.1f} mV",
            "-",
            "-" if d1dr is None else f"{1e3*d1dr.reference_width_V:.1f} mV",
            "-" if d2dr is None else f"{1e3*d2dr.reference_width_V:.1f} mV",
            f"{1e3*edr.reference_width_V:.1f} mV",
            "-" if selected is None or selected_name == "D0" else f"{1e3*((wdr.reference_width_V if selected_name=='W' else d1dr.reference_width_V if selected_name=='D1' else d2dr.reference_width_V)-edr.reference_width_V):+.1f} mV",
        )

        for title, suppressed in (("Drift [reversals reported]", False), ("Drift [reversals suppressed]", True)):
            ed = _empirical_drift(e, suppressed)
            d2d = None if d2 is None else _d2_drift(d2, suppressed)
            w_pc = None if wdr is None else wdr.mean_error_per_crossing_V
            d1_pc = None if d1dr is None else d1dr.mean_error_per_crossing_V
            d2_pc = None if d2d is None else d2d.mean_error_per_crossing_V
            sel_pc = w_pc if selected_name == "W" else d1_pc if selected_name == "D1" else d2_pc if selected_name == "D2" else None
            sel_ratio = None if selected_name == "D0" else (wdr.per_crossing_ratio if selected_name == "W" else d1dr.per_crossing_ratio if selected_name == "D1" else d2d.per_crossing_ratio if selected_name == "D2" else None)
            add(
                f"{title} / crossing",
                "-" if wdr is None else f"{1e3*w_pc:+.3f} mV / {100*wdr.per_crossing_ratio:+.2f}%",
                "-",
                "-" if d1dr is None else f"{1e3*d1_pc:+.3f} mV / {100*d1dr.per_crossing_ratio:+.2f}%",
                "-" if d2d is None else f"{1e3*d2_pc:+.3f} mV / {100*d2d.per_crossing_ratio:+.2f}%",
                f"{1e3*ed.per_crossing_drift_V:+.3f} mV / {100*ed.per_crossing_ratio:+.2f}%",
                "-" if sel_pc is None else f"{1e3*(sel_pc-ed.per_crossing_drift_V):+.3f} mV / {100*(sel_ratio-ed.per_crossing_ratio):+.2f} pp",
            )
            d1_rate = None if d1dr is None else d1dr.drift_rate_V_s
            d2_rate = None if d2d is None else d2d.drift_rate_V_s
            sel_rate = d1_rate if selected_name == "D1" else d2_rate if selected_name == "D2" else None
            er = getattr(ed, "drift_rate_V_s", None)
            add(f"{title} / second", "N/A" if wdr is not None else "-", "-", "-" if d1_rate is None else f"{1e3*d1_rate:+.2f} mV/s", "-" if d2_rate is None else f"{1e3*d2_rate:+.2f} mV/s", "-" if er is None or not np.isfinite(er) else f"{1e3*er:+.2f} mV/s", "-" if sel_rate is None or er is None or not np.isfinite(er) else f"{1e3*(sel_rate-er):+.2f} mV/s")

        return rows


def compare_models(empirical_metrics, statistical_metrics=None, first_order_metrics=None, zeroth_order_metrics=None, characterization_only_metrics=None, selected_order="D2"):
    return ModelComparison(empirical=empirical_metrics, second_order=statistical_metrics, first_order=first_order_metrics, zeroth_order=zeroth_order_metrics, characterization_only=characterization_only_metrics, selected_order=selected_order)


def print_comparison(comparison):
    rows = comparison.rows()
    selected = str(comparison.selected_order).upper()
    headers = ("Metric", "W static", "D0 zeroth", "D1 first", "D2 second", "Empirical", f"Difference ({selected} - Emp)")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    def line(values):
        return " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(values))

    print("\nMODEL COMPARISON")
    print(line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(line(row))
