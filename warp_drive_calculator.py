#!/usr/bin/env python3
"""Alcubierre Warp Drive Calculator - Terminal Edition"""

import numpy as np
from scipy import constants
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple
from functools import wraps
from enum import Enum, auto

C = constants.c
G = constants.G
H = constants.h
HBAR = constants.hbar
K_B = constants.k

LY = 9.461e15
AU = 1.496e11
PC = 3.086e16
M_SUN = 1.989e30
M_EARTH = 5.972e24
M_JUPITER = 1.898e27
YEAR_S = 31536000

CONFIG_PATH = Path.home() / ".warp_config.json"
HISTORY_PATH = Path.home() / ".warp_history.json"


class RiskLevel(Enum):
    SAFE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
    FATAL = auto()


@dataclass
class BubbleConfig:
    radius: float = 100.0
    sigma: float = 5.0
    name: str = "standard"


@dataclass
class WarpResult:
    module: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: dict = field(default_factory=dict)


@dataclass
class Star:
    name: str
    distance: float
    spectral: str
    mass: float = 1.0
    habitable: bool = False


@dataclass
class Mission:
    origin: str
    destination: str
    warp: float
    cargo: float
    crew: int


STARS = {
    k: Star(**v) for k, v in {
        "sol": {"name": "Sol", "distance": 0, "spectral": "G2V", "mass": 1.0, "habitable": True},
        "proxima": {"name": "Proxima Centauri", "distance": 4.24, "spectral": "M5.5Ve", "mass": 0.12, "habitable": True},
        "alpha_a": {"name": "Alpha Centauri A", "distance": 4.37, "spectral": "G2V", "mass": 1.1, "habitable": True},
        "barnard": {"name": "Barnard's Star", "distance": 5.96, "spectral": "M4Ve", "mass": 0.14},
        "wolf359": {"name": "Wolf 359", "distance": 7.86, "spectral": "M6.5Ve", "mass": 0.09},
        "sirius_a": {"name": "Sirius A", "distance": 8.58, "spectral": "A1V", "mass": 2.06},
        "ross154": {"name": "Ross 154", "distance": 9.69, "spectral": "M3.5Ve", "mass": 0.17},
        "epsilon": {"name": "Epsilon Eridani", "distance": 10.52, "spectral": "K2V", "mass": 0.82, "habitable": True},
        "tau_ceti": {"name": "Tau Ceti", "distance": 11.91, "spectral": "G8.5V", "mass": 0.78, "habitable": True},
        "vega": {"name": "Vega", "distance": 25.04, "spectral": "A0Va", "mass": 2.14},
        "arcturus": {"name": "Arcturus", "distance": 36.7, "spectral": "K1.5III", "mass": 1.08},
        "capella": {"name": "Capella", "distance": 42.9, "spectral": "G5III", "mass": 2.69},
        "rigel": {"name": "Rigel", "distance": 860, "spectral": "B8Ia", "mass": 21},
        "betelgeuse": {"name": "Betelgeuse", "distance": 700, "spectral": "M1Ia", "mass": 16.5},
        "deneb": {"name": "Deneb", "distance": 2615, "spectral": "A2Ia", "mass": 19},
        "polaris": {"name": "Polaris", "distance": 433, "spectral": "F7Ib", "mass": 5.4},
        "sgr_a": {"name": "Sagittarius A*", "distance": 26000, "spectral": "SMBH", "mass": 4e6},
    }.items()
}

PRESETS = {
    "shuttle": BubbleConfig(10, 20, "shuttle"),
    "standard": BubbleConfig(100, 5, "standard"),
    "cruiser": BubbleConfig(500, 3, "cruiser"),
    "carrier": BubbleConfig(2000, 1, "carrier"),
    "station": BubbleConfig(5000, 0.5, "station"),
}


class Physics:
    @staticmethod
    def warp_to_c(wf: float) -> float:
        return wf ** (10 / 3)

    @staticmethod
    def c_to_warp(v_c: float) -> float:
        return v_c ** (3 / 10) if v_c > 0 else 0

    @staticmethod
    def shape(r: float, R: float, sigma: float) -> float:
        return (np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))) / (2 * np.tanh(sigma * R))

    @staticmethod
    def shape_deriv(r: float, R: float, sigma: float) -> float:
        sp = 1 / np.cosh(sigma * (r + R)) ** 2
        sm = 1 / np.cosh(sigma * (r - R)) ** 2
        return sigma * (sp - sm) / (2 * np.tanh(sigma * R))

    @staticmethod
    def energy_density(r: float, v: float, R: float, sigma: float) -> float:
        df = Physics.shape_deriv(r, R, sigma)
        return -(C ** 4 / (8 * np.pi * G)) * (v ** 2 / (4 * C ** 2)) * (df ** 2)

    @staticmethod
    def total_energy(v: float, R: float, sigma: float) -> float:
        integrand = lambda r: Physics.energy_density(r, v, R, sigma) * 4 * np.pi * r ** 2
        result, _ = quad(integrand, 0, R * 5, limit=100)
        return result

    @staticmethod
    def gamma(v_c: float) -> float:
        return 1 / np.sqrt(1 - min(v_c, 0.9999) ** 2) if v_c < 1 else float('inf')

    @staticmethod
    def doppler(v_c: float, approaching: bool = True) -> float:
        b = min(abs(v_c), 0.9999)
        return np.sqrt((1 + b) / (1 - b)) if approaching else np.sqrt((1 - b) / (1 + b))

    @staticmethod
    def hawking_temp(mass: float) -> float:
        return (HBAR * C ** 3) / (8 * np.pi * G * max(mass, 1e-30) * K_B)

    @staticmethod
    def schwarzschild(mass: float) -> float:
        return 2 * G * mass / C ** 2


class Formatter:
    @staticmethod
    def sci(v: float, p: int = 3) -> str:
        if v == 0:
            return "0"
        if abs(v) == float('inf'):
            return "inf"
        if np.isnan(v):
            return "nan"
        exp = int(np.floor(np.log10(abs(v))))
        mant = v / (10 ** exp)
        return f"{mant:.{p}f}e{exp}"

    @staticmethod
    def time(s: float) -> str:
        units = [(60, "s"), (3600, "m"), (86400, "h"), (YEAR_S, "d"), (float('inf'), "y")]
        for limit, suffix in units:
            if abs(s) < limit:
                divisors = {"s": 1, "m": 60, "h": 3600, "d": 86400, "y": YEAR_S}
                return f"{s / divisors[suffix]:.2f}{suffix}"
        return f"{s / YEAR_S:.2f}y"

    @staticmethod
    def dist(m: float) -> str:
        if abs(m) < 1e6:
            return f"{m:.1f}m"
        if abs(m) < AU:
            return f"{m / 1e6:.2f}Mm"
        if abs(m) < LY:
            return f"{m / AU:.3f}AU"
        return f"{m / LY:.4f}ly"

    @staticmethod
    def mass(kg: float) -> str:
        if abs(kg) < M_EARTH:
            return f"{Formatter.sci(kg)}kg"
        if abs(kg) < M_JUPITER:
            return f"{kg / M_EARTH:.3f}Me"
        if abs(kg) < M_SUN:
            return f"{kg / M_JUPITER:.3f}Mj"
        return f"{kg / M_SUN:.4f}Ms"

    @staticmethod
    def energy(j: float) -> str:
        return f"{Formatter.sci(j)}J"

    @staticmethod
    def risk(level: RiskLevel) -> str:
        return level.name


class Terminal:
    WIDTH = 56

    @staticmethod
    def clear():
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def box(title: str):
        w = Terminal.WIDTH
        pad = (w - 4 - len(title)) // 2
        print("+" + "-" * (w - 2) + "+")
        print("|" + " " * pad + title + " " * (w - 4 - pad - len(title)) + "|")
        print("+" + "-" * (w - 2) + "+")

    @staticmethod
    def line():
        print("-" * Terminal.WIDTH)

    @staticmethod
    def row(label: str, value: str):
        gap = Terminal.WIDTH - len(label) - len(value) - 4
        print(f"  {label}{'.' * max(1, gap)}{value}")

    @staticmethod
    def blank():
        print()

    @staticmethod
    def text(msg: str):
        print(f"  {msg}")

    @staticmethod
    def table(headers: List[str], rows: List[List[str]]):
        widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
        fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("  " + "  ".join("-" * w for w in widths))
        for r in rows:
            print(fmt.format(*[str(v) for v in r]))

    @staticmethod
    def bar(value: float, max_val: float, width: int = 20) -> str:
        ratio = min(abs(value) / max(abs(max_val), 1e-30), 1.0)
        filled = int(ratio * width)
        return "#" * filled + "." * (width - filled)

    @staticmethod
    def chart(data: List[Tuple[float, float]], x_label: str, y_label: str, width: int = 40, height: int = 12):
        if not data:
            return
        xs, ys = zip(*data)
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        y_range = y_max - y_min or 1
        x_range = x_max - x_min or 1
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        for x, y in data:
            col = int((x - x_min) / x_range * (width - 1))
            row = height - 1 - int((y - y_min) / y_range * (height - 1))
            col = max(0, min(width - 1, col))
            row = max(0, min(height - 1, row))
            grid[row][col] = '*'
        print(f"  {y_label}")
        for i, row in enumerate(grid):
            prefix = f"{y_max - i * y_range / (height - 1):>8.1e}|" if i % 3 == 0 else "        |"
            print(f"  {prefix}{''.join(row)}")
        print("  " + " " * 8 + "+" + "-" * width)
        print(f"  {' ' * 8}{x_min:<10.1e}{' ' * (width - 20)}{x_max:>10.1e}")
        print(f"  {' ' * ((8 + width) // 2)}{x_label}")


class Input:
    @staticmethod
    def number(prompt: str, default: float = None, lo: float = None, hi: float = None) -> float:
        while True:
            d = f" [{default}]" if default is not None else ""
            raw = input(f"  {prompt}{d}: ").strip()
            if raw == "" and default is not None:
                return float(default)
            try:
                val = float(raw)
                if lo is not None and val < lo:
                    Terminal.text(f"Min: {lo}")
                    continue
                if hi is not None and val > hi:
                    Terminal.text(f"Max: {hi}")
                    continue
                return val
            except ValueError:
                Terminal.text("Invalid")

    @staticmethod
    def integer(prompt: str, default: int = None, lo: int = None, hi: int = None) -> int:
        return int(Input.number(prompt, default, lo, hi))

    @staticmethod
    def choice(prompt: str, options: List[str]) -> str:
        while True:
            raw = input(f"  {prompt}: ").strip().lower()
            if raw in options:
                return raw
            Terminal.text(f"Options: {options}")

    @staticmethod
    def confirm(prompt: str) -> bool:
        return Input.choice(prompt, ['y', 'n']) == 'y'

    @staticmethod
    def select(prompt: str, items: Dict[str, any]) -> str:
        keys = list(items.keys())
        for i, k in enumerate(keys, 1):
            Terminal.text(f"{i}. {k}")
        Terminal.blank()
        idx = Input.integer(prompt, 1, 1, len(keys))
        return keys[idx - 1]

    @staticmethod
    def pause():
        input("  [Enter]...")


class Storage:
    @staticmethod
    def load_config() -> dict:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text())
        return {"precision": 4, "warp": 5.0, "dist": 4.24, "preset": "standard"}

    @staticmethod
    def save_config(cfg: dict):
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

    @staticmethod
    def load_history() -> List[dict]:
        if HISTORY_PATH.exists():
            return json.loads(HISTORY_PATH.read_text())
        return []

    @staticmethod
    def save_result(result: WarpResult):
        history = Storage.load_history()
        history.append(asdict(result))
        history = history[-100:]
        HISTORY_PATH.write_text(json.dumps(history, indent=2))

    @staticmethod
    def clear_history():
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()


class Module:
    registry: Dict[str, 'Module'] = {}

    def __init__(self, key: str, name: str, category: str):
        self.key = key
        self.name = name
        self.category = category
        Module.registry[key] = self

    def run(self):
        raise NotImplementedError


def module(key: str, name: str, category: str):
    def decorator(cls):
        instance = cls(key, name, category)
        return instance
    return decorator


@module("warp", "Warp Velocity", "Velocity")
class WarpVelocity(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("WARP VELOCITY")
        Terminal.blank()
        Terminal.text("v = w^(10/3) * c")
        Terminal.blank()
        Terminal.line()
        wf = Input.number("Warp factor", 5.0, 1.0, 9.99)
        dist = Input.number("Distance ly", 4.24, 0.001, 1e6)
        v_c = Physics.warp_to_c(wf)
        t = (dist * LY) / (v_c * C)
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Warp", f"{wf:.2f}")
        Terminal.row("Speed", f"{v_c:.2f}c")
        Terminal.row("Distance", f"{dist:.2f}ly")
        Terminal.row("Time", Formatter.time(t))
        Terminal.blank()
        Storage.save_result(WarpResult("warp", data={"wf": wf, "v_c": v_c, "t": t}))
        Input.pause()


@module("rel", "Relativistic", "Velocity")
class Relativistic(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("RELATIVISTIC")
        Terminal.blank()
        Terminal.text("gamma = 1/sqrt(1-v^2)")
        Terminal.blank()
        Terminal.line()
        v_c = Input.number("Velocity c", 0.9, 0.01, 0.9999)
        m0 = Input.number("Rest mass kg", 1000, 0.001, 1e30)
        gamma = Physics.gamma(v_c)
        L = np.sqrt(1 - v_c ** 2)
        m_rel = m0 * gamma
        KE = (gamma - 1) * m0 * C ** 2
        p = gamma * m0 * v_c * C
        d_app = Physics.doppler(v_c, True)
        d_rec = Physics.doppler(v_c, False)
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Gamma", f"{gamma:.4f}")
        Terminal.row("Length", f"{L:.4f}")
        Terminal.row("Rel mass", Formatter.mass(m_rel))
        Terminal.row("Momentum", Formatter.sci(p))
        Terminal.row("KE", Formatter.energy(KE))
        Terminal.row("Doppler+", f"{d_app:.4f}")
        Terminal.row("Doppler-", f"{d_rec:.4f}")
        Terminal.blank()
        rows = [[f"{v:.2f}", f"{Physics.gamma(v):.2f}", f"{np.sqrt(1 - v ** 2):.4f}"]
                for v in [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]]
        Terminal.table(["v/c", "gamma", "L"], rows)
        Terminal.blank()
        Storage.save_result(WarpResult("rel", data={"v_c": v_c, "gamma": gamma}))
        Input.pause()


@module("energy", "Energy Req", "Energy")
class EnergyReq(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("ENERGY REQ")
        Terminal.blank()
        Terminal.text("E = -c^4/(8piG) * ...")
        Terminal.blank()
        for k, p in PRESETS.items():
            Terminal.text(f"{k}: R={p.radius}m s={p.sigma}")
        Terminal.blank()
        Terminal.line()
        use_preset = Input.confirm("Use preset? y/n")
        if use_preset:
            preset = Input.select("Preset", PRESETS)
            cfg = PRESETS[preset]
            R, sigma = cfg.radius, cfg.sigma
        else:
            R = Input.number("Radius m", 100, 1, 10000)
            sigma = Input.number("Sharpness", 5, 0.1, 100)
        v_c = Input.number("Velocity c", 1.0, 0.01, 10)
        v = v_c * C
        E = Physics.total_energy(v, R, sigma)
        m_eq = abs(E) / C ** 2
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Radius", f"{R}m")
        Terminal.row("Wall", f"{1 / sigma:.2f}m")
        Terminal.row("Velocity", f"{v_c}c")
        Terminal.row("Energy", Formatter.energy(E))
        Terminal.row("Mass eq", Formatter.mass(m_eq))
        Terminal.row("vs Jupiter", f"{m_eq / M_JUPITER:.2e}x")
        Terminal.blank()
        Storage.save_result(WarpResult("energy", data={"E": E, "m": m_eq}))
        Input.pause()


@module("exotic", "Exotic Matter", "Energy")
class ExoticMatter(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("EXOTIC MATTER")
        Terminal.blank()
        Terminal.text("Negative energy")
        Terminal.blank()
        Terminal.line()
        E = Input.number("Energy J", -1e45, -1e60, 0)
        m_eq = abs(E) / C ** 2
        casimir = -(np.pi ** 2 * HBAR * C) / 240
        casimir_vol = abs(E) / abs(casimir)
        squeezed = abs(E) / (HBAR * 5e14 * 0.01)
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Energy", Formatter.energy(E))
        Terminal.row("Mass eq", Formatter.mass(m_eq))
        Terminal.blank()
        Terminal.text("Generation:")
        Terminal.row("Casimir vol", f"{Formatter.sci(casimir_vol)}m3")
        Terminal.row("Squeezed ph", Formatter.sci(squeezed))
        Terminal.blank()
        Storage.save_result(WarpResult("exotic", data={"E": E}))
        Input.pause()


@module("efficiency", "Efficiency", "Energy")
class Efficiency(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("EFFICIENCY")
        Terminal.blank()
        Terminal.text("Energy per ly")
        Terminal.blank()
        Terminal.line()
        R = Input.number("Radius m", 100, 1, 10000)
        sigma = Input.number("Sharpness", 5, 0.1, 100)
        dist = Input.number("Distance ly", 10, 0.1, 1e5)
        Terminal.blank()
        Terminal.line()
        Terminal.box("TABLE")
        Terminal.blank()
        rows = []
        for wf in range(1, 10):
            v_c = Physics.warp_to_c(wf)
            E = abs(Physics.total_energy(v_c * C, R, sigma))
            t = (dist * LY) / (v_c * C)
            rows.append([str(wf), f"{v_c:.1f}c", Formatter.time(t), Formatter.sci(E / dist)])
        Terminal.table(["Warp", "Speed", "Time", "E/ly"], rows)
        Terminal.blank()
        Storage.save_result(WarpResult("efficiency", data={"R": R, "sigma": sigma}))
        Input.pause()


@module("shape", "Shape Func", "Geometry")
class ShapeFunc(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("SHAPE FUNC")
        Terminal.blank()
        Terminal.text("f(r) = tanh form")
        Terminal.blank()
        Terminal.line()
        R = Input.number("Radius m", 100, 1, 10000)
        sigma = Input.number("Sharpness", 5, 0.1, 100)
        Terminal.blank()
        Terminal.line()
        Terminal.box("PROFILE")
        Terminal.blank()
        data = []
        for i in range(16):
            r = (i / 15) * R * 2
            f = Physics.shape(r, R, sigma)
            df = Physics.shape_deriv(r, R, sigma)
            data.append((r, f))
            bar = Terminal.bar(f, 1.0, 25)
            Terminal.text(f"r={r:6.1f} |{bar}| {f:.3f}")
        Terminal.blank()
        wall = 2 / sigma
        Terminal.row("Wall thick", f"{wall:.2f}m")
        Terminal.row("Safe zone", f"{R - wall:.1f}m")
        Terminal.blank()
        Storage.save_result(WarpResult("shape", data={"R": R, "sigma": sigma}))
        Input.pause()


@module("density", "Energy Dens", "Geometry")
class EnergyDens(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("ENERGY DENS")
        Terminal.blank()
        Terminal.text("rho(r) profile")
        Terminal.blank()
        Terminal.line()
        R = Input.number("Radius m", 100, 1, 10000)
        sigma = Input.number("Sharpness", 5, 0.1, 100)
        v_c = Input.number("Velocity c", 1.0, 0.01, 10)
        v = v_c * C
        Terminal.blank()
        Terminal.line()
        Terminal.box("PROFILE")
        Terminal.blank()
        densities = []
        for i in range(16):
            r = (i / 15) * R * 2 + 0.001
            rho = Physics.energy_density(r, v, R, sigma)
            densities.append((r, rho))
        min_rho = min(d[1] for d in densities)
        for r, rho in densities:
            bar = Terminal.bar(rho, min_rho, 20)
            Terminal.text(f"r={r:6.1f} |{bar}| {Formatter.sci(rho)}")
        E_total = Physics.total_energy(v, R, sigma)
        Terminal.blank()
        Terminal.row("Total E", Formatter.energy(E_total))
        Terminal.row("Mass eq", Formatter.mass(abs(E_total) / C ** 2))
        Terminal.blank()
        Storage.save_result(WarpResult("density", data={"E": E_total}))
        Input.pause()


@module("volume", "Natario Vol", "Geometry")
class NatarioVol(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("NATARIO VOL")
        Terminal.blank()
        Terminal.text("n = 1 + v^2/c^2")
        Terminal.blank()
        Terminal.line()
        v_c = Input.number("Velocity c", 0.5, 0.0, 10)
        n = 1 + v_c ** 2
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Velocity", f"{v_c}c")
        Terminal.row("Volume n", f"{n:.6f}")
        gamma_str = f"{Physics.gamma(v_c):.4f}" if v_c < 1 else "FTL"
        Terminal.row("Gamma", gamma_str)
        Terminal.blank()
        Terminal.text("Volume curve:")
        data = [(v, 1 + v ** 2) for v in np.linspace(0, 3, 20)]
        Terminal.chart(data, "v/c", "n", 35, 10)
        Terminal.blank()
        Storage.save_result(WarpResult("volume", data={"v_c": v_c, "n": n}))
        Input.pause()


@module("tidal", "Tidal Force", "Dynamics")
class TidalForce(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("TIDAL FORCE")
        Terminal.blank()
        Terminal.text("Stress analysis")
        Terminal.blank()
        Terminal.line()
        R = Input.number("Radius m", 100, 1, 10000)
        sigma = Input.number("Sharpness", 5, 0.1, 100)
        v_c = Input.number("Velocity c", 1.0, 0.01, 10)
        size = Input.number("Object m", 2, 0.1, 100)
        delta = 1 / sigma
        gradient = (v_c * C) ** 2 / (C ** 2 * delta ** 2)
        tidal_a = gradient * size
        tidal_g = tidal_a / 9.81
        risk = (
            RiskLevel.SAFE if tidal_g < 0.5 else
            RiskLevel.LOW if tidal_g < 1 else
            RiskLevel.MEDIUM if tidal_g < 5 else
            RiskLevel.HIGH if tidal_g < 20 else
            RiskLevel.CRITICAL if tidal_g < 100 else
            RiskLevel.FATAL
        )
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Gradient", f"{Formatter.sci(gradient)}/m")
        Terminal.row("Tidal acc", f"{Formatter.sci(tidal_a)}m/s2")
        Terminal.row("Tidal G", f"{tidal_g:.2f}g")
        Terminal.row("Risk", Formatter.risk(risk))
        Terminal.row("Safe zone", f"{max(0, R - 3 * delta):.1f}m")
        Terminal.blank()
        Storage.save_result(WarpResult("tidal", data={"tidal_g": tidal_g, "risk": risk.name}))
        Input.pause()


@module("dynamics", "Bubble Dyn", "Dynamics")
class BubbleDyn(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("BUBBLE DYN")
        Terminal.blank()
        Terminal.text("Acceleration")
        Terminal.blank()
        Terminal.line()
        R = Input.number("Radius m", 100, 1, 10000)
        sigma = Input.number("Sharpness", 5, 0.1, 100)
        v_i = Input.number("Initial c", 0.0, 0.0, 10)
        v_f = Input.number("Final c", 1.0, 0.01, 10)
        t_acc = Input.number("Accel time s", 100, 0.1, 86400)
        dv = v_f - v_i
        acc_c = dv / t_acc
        acc_ms = acc_c * C
        E_i = abs(Physics.total_energy(v_i * C, R, sigma))
        E_f = abs(Physics.total_energy(v_f * C, R, sigma))
        dE = E_f - E_i
        power = dE / t_acc
        g_force = acc_ms / 9.81
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Delta v", f"{dv:.4f}c")
        Terminal.row("Accel", f"{Formatter.sci(acc_ms)}m/s2")
        Terminal.row("G-force", f"{Formatter.sci(g_force)}g")
        Terminal.row("Delta E", Formatter.energy(dE))
        Terminal.row("Power", f"{Formatter.sci(power)}W")
        Terminal.blank()
        Storage.save_result(WarpResult("dynamics", data={"dv": dv, "power": power}))
        Input.pause()


@module("radiation", "Radiation", "Dynamics")
class Radiation(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("RADIATION")
        Terminal.blank()
        Terminal.text("Hawking analog")
        Terminal.blank()
        Terminal.line()
        v_c = Input.number("Velocity c", 1.0, 0.01, 10)
        R = Input.number("Radius m", 100, 1, 10000)
        sigma = Input.number("Sharpness", 5, 0.1, 100)
        delta = 1 / sigma
        surf_g = (v_c * C) ** 2 / (2 * C * delta)
        T = (HBAR * surf_g) / (2 * np.pi * K_B * C)
        area = 4 * np.pi * R ** 2
        P = 5.67e-8 * area * T ** 4
        wl = 2.898e-3 / T if T > 0 else float('inf')
        spectrum = (
            "Radio" if T < 0.1 else
            "Microwave" if T < 10 else
            "IR" if T < 1000 else
            "Visible" if T < 10000 else
            "UV/X"
        )
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Surf grav", f"{Formatter.sci(surf_g)}m/s2")
        Terminal.row("Temp", f"{Formatter.sci(T)}K")
        Terminal.row("Peak wl", f"{Formatter.sci(wl)}m")
        Terminal.row("Spectrum", spectrum)
        Terminal.row("Power", f"{Formatter.sci(P)}W")
        Terminal.blank()
        Storage.save_result(WarpResult("radiation", data={"T": T, "P": P}))
        Input.pause()


@module("nav", "Navigation", "Analysis")
class Navigation(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("NAVIGATION")
        Terminal.blank()
        Terminal.text("Star database:")
        Terminal.blank()
        rows = [[s.name[:12], f"{s.distance:.1f}", s.spectral, "Y" if s.habitable else ""]
                for s in sorted(STARS.values(), key=lambda x: x.distance)[:15]]
        Terminal.table(["Star", "ly", "Type", "Hab"], rows)
        Terminal.blank()
        Terminal.line()
        origin = Input.choice("Origin", list(STARS.keys())) if Input.confirm("Choose origin?") else "sol"
        dest = Input.select("Destination", {k: v.name for k, v in STARS.items()})
        wf = Input.number("Warp factor", 5.0, 1.0, 9.99)
        s1, s2 = STARS[origin], STARS[dest]
        dist = abs(s2.distance - s1.distance) or 0.001
        v_c = Physics.warp_to_c(wf)
        t = (dist * LY) / (v_c * C)
        Terminal.blank()
        Terminal.line()
        Terminal.box("ROUTE")
        Terminal.blank()
        Terminal.row("From", s1.name)
        Terminal.row("To", s2.name)
        Terminal.row("Distance", f"{dist:.2f}ly")
        Terminal.row("Warp", f"{wf}")
        Terminal.row("Speed", f"{v_c:.2f}c")
        Terminal.row("ETA", Formatter.time(t))
        Terminal.blank()
        Storage.save_result(WarpResult("nav", data={"from": origin, "to": dest, "dist": dist, "t": t}))
        Input.pause()


@module("mission", "Mission Plan", "Analysis")
class MissionPlan(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("MISSION PLAN")
        Terminal.blank()
        Terminal.text("Full mission sim")
        Terminal.blank()
        Terminal.line()
        origin = "sol"
        dest = Input.select("Destination", {k: v.name for k, v in STARS.items()})
        wf = Input.number("Warp factor", 6.0, 1.0, 9.99)
        cargo = Input.number("Cargo tons", 100, 0, 1e6)
        crew = Input.integer("Crew size", 50, 1, 10000)
        preset = Input.select("Ship class", {k: f"R={v.radius}m" for k, v in PRESETS.items()})
        cfg = PRESETS[preset]
        s1, s2 = STARS[origin], STARS[dest]
        dist = abs(s2.distance - s1.distance) or 0.001
        v_c = Physics.warp_to_c(wf)
        t_travel = (dist * LY) / (v_c * C)
        E_warp = abs(Physics.total_energy(v_c * C, cfg.radius, cfg.sigma))
        t_acc = 3600
        E_acc = E_warp * 0.1
        E_total = E_warp + E_acc
        life_support = crew * 500 * t_travel
        delta = 1 / cfg.sigma
        tidal_g = ((v_c * C) ** 2 / (C ** 2 * delta ** 2) * 2) / 9.81
        risk = (
            RiskLevel.SAFE if tidal_g < 1 else
            RiskLevel.LOW if tidal_g < 5 else
            RiskLevel.MEDIUM if tidal_g < 20 else
            RiskLevel.HIGH
        )
        Terminal.blank()
        Terminal.line()
        Terminal.box("MISSION SUMMARY")
        Terminal.blank()
        Terminal.row("Route", f"{s1.name} > {s2.name}")
        Terminal.row("Distance", f"{dist:.2f}ly")
        Terminal.row("Warp", f"{wf}")
        Terminal.row("Travel", Formatter.time(t_travel))
        Terminal.blank()
        Terminal.row("Ship", preset)
        Terminal.row("Cargo", f"{cargo}t")
        Terminal.row("Crew", f"{crew}")
        Terminal.blank()
        Terminal.row("Warp E", Formatter.energy(E_warp))
        Terminal.row("Accel E", Formatter.energy(E_acc))
        Terminal.row("Life sup", Formatter.energy(life_support))
        Terminal.row("Total E", Formatter.energy(E_total))
        Terminal.blank()
        Terminal.row("Crew risk", Formatter.risk(risk))
        Terminal.row("Tidal G", f"{tidal_g:.2f}g")
        Terminal.blank()
        Storage.save_result(WarpResult("mission", data={"dest": dest, "dist": dist, "crew": crew}))
        Input.pause()


@module("causality", "Causality", "Analysis")
class Causality(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("CAUSALITY")
        Terminal.blank()
        Terminal.text("Paradox check")
        Terminal.blank()
        Terminal.line()
        v_c = Input.number("Velocity c", 2.0, 0.01, 100)
        dist = Input.number("Distance ly", 10, 0.01, 1e5)
        obs_v = Input.number("Observer c", 0.5, 0.0, 0.99)
        t_travel = (dist * LY) / (v_c * C)
        t_light = (dist * LY) / C
        violation = v_c > 1
        arrival = "Before light" if t_travel < t_light else "After light"
        gamma_obs = Physics.gamma(obs_v) if obs_v < 1 else 1
        t_obs = abs(gamma_obs * (t_travel - obs_v * dist * YEAR_S / C))
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Ship v", f"{v_c}c")
        Terminal.row("Distance", f"{dist}ly")
        Terminal.row("T travel", Formatter.time(t_travel))
        Terminal.row("T light", Formatter.time(t_light))
        Terminal.row("Arrival", arrival)
        Terminal.blank()
        Terminal.row("Observer v", f"{obs_v}c")
        Terminal.row("T observed", Formatter.time(t_obs))
        Terminal.blank()
        Terminal.row("Violation", "YES" if violation else "NO")
        Terminal.row("Paradox", "POSSIBLE" if violation else "NONE")
        Terminal.blank()
        Storage.save_result(WarpResult("causality", data={"violation": violation}))
        Input.pause()


@module("horizon", "Horizons", "Analysis")
class Horizons(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("HORIZONS")
        Terminal.blank()
        Terminal.text("Event horizon")
        Terminal.blank()
        Terminal.line()
        v_c = Input.number("Velocity c", 1.0, 0.01, 10)
        R = Input.number("Radius m", 100, 1, 10000)
        has_horizon = v_c >= 1
        E = abs(Physics.total_energy(v_c * C, R, 5))
        m_eq = E / C ** 2
        r_s = Physics.schwarzschild(m_eq)
        comm_delay = R / C if has_horizon else 0
        signal_loss = 1 - np.exp(-v_c) if has_horizon else 0
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        Terminal.row("Velocity", f"{v_c}c")
        Terminal.row("Horizon", "Apparent" if has_horizon else "None")
        Terminal.row("Eff R_s", f"{Formatter.sci(r_s)}m")
        Terminal.row("Comm delay", f"{comm_delay:.6f}s")
        Terminal.row("Signal loss", f"{signal_loss * 100:.1f}%")
        Terminal.blank()
        Storage.save_result(WarpResult("horizon", data={"has_horizon": has_horizon}))
        Input.pause()


@module("optimize", "Optimize", "Analysis")
class Optimize(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("OPTIMIZE")
        Terminal.blank()
        Terminal.text("Min energy config")
        Terminal.blank()
        Terminal.line()
        v_c = Input.number("Target c", 1.0, 0.01, 10)
        R_min = Input.number("Min R m", 10, 1, 1000)
        R_max = Input.number("Max R m", 1000, 10, 10000)
        v = v_c * C
        results = []
        Terminal.blank()
        Terminal.text("Scanning...")
        for R in np.linspace(R_min, R_max, 8):
            for sigma in [1, 2, 5, 10, 20]:
                E = abs(Physics.total_energy(v, R, sigma))
                results.append((R, sigma, E))
        results.sort(key=lambda x: x[2])
        Terminal.blank()
        Terminal.line()
        Terminal.box("TOP CONFIGS")
        Terminal.blank()
        rows = [[f"{R:.0f}", f"{s:.0f}", Formatter.sci(E)] for R, s, E in results[:6]]
        Terminal.table(["R(m)", "sigma", "Energy(J)"], rows)
        Terminal.blank()
        best = results[0]
        Terminal.row("Optimal R", f"{best[0]:.1f}m")
        Terminal.row("Optimal s", f"{best[1]:.1f}")
        Terminal.row("Min E", Formatter.energy(best[2]))
        Terminal.blank()
        Storage.save_result(WarpResult("optimize", data={"best_R": best[0], "best_E": best[2]}))
        Input.pause()


@module("converter", "Converter", "Tools")
class Converter(Module):
    CONVERSIONS = {
        "dist": {
            "units": ["m", "km", "AU", "ly", "pc"],
            "to_base": {"m": 1, "km": 1e3, "AU": AU, "ly": LY, "pc": PC},
        },
        "time": {
            "units": ["s", "min", "hr", "d", "yr"],
            "to_base": {"s": 1, "min": 60, "hr": 3600, "d": 86400, "yr": YEAR_S},
        },
        "mass": {
            "units": ["kg", "Me", "Mj", "Ms"],
            "to_base": {"kg": 1, "Me": M_EARTH, "Mj": M_JUPITER, "Ms": M_SUN},
        },
        "velocity": {
            "units": ["m/s", "km/s", "c"],
            "to_base": {"m/s": 1, "km/s": 1e3, "c": C},
        },
    }

    def run(self):
        Terminal.clear()
        Terminal.box("CONVERTER")
        Terminal.blank()
        cat = Input.select("Category", {k: k for k in self.CONVERSIONS})
        conv = self.CONVERSIONS[cat]
        Terminal.blank()
        Terminal.text(f"Units: {conv['units']}")
        val = Input.number("Value", 1.0)
        from_u = Input.choice("From", conv['units'])
        base = val * conv['to_base'][from_u]
        Terminal.blank()
        Terminal.line()
        Terminal.box("RESULT")
        Terminal.blank()
        for u in conv['units']:
            converted = base / conv['to_base'][u]
            Terminal.row(u, Formatter.sci(converted))
        Terminal.blank()
        Input.pause()


@module("constants", "Constants", "Tools")
class Constants(Module):
    CONST_DATA = [
        ("c", C, "m/s"),
        ("G", G, "m3/kg/s2"),
        ("h", H, "J*s"),
        ("hbar", HBAR, "J*s"),
        ("k_B", K_B, "J/K"),
        ("AU", AU, "m"),
        ("ly", LY, "m"),
        ("pc", PC, "m"),
        ("M_Earth", M_EARTH, "kg"),
        ("M_Jupiter", M_JUPITER, "kg"),
        ("M_Sun", M_SUN, "kg"),
        ("Planck_l", np.sqrt(HBAR * G / C ** 3), "m"),
        ("Planck_t", np.sqrt(HBAR * G / C ** 5), "s"),
        ("Planck_m", np.sqrt(HBAR * C / G), "kg"),
    ]

    def run(self):
        Terminal.clear()
        Terminal.box("CONSTANTS")
        Terminal.blank()
        rows = [[n, Formatter.sci(v), u] for n, v, u in self.CONST_DATA]
        Terminal.table(["Name", "Value", "Unit"], rows)
        Terminal.blank()
        Input.pause()


@module("history", "History", "Tools")
class History(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("HISTORY")
        Terminal.blank()
        history = Storage.load_history()
        if not history:
            Terminal.text("No history")
            Input.pause()
            return
        Terminal.text(f"{len(history)} records")
        Terminal.blank()
        rows = [[e.get("timestamp", "")[:16], e.get("module", "?")] for e in history[-12:]]
        Terminal.table(["Time", "Module"], rows)
        Terminal.blank()
        if Input.confirm("Clear history?"):
            Storage.clear_history()
            Terminal.text("Cleared")
        Input.pause()


@module("config", "Config", "Tools")
class Config(Module):
    def run(self):
        Terminal.clear()
        Terminal.box("CONFIG")
        Terminal.blank()
        cfg = Storage.load_config()
        for k, v in cfg.items():
            Terminal.row(k, str(v))
        Terminal.blank()
        Terminal.line()
        if Input.confirm("Edit config?"):
            cfg["precision"] = Input.integer("Precision", cfg.get("precision", 4), 1, 10)
            cfg["warp"] = Input.number("Default warp", cfg.get("warp", 5.0), 1, 9.99)
            cfg["dist"] = Input.number("Default dist", cfg.get("dist", 4.24), 0.01, 1e6)
            Storage.save_config(cfg)
            Terminal.text("Saved")
        Input.pause()


class App:
    CATEGORIES = ["Velocity", "Energy", "Geometry", "Dynamics", "Analysis", "Tools"]

    def menu(self):
        Terminal.clear()
        Terminal.box("WARP DRIVE CALC")
        Terminal.blank()
        grouped = {c: [] for c in self.CATEGORIES}
        for key, mod in Module.registry.items():
            grouped.setdefault(mod.category, []).append((key, mod))
        idx = 1
        key_map = {}
        for cat in self.CATEGORIES:
            modules = grouped.get(cat, [])
            if not modules:
                continue
            Terminal.text(cat.upper())
            for key, mod in modules:
                Terminal.text(f"  {idx:2}. {mod.name}")
                key_map[str(idx)] = key
                idx += 1
            Terminal.blank()
        Terminal.text(" q. Quit")
        Terminal.blank()
        return key_map

    def run(self):
        while True:
            key_map = self.menu()
            choice = input("  Select: ").strip().lower()
            if choice == 'q':
                Terminal.clear()
                Terminal.box("OFFLINE")
                sys.exit(0)
            if choice in key_map:
                Module.registry[key_map[choice]].run()


if __name__ == "__main__":
    App().run()
