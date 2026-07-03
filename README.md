# RADIANT — Eternal Matter Harvester Lattice (EMHL)

A single‑file, to‑scale **digital twin** of the *Eternal Matter Harvester Lattice*: a
self‑replicating megastructure that harvests radiation into usable matter, conditions it
with real chemistry, and grows toward galactic scale — modeled in **real physics (SI units)**,
rendered as a **navigable, interactive 3D machine**, and checked by an on‑board
**conservation + Second‑Law audit**.

> **Honest by design.** Built in the spirit of the reference monoliths (`Main.py`,
> `GmansRunV1.17.py`, `Simulation.py`): every extraordinary claim is grounded in a cited
> formula and a conservation check. No vacuum mining, no over‑unity, no global entropy
> decrease. Where the literal goal meets a hard physical limit, that limit is treated as an
> **engineering hurdle** and pushed to the ceiling real physics allows — and flagged clearly
> where it crosses into frontier science.

- **One file:** [`Radiant.py`](Radiant.py) — ~5,800 lines, 84‑check self‑test.
- **Dependencies:** `numpy`, `matplotlib`, `pygame` (scipy optional; pygame only for the 3D viewer).

---

## Quick start

```bash
python3 -m pip install numpy matplotlib pygame
python3 Radiant.py            # opens the interactive 3D MACHINE viewer (default)
python3 Radiant.py --selftest # 84 physics/geometry unit checks + conservation audit
python3 Radiant.py --report   # full digital-twin run + all reports + analysis figures
python3 Radiant.py --info     # what every part does + the math, chemistry & validity
```

Running with **no arguments** opens the live 3D world — exactly like `Main.py` /
`GmansRunV1.17.py`. If you have no display, use `--render-frames` (saves annotated PNGs) or the
text reports below.

---

## The 3D viewer

The default view is **THE MACHINE** — real, solid, flat‑shaded hardware to scale:

- **LEFT — Harvester Lattice:** a 3D net/grid of solid **Penning‑trap field cells** (vacuum
  chamber + ring/endcap electrodes + solenoid coil + metamaterial cavity) catching e⁺/e⁻ pairs.
- **CENTER — Replicator Factories:** von‑Neumann machines building copies of themselves.
- **RIGHT — Expansion:** new cells growing the net → *harvest → build → replicate → expand*.

### Controls

| Key / input | Action |
|---|---|
| **drag** / **wheel** / **arrow keys** | orbit / zoom / pan |
| **M** | MACHINE view (default: harvest / build / replicate) |
| **C** | CHEMISTRY flow view (secondary; the radiation→matter pipeline) |
| **X** | SHOWCASE one field cell in **ultra detail, to scale** (Penning‑trap harvester) |
| **[** and **]** | pick the **output element** (spewed as a cloud) + its synthesis proof |
| **I** | ABOUT / VALIDITY panel (argues the science from known physics) |
| **P** | cycle the on‑screen **math proof** (verified live) |
| **1–9, 0 / TAB / click** | in CHEMISTRY view, fly the camera **into** a stage's real mechanism |
| **R** | reset view · **ESC** | quit |

The window opens at your desktop resolution (up to 1920×1080). Every panel — telemetry,
info/about, math proof, total‑system scale ladder — updates live from the real physics engine.

Headless equivalents:

```bash
python3 Radiant.py --render-frames 8   # save annotated MACHINE/HERO/CHEMISTRY frames (PNG)
python3 Radiant.py --view-cell         # zoom to the live particle separation cell
python3 Radiant.py --render-cell 6     # headless frames of the particle cell
```

---

## What it models (all real physics, SI)

The full matter chain, each link grounded and each hard wall maximized:

1. **Harvest** — e⁺/e⁻ pairs from radiation via **Bethe–Heitler** beam‑on‑target and
   **Breit–Wheeler** γγ (real: SLAC E144, 1997). The vacuum itself can't be mined (Schwinger
   rate is exp‑suppressed — proofs P1–P3), so the real mechanism is maximized instead.
2. **Separate** — a **Penning trap** (Boris‑pusher particle sim) drives e⁺ and e⁻ to opposite
   electrodes so they **cannot annihilate** (measured ~50 % annihilation cut vs. control).
3. **Recombine** — `p + e⁻ → H + γ` radiative recombination (needs **protons** → solar wind).
4. **Build elements** — fusion up the binding curve to **iron** (measured AME2020 Q‑values,
   Gamow‑gated); **r‑process** neutron capture past iron (gold, uranium).
5. **Create matter** — energy→mass across every real channel to the **E=mc²** ceiling; net
   baryons via an engineered **baryogenesis** reactor (Sakharov + sphalerons — flagged frontier).
6. **Self‑power & profit** — fusing a slice of harvested H returns ~10⁷× its organizing cost →
   the plant **self‑powers**, external demand → 0, EROI and profit **outscale**.
7. **Organize & replicate** — ISRU structuring + von‑Neumann replication → exponential growth.
8. **Expand** — colonization wavefront to the **ΛCDM reachable horizon** → **~51 galaxies of
   matter** total.

**Honest coda:** structured matter still radiates away (proton τ > 10³⁴ yr), so "eternal" means
a **dynamic equilibrium** — the machine must keep harvesting (it out‑paces decay by ~10³³×).

---

## Command reference

### Viewer / rendering
| Command | Does |
|---|---|
| `python3 Radiant.py` | interactive 3D **machine** viewer (default) |
| `--view-cell` | 3D viewer zoomed to the live particle separation cell |
| `--render-frames [N]` | headless: save N annotated 3D frames (`radiant_system_*.png`) |
| `--render-cell [N]` | headless: save N particle‑cell frames (`radiant_view_*.png`) |

### Reports & analysis (text + figures)
| Command | Does |
|---|---|
| `--report` | full twin run + audit + all reports + `radiant_emhl.png`, `radiant_physics.png` |
| `--selftest` | 84 physics/geometry unit checks + conservation audit |
| `--info` | explain every part + the math/chemistry + a **validity** argument |
| `--proofs` | print & verify the math proofs (P1–P12) |
| `--chemistry` | radiation → atoms → elements → solids (with real energies) |
| `--output` | custom chemistry output: any stable element/molecule + synthesis proof |
| `--nuclear` | binding energies + Penning‑trap physics |
| `--feedstock` | what the ambient radiation *really* delivers (to scale) |
| `--parts` | every plant subsystem, live telemetry & mass/energy flow |
| `--live [TICKS]` | run the plant continuously (rolling dashboard) |
| `--trap [STEPS]` | particle‑level Penning‑trap micro‑sim (Boris pusher) → `radiant_trap.png` |
| `--create` | maximize matter creation (every energy→mass channel) → `radiant_creation.png` |
| `--baryogenesis` | break the net‑baryon barrier (Sakharov + sphalerons + CP) |
| `--profit` | efficiency & EROI: less power in, profit outscales → `radiant_profit.png` |
| `--rprocess` | build elements beyond iron (neutron‑capture cost) |
| `--hurdles` | every physical wall → mitigation → **maximized** function |
| `--galactic` | von‑Neumann galactic/cosmic expansion → `radiant_galactic.png` |
| `--physics` | the physics/chemistry figure (`radiant_physics.png`) |
| `--qed`, `--feasibility`, `--sweep` | QED reality check · engineering feasibility · parameter sweep |
| `--no-show` | save figures to PNG without opening a window |

---

## The honesty layer

Every run is checked by `honesty_audit()` and the `--selftest` suite:

- **No over‑unity:** created rest‑mass energy ≤ energy drawn (E=mc²).
- **Reservoir respected:** energy drawn ≤ the free‑energy budget.
- **Baryons conserved:** organized mass ≤ accessible baryons.
- **Second Law:** `dS_total = dS_local + dS_export ≥ 0` always.
- **Vacuum is not a free lunch:** Schwinger pairs/cell/age‑of‑universe ≈ 0.

**What is *not* claimed:** mining the vacuum for net matter, over‑unity energy, or lowering the
universe's entropy. The single flagged‑speculative knob is the **CP‑violation source** for
engineered baryogenesis; the mechanism around it (electroweak sphalerons) is textbook.

---

## Requirements & notes

- Python 3.9+; `numpy`, `matplotlib` required. `pygame` for the 3D viewer (`--view*`).
- `scipy` is optional and not required.
- Headless machines: use `--render-frames`, `--report --no-show`, and the text reports.

See **[Overview.md](Overview.md)** for the section‑by‑section technical index, and
**[ProjectGoal.md](ProjectGoal.md)** for the original vision this twin realizes.

---

*Built to be technically rigorous yet visionary — grounded in pair production, strong‑field QED,
nuclear binding, thermodynamics, ISRU and self‑replication, and honest about every limit.*
