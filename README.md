# RADIANT — Eternal Matter Harvester Lattice (EMHL)

A single‑file, to‑scale **digital twin** of the *Eternal Matter Harvester Lattice*: a
self‑replicating megastructure that harvests radiation into usable matter, conditions it
with real chemistry, and grows toward galactic scale — modeled in **real physics (SI units)**,
rendered as a **navigable, interactive 3D machine**, and checked by an on‑board
**conservation + Second‑Law audit**.

**The primary objective: maximum matter creation, as fast as possible, as efficiently as
possible.** Every stage is pushed to its physical ceiling, every loss minimized, every rate
maximized. The system is fully autonomous — it self‑builds, self‑seeds, self‑replicates, and
self‑repeats the entire matter‑creation cycle from void to galaxy.

> **Honest by design.** Built in the spirit of the reference monoliths (`Main.py`,
> `GmansRunV1.17.py`, `Simulation.py`): every extraordinary claim is grounded in a cited
> formula and a conservation check. No vacuum mining, no over‑unity, no global entropy
> decrease. Where the literal goal meets a hard physical limit, that limit is treated as an
> **engineering hurdle** and pushed to the ceiling real physics allows — and flagged clearly
> where it crosses into frontier science.

- **One file:** [`Radiant.py`](Radiant.py) — ~9,650 lines, 217‑check self‑test.
- **Dependencies:** `numpy`, `matplotlib`, `pygame` (scipy optional; pygame only for the 3D viewer).

---

## Quick start

```bash
python3 -m pip install numpy matplotlib pygame
python3 Radiant.py            # opens the interactive 3D MACHINE viewer (default)
python3 Radiant.py --selftest # 217 physics/geometry unit checks + conservation audit
python3 Radiant.py --run      # autonomous closed loop: seed -> galaxy (8 systems, ~2 sec)
python3 Radiant.py --efficiency  # pipeline throughput: achieved vs theoretical-max per stage
python3 Radiant.py --trl      # Technology Readiness Level audit (NASA 9-level scale)
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
| **T** | LIVE TIME‑LAPSE: seed craft → void → replicate → galaxy (cinematic, pausable, loopable) |
| **L** | toggle LOOP mode for the live time‑lapse (default ON — seamless cycle repeat) |
| **SPACE** | pause/resume the live time‑lapse (or pause the normal viewer) |
| **[** and **]** | pick the **output element** (spewed as a cloud) + its synthesis proof |
| **I** | ABOUT / VALIDITY panel (argues the science from known physics) |
| **P** | cycle the on‑screen **math proof** (verified live) |
| **1–9, 0 / TAB / click** | in CHEMISTRY view, fly the camera **into** a stage's real mechanism |
| **R** | reset view · **ESC** | exit mode / quit |

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
   **Gamow‑gated** at the plant's real fusion temperature); **r‑process** neutron capture past
   iron (gold, uranium). Triple‑alpha (He→C) uses the real **Hoyle‑state narrow‑resonance rate**
   (deBoer et al. 2017), not Coulomb tunnelling — ignites at ~1e8 K like real stellar cores.
   C12(α,γ)O16 uses the **measured S‑factor** (deBoer et al. 2017, S(300keV)=140 keV·b).
5. **Create matter** — energy→mass across every real channel to the **E=mc²** ceiling; net
   baryons via an engineered **baryogenesis** reactor (Sakharov + sphalerons — flagged frontier).
6. **Self‑power & profit** — fusing a slice of harvested H returns ~10⁷× its organizing cost →
   the plant **self‑powers**, external demand → 0, EROI and profit **outscale**.
7. **Organize & replicate** — ISRU structuring + von‑Neumann replication → exponential growth.
   Replication rate is **power‑constrained** (not just assembly‑limited): a station orbiting a
   dim M dwarf genuinely replicates slower than one orbiting a bright A/B star.
8. **Expand** — colonization wavefront at **0.1c** (Tsiolkovsky‑costed probe propulsion, ISM
   transit survival probability) to the **ΛCDM reachable horizon** → **~51 galaxies of
   matter** total.

### The autonomous closed loop (void → galaxy)

The system models a **true closed loop**: a seed probe arrives at a star → bootstraps
(~87 days: deceleration → deployment → first harvest → first matter → first fusion → first
replication) → swarm grows exponentially → saturates at carrying capacity → launches real
Tsiolkovsky‑costed seed probes → each surviving probe spawns a new `SystemColonizer` at the
next star → **the cycle repeats, faster each generation** (evolved parameters propagate via
seed data). Key systems:

- **`SystemColonizer`** — one colonized star system with its own swarm, stockpile, riser,
  intelligence, bootstrap sequence, and probe‑launch bookkeeping.
- **`ConstructionPhaseManager`** — autonomously selects target elements based on swarm state
  (H for fuel during bootstrap, Fe for structure during growth, Au/U for probes at saturation).
  No human picks the element — the machine decides.
- **`PipelineBalancer`** — measures every plant stage's achieved rate against its theoretical‑max
  ceiling, identifies the bottleneck, and auto‑adjusts parameters to unblock it.
  `total_matter_kg_per_s` is THE key metric.
- **`StellarType`** — real stellar classes (M dwarf through Giant) with real mass‑luminosity‑
  lifetime physics and Kroupa‑IMF demographics (~76% M dwarfs). Each colonized system orbits a
  real, demographically‑weighted star.
- **`BootstrappingSequence`** — 10‑phase deployment timeline from probe arrival to first
  replication; gates swarm growth until bootstrap completes.
- **Station failure modes** — real ~1.5%/yr all‑cause failure rate (Aon/XL Insurance,
  Seradata/SpaceTrak database), competing against replication every step.

**Honest coda:** structured matter still radiates away (proton τ > 10³⁴ yr), so "eternal" means
a **dynamic equilibrium** — the machine must keep harvesting (it out‑paces decay by ~10³³×).

---

## Command reference

### Viewer / rendering
| Command | Does |
|---|---|
| `python3 Radiant.py` | interactive 3D **machine** viewer (default; press **T** for live time‑lapse) |
| `--view-cell` | 3D viewer zoomed to the live particle separation cell |
| `--render-frames [N]` | headless: save N annotated 3D frames (`radiant_system_*.png`) |
| `--render-cell [N]` | headless: save N particle‑cell frames (`radiant_view_*.png`) |
| `--timelapse [N]` | headless: save N cinematic time‑lapse frames (`radiant_timelapse_*.png`) |

### Autonomous loop & throughput
| Command | Does |
|---|---|
| `--run [N]` | autonomous closed loop: seed → bootstrap → saturation → seed launch → galaxy
|              | (N systems in full physics, default 8; ~2 sec wall‑clock) |
| `--efficiency` | PipelineBalancer: achieved vs theoretical‑max throughput per stage, bottleneck + auto‑tune |
| `--trl` | Technology Readiness Level audit (NASA 9‑level scale) — which parts are flight‑proven vs speculative |

### Reports & analysis (text + figures)
| Command | Does |
|---|---|
| `--report` | full twin run + audit + all reports + `radiant_emhl.png`, `radiant_physics.png` |
| `--selftest` | 217 physics/geometry/audit unit checks + extended honesty audit |
| `--info` | explain every part + the math/chemistry + a **validity** argument |
| `--proofs` | print & verify the math proofs (P1–P12) |
| `--chemistry` | radiation → atoms → elements → solids (with real energies) |
| `--output` | custom chemistry output: any stable element/molecule + synthesis proof |
| `--target Z` | auto‑target element Z: machine auto‑adjusts to produce it (e.g. `--target 79` for Au) |
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

Every run is checked by `honesty_audit()` (6 original checks) plus `extended_honesty_audit()`
(21 additional checks covering swarm, riser, intelligence, galactic, probe, and closed‑loop
conservation) and the 217‑check `--selftest` suite:

- **No over‑unity:** created rest‑mass energy ≤ energy drawn (E=mc²).
- **Reservoir respected:** energy drawn ≤ the free‑energy budget.
- **Baryons conserved:** organized mass ≤ accessible baryons.
- **Second Law:** `dS_total = dS_local + dS_export ≥ 0` always.
- **Vacuum is not a free lunch:** Schwinger pairs/cell/age‑of‑universe ≈ 0.
- **Swarm bounded:** N ≤ K_stations (carrying capacity) and hard safety cap.
- **Galactic causality:** expansion speed ≤ c; systems ≤ reachable cosmos.
- **Probe Tsiolkovsky honesty:** launch mass ≥ payload × count (propellant is real).
- **Closed‑loop invariant:** seeds launched only after saturation, never before.
- **Bootstrap gates growth:** swarm held at seed population until bootstrap completes.

**Technology Readiness Level audit** (`--trl`): grades every subsystem on the NASA 9‑level
engineering‑maturity scale — a different question than "is the physics real." Most subsystems
rest on flight‑proven or lab‑demonstrated technology (pair sources, Penning traps, ISRU,
waste‑heat rejection, adaptive control: TRL 6–9). The weakest links are named explicitly:
engineered CP‑violation baryogenesis (TRL 0, no known engineering path), von Neumann
self‑replication (TRL 1, no working universal assembler), and fusion rocket propulsion
(TRL 3, no flight engine). The report states outright that "100% guaranteed" is not an honest
claim for a system with TRL 0–3 components.

**What is *not* claimed:** mining the vacuum for net matter, over‑unity energy, or lowering the
universe's entropy. The single flagged‑speculative knob is the **CP‑violation source** for
engineered baryogenesis; the mechanism around it (electroweak sphalerons) is textbook.

---

## Requirements & notes

- Python 3.9+; `numpy`, `matplotlib` required. `pygame` for the 3D viewer (`--view*`).
- `scipy` is optional and not required.
- Headless machines: use `--render-frames`, `--report --no-show`, and the text reports.

See **[Overview.md](Overview.md)** for the section‑by‑section technical index,
**[narrative.md](narrative.md)** for the full project status and roadmap, and
**[ProjectGoal.md](ProjectGoal.md)** for the original vision this twin realizes.

---

*Built to be technically rigorous yet visionary — grounded in pair production, strong‑field QED,
nuclear binding (Gamow tunnelling, Hoyle resonance, measured S‑factors), thermodynamics, ISRU,
self‑replication, Tsiolkovsky propulsion, and honest about every limit — including which limits
are "we haven't built it yet" versus "no known engineering path at all."*
