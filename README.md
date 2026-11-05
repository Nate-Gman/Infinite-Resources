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

- **One file:** [`Radiant.py`](Radiant.py) — ~13,000 lines, 409‑check self‑test.
- **Dependencies:** `numpy`, `matplotlib`, `pygame` (scipy optional; pygame only for the 3D viewer).

---

## Quick start

```bash
python3 -m pip install numpy matplotlib pygame
python3 Radiant.py            # opens the interactive 3D MACHINE viewer (default)
python3 Radiant.py --selftest # 417 physics/geometry/powerplant unit checks + conservation audit
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
| **X** | SHOWCASE one field cell in **ultra detail, to scale** (Penning‑trap harvester): the complete real instrument — **41 evidence‑tagged parts** in 5 subsystems (electrode stack incl. segmented rotating‑wall sectors, gates, Surko accumulator, collectors; source beamline incl. W target, degrader, Ne remoderator, steering coils, dump, proton feed; magnet & cryo incl. solenoid, mirror/shim coils, MLI, LHe turret, HTS leads, quench dump, burst disk, NMR probe, thermometry; vacuum & diagnostics incl. ion pump, gate valve, NEG, RGA, 511 keV scintillator ring, image‑current amp, RW‑RF, feedthroughs, standoffs, readout; structure) — live **leader‑line callouts** tracking every part, animated drive beam + target flash + e⁻/e⁺ gyro‑separation + proton feed + residual 511 keV streaks inside the bore, a derived **TRAP NUMBERS** sheet (140 GHz cyclotron, 0.67 µm gyro radius, ħω>kT ground state, Brillouin limit, gas‑lifetime, mirror loss cone, TeV‑per‑e⁺ economics — all tagged), an **AXIAL V(z)** inset, B=cutaway |
| **T** | LIVE TIME‑LAPSE: seed craft → void → replicate → galaxy (cinematic, pausable, loopable). Era banner carries a live physics line (probe speed, replication rate, wavefront, survival odds — all model telemetry); bottom strip shows the **era‑segmented log timeline** + per‑frame sim‑time step (`+985.7 yr (log‑compressed)`); **FABRICATION TIMESCALE panel** derives the real floors live: minting 1e38 kg from starlight is bounded at **7.44 Gyr @100% capture** (η=1%→744 Gyr), causality bounds expansion at **0.50 Myr** to the far edge, and a consistency check shows the sim's structure would need >100% of all starlight if radiation‑minted — proving growth is ISRU baryon reprocessing, not radiation |
| **V** | STATS: all 8 analysis figures **inside the viewer** (twin, physics, galactic, trap, profit, powerplant, gold, creation) — 1‑8 / `[` `]` / click to switch |
| **L** | toggle LOOP mode for the live time‑lapse (default ON — seamless cycle repeat) |
| **SPACE** | pause/resume the live time‑lapse (or pause the normal viewer); while paused **RIGHT** single‑steps one frame (forward only — the sim has no rewind) |
| **[** and **]** | pick the **output element** (spewed as a cloud) + its synthesis proof, or cycle fuels in POWERPLANT view |
| **I** | ABOUT / VALIDITY panel (argues the science from known physics) |
| **P** | cycle the on‑screen **math proof** (verified live) |
| **1–9, 0 / TAB / click** | in CHEMISTRY view, fly the camera **into** a stage's real mechanism |
| **W** | POWERPLANT view: typed fuel → continuous power + GOLD (3D hardware, live telemetry) |
| **F** | TYPE A FUEL: live typed‑fuel entry for the powerplant (resolves any alias) |
| **A** | MAXIMIZE GOLD (Au): toggle the gold‑maximized blanket in the POWERPLANT tab |
| **R** | reset view |
| **ESC** | exit mode / quit |

The window opens at your desktop resolution (up to 1920×1080). Every panel — telemetry,
info/about, math proof, total‑system scale ladder — updates live from the real physics engine.

Headless equivalents:

```bash
python3 Radiant.py --render-frames 8   # save annotated MACHINE/HERO/CHEMISTRY/POWERPLANT frames (PNG)
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
9. **THE RADIANT POWERPLANT** (Section 33) — a continuously‑operating fusion powerplant that
   burns a **typed fuel** (D‑T, boron, seawater, hydrogen — anything the operator types),
   exports electricity to a grid, and precipitates **stable Au‑197** out of a mercury blanket
   via the real Hg‑198(n,2n)Hg‑197 → electron capture → Au‑197 nuclear route (Sherr,
   Bainbridge & Anderson, Phys. Rev. 1941). Every failure mode is an **engineering hurdle**:
   unknown fuel → substitute (H8), unrunnable fuel → substitute (H9), aneutronic fuel → D‑T
   neutron pilot (H10), tritium breeding → gold duty as neutron multiplier (H11), natural
   mercury → Be/Pb multiplier layer (H11b), unusable feedstock → forced enrichment (H11c),
   activated bullion → shielded decay + assay hold (H12), market saturation → fleet ceiling
   (H13). The invariant: **gold is the
   product**, for any typed input. Proofs P17–P22 verify the physics live.
   The POWER tab also meters the full **A=197 electron‑capture cascade** —
   Pb‑197 → Tl‑197 → Hg‑197 → Au‑197 (every link a real isotope, Z 82→79) —
   with live tank inventories, per‑stage decay heat (~38 MWth recovered), and
   the honest cost of feeding it (~63 MWe of spallation beam): the cascade is
   a gold machine, not a free power source.

---

## PROOF: mercury → gold, at mass scale

**Full derivation and citations: [About.md](About.md).** The condensed proof:

### It is nuclear, not chemical — and that distinction is the whole point

**Chemistry cannot make gold.** Chemistry is the physics of *electrons*; the identity of an
element is set by **Z**, the proton count. Chemistry never changes Z, so it can never change one
element into another. That is precisely why alchemy failed for eighteen centuries — the
alchemists were using the wrong force.

The nuclear force changes Z. Mercury is Z = 80, gold is Z = 79, and the gap is one proton:

```
D + T  →  n (14.06 MeV)                                    [Q = 17.589 MeV, AME2020]
¹⁹⁸Hg + n  →  ¹⁹⁷Hg + 2n          threshold 8.5286 MeV, σ = 2.1 b   [ENDF/B-VIII.0 MT=16]
¹⁹⁷Hg  --(electron capture, t½ = 64.14 h, 100 %)-->  ¹⁹⁷Au         [ENSDF A=197]
```

**¹⁹⁷Au is the only stable isotope of gold.** The product is indistinguishable from mined gold,
because it *is* gold. Chemistry then does what chemistry is good at — retort distillation
separates Au from Hg (357 °C vs 2,970 °C boiling points), the mercury is recycled to the
blanket, and the bullion is assayed and cast. That back end is **TRL 9, ordinary industry.**

### Threshold vs Q-value — the distinction the ledger depends on

Two numbers describe ¹⁹⁸Hg(n,2n), and using the wrong one silently corrupts the energy books:

```
|Q|   = 8.4854 MeV     energy ABSORBED (converted to nuclear mass)  → the HEAT LEDGER
E_th  = 8.5286 MeV     minimum neutron KE for the reaction to GO    → the GATE
```

They differ by the **recoil** the products must carry to conserve momentum — energy that is
kinetic, thermalizes in the blanket, and is still collected. Charging it to the heat budget
bills the plant for energy it never lost.

The threshold is **derived** from |Q| (`E_th = |Q|·(mₙ+M)/M`), not stored separately, because
two literals describing one reaction will eventually disagree — as these had, by 0.017 MeV.
Ten self-test checks guard the relationship. *(Full account: [About.md §11b](About.md).)*

### The books balance exactly

```
nucleons:  198 + 1  =  197 + 2   →   199 = 199        ✓
charge:         80  =       80                        ✓
lepton:    e⁻ (+1)  →  νₑ (+1)                        ✓
mass:  m(¹⁹⁷Au) = 3.2713e−25 kg  <  m(¹⁹⁸Hg) = 3.2879e−25 kg
```

**1 kg of gold requires 1.005 kg of mercury in.** This is a refinery, not a matter source — no
mass is created, nucleons are rearranged. *(Proof P21.)*

### Why the yield is tonnes, not grams

Mercury barely slows a fast neutron: ξ = 2/(A + ⅔) = **0.010067** per elastic collision, so

```
k = ln(14.06 / 8.5286) / ξ = 49.7 collisions above threshold
```

The neutron gets **~50 independent chances** to make gold before it degrades below the gate.

### Gold *pays its own neutron bill*

(n,2n) is **one neutron in, two out** — a multiplier. Per source neutron:

```
produced = 1 + x + m        required = TBR + parasitic = 1.05 + 0.15 = 1.20
self-sufficient  ⟺  x ≥ 0.20 − m
```

**The plant must make gold in order to breed its own tritium.** Gold is not a parasite on the
neutron budget; it is the precondition for closing it. *(Proof P18.)*

And it cannot run away: after one (n,2n) the daughters share 14.06 − 8.4854 = 5.57 MeV, i.e.
**~2.79 MeV each — far below the 8.5286 MeV threshold.** One neutron makes at most one gold atom.
*(Proof P19.)*

### The scale, in six lines of arithmetic

```
net 1,000 MWe / (1 − 0.12 recirc) / 0.42 Brayton  =  2,705.6 MWth
Q_thermal = 17.589 + 5.023 − 4.243 + 1.0          =  19.369 MeV  = 3.1013e−12 J
fusion rate = 2.7056e9 / 3.1013e−12               =  8.7184e20 /s
gold atoms  = 8.7184e20 × 1.0 × 0.500             =  4.3592e20 /s
mass        = 4.3592e20 × 3.271262e−25            =  1.42602e−4 kg/s
annual @ 85 % availability                        =  3,825.1 kg/yr
```

| One 1 GWe unit produces, continuously | |
|---|---|
| **Gold** | **10.47 kg/day · 3.83 tonnes/yr · 307 Good Delivery bars** |
| Value | $557.4 M/yr at $4,532.80/ozt |
| **And electricity** | **690,492 homes, or 10.56 cities of 50,000** |
| Mercury consumed | 3,844 kg/yr |
| Energy cost of the gold | 484.9 MWh/kg → **6.7× margin, profitable but not free** |

### Scaled to the world as it is today

```
3.6 TW world electricity = 3,600 units → 13,771 t gold/yr → 4.17× world mine production
```

**If every watt the world generates came from this plant, it would also mint 4.17× the entire
world gold mining industry every year** — 6.24 % of all gold ever mined, annually.

### Maximized — and the wall that stops you

3.83 t/yr is a conservative setpoint, not a limit. Same reactor, same site, same grid:

| Configuration | x | Gold t/yr | vs base | Fuel |
|---|---|---|---|---|
| baseline | 0.5000 | 3.825 | 1.000× | 1.000× |
| run the blanket at its cross-section ceiling | 0.5478 | 4.281 | 1.119× | 1.021× |
| + enrich to 100 % Hg-198 | 0.6087 | 4.890 | 1.279× | 1.050× |
| **+ thicken the blanket, add a reflector** | **0.6977** | **5.843** | **1.528×** | 1.095× |
| *asymptote (unbuildable)* | *0.7241* | *6.143* | *1.607×* | *1.109×* |

**5.84 t/yr — 470 bullion bars, $851.6 M/yr — for 1.095× the fuel, with electricity unchanged
at 1,000 MWe.** At a fixed thermal rating, gold ∝ x / Q_thermal(x); the grid does not pay for
the extra gold, the fuel bill does (deuterium is seawater, tritium is bred on site).

**The wall:** `x ≤ σ(n,2n)/σ_nonelastic = 2.1/2.9 = 0.7241`. Of the nonelastic budget, 0.8 b
goes to (n,γ), (n,p) and (n,α) and returns something that is not gold — **27.6 % of neutrons are
physically unavailable.** No lever moves it. *(Proof P23; the self-test sweeps every enrichment
× leakage combination to confirm nothing exceeds it.)*

**The catch:** more gold per unit means *fewer* units before H13 bites — the fleet ceiling
tightens from ~862 to ~565 units (1.53× fewer). Physical maximization and fleet-scale economics
point in opposite directions: maximize the blanket for a few units; at thousands, the
electricity is the product and the conservative duty is the better business.

World scale at the maximized duty: 3,600 units → **21,036 t/yr = 6.37× world mine production.**

```bash
python3 Radiant.py --maximize-gold     # the full ladder, with costs
```

### It has been done

> **Sherr, Bainbridge & Anderson (1941), *Phys. Rev.* 60, 473** — fast neutrons on mercury,
> gold chemically recovered. The reaction is not a proposal; what it lacked was a cheap,
> continuous neutron source. A D-T reactor is one, at 8.7 × 10²⁰ neutrons/second.

### ⚠ And what is NOT proven

| | TRL | |
|---|---|---|
| Liquid-Hg loop · gold refining/assay | **9** | SNS runs one since 2006; ordinary industry |
| Hg-198 enrichment | **7** | Industrial for Hg-196/202 today |
| ¹⁹⁸Hg(n,2n) transmutation | **4** | Measured & demonstrated 1941; never industrialised |
| **D-T fusion power island (net electric)** | **4** | **⚠ THE WEAKEST LINK** |
| Tritium breeding blanket | **3** | None has ever operated in a power reactor |

**No device has ever produced net electrical power from fusion.** NIF reached scientific
ignition in 2022 — that is not a plant putting megawatts on a grid. **Every gold figure above is
conditional on a D-T power plant existing.** The transmutation chain is the mature half of this
design; the reactor is the open problem. Nothing is over-unity — but it is not shovel-ready.

### Check every claim yourself

```bash
python3 Radiant.py --powerplant   # full derivation + radiant_powerplant.png
python3 Radiant.py --proofs       # P17–P22 verify the gold chain live
python3 Radiant.py --selftest     # 409 checks
python3 Radiant.py --trl          # the maturity audit above, unfiltered
python3 Radiant.py --maximize-gold # push the blanket to its cross-section ceiling
python3 Radiant.py --fleet        # scaled to world electricity + the gold market
python3 Radiant.py                # press W for the live POWERPLANT tab
```


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
| `--report` | full twin run + audit + all reports + `radiant_emhl.png`, `radiant_physics.png` (the same sheets the **V STATS** tab renders inside the viewer — no detached windows there) |
| `--selftest` | 409 physics/geometry/powerplant/audit unit checks + extended honesty audit |
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
| `--timescale` | TIME TO MAKE: honest years‑to‑matter per channel — raw ambient pairs [MEAS] vs ×10⁹ enhancer [*FTR] vs ISRU forge [MODEL], benchmarked against the universe's age |
| `--elements` | production route for every element Z=1–92: recombination/fusion [MEAS], Li/Be/B spallation [MEAS], r‑process [MODEL], Z>92 none [*FTR] |
| `--powerplant [FUEL]` | THE RADIANT POWERPLANT: type a fuel → continuous power + GOLD, rated vs real cities & gold market → `radiant_powerplant.png` |
| `--fuel FUEL` | alias for `--powerplant FUEL` (e.g. `--fuel boron`, `--fuel seawater`) |
| `--fuels` | list every fusion fuel the powerplant accepts (keys, aliases, Q, neutron energy, difficulty) |
| `--maximize-gold` | push the blanket to the cross-section ceiling; the full lever ladder with costs |
| `--fleet` | scale the powerplant to the world today: grid coverage, gold output vs world mine supply, fleet ceiling |
| `--mwe MW` | powerplant net electrical output in MW (default 1000) |
| `--enrichment F` | Hg‑198 enrichment 0–1 (default 0.90; 0.0997 = natural mercury → triggers H11b) |
| `--gold-price USD/oz` | override the gold spot price |
| `--years Y` | years of continuous powerplant operation to simulate (default 1) |
| `--hurdles` | every physical wall → mitigation → **maximized** function |
| `--galactic` | von‑Neumann galactic/cosmic expansion → `radiant_galactic.png` |
| `--physics` | the physics/chemistry figure (`radiant_physics.png`) |
| `--qed`, `--feasibility`, `--sweep` | QED reality check · engineering feasibility · parameter sweep |
| `--no-show` | save figures to PNG without opening a window |

---

## The honesty layer

Every run is checked by `honesty_audit()` (6 original checks) plus `extended_honesty_audit()`
(21 additional checks covering swarm, riser, intelligence, galactic, probe, and closed‑loop
conservation) and the 409‑check `--selftest` suite (covering powerplant physics, fuel catalog, neutron economy, transmutation blanket, viewer rendering, and conservation):

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

See **[About.md](About.md)** for the full, cited proof that this makes gold at mass scale,
**[Overview.md](Overview.md)** for the section‑by‑section technical index (including Section 33: the powerplant),
**[narrative.md](narrative.md)** for the full project status and roadmap, and
**[ProjectGoal.md](ProjectGoal.md)** for the original vision this twin realizes.

---

*Built to be technically rigorous yet visionary — grounded in pair production, strong‑field QED,
nuclear binding (Gamow tunnelling, Hoyle resonance, measured S‑factors), thermodynamics, ISRU,
self‑replication, Tsiolkovsky propulsion, and honest about every limit — including which limits
are "we haven't built it yet" versus "no known engineering path at all."*
