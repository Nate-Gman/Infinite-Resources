# RADIANT — Technical Overview

Authoritative feature index for [`Radiant.py`](Radiant.py) (~5,800 lines, single file).
Each entry names the **SECTION marker** (grep the text to jump), what it does, and the key
functions/classes. Conventions: PascalCase classes, snake_case functions, UPPER_SNAKE
constants, all physics in **SI units (CODATA 2018 / PDG 2024)**.

Run `python3 Radiant.py --selftest` (84 checks) to validate every subsystem.

---

## Table of contents (by SECTION marker)

| Section | Contents | Key names |
|---|---|---|
| **1 — PHYSICAL CONSTANTS** | CODATA/PDG constants + derived scales | `C_LIGHT, M_E, E_SCHWINGER, LAMBDA_C, M_E_C2, PAIR_ENERGY` |
| **2 — ENGINEERING SPECIFICATION** | to‑scale `SPEC` dict (cells, laser, trap, reservoir, swarm, efficiency) | `SPEC` |
| **3 — STRONG‑FIELD QED** | Schwinger rate, dynamically‑assisted, Breit–Wheeler, Bethe–Heitler | `schwinger_rate_density, dynamically_assisted_rate_density, field_from_intensity, breit_wheeler_threshold_ok, bethe_heitler_xsec, qed_pair_flux_per_cell` |
| **4 — MATTER CONDITIONING** | E=mc² accounting, fusion yield, ISRU cost | `energy_to_make_mass, nucleosynthesis_yield, isru_organize_energy` |
| **4b — HURDLE 1: antimatter factory** | Bethe–Heitler beam‑on‑target positron source | `positron_factory_yield` |
| **5 — THERMODYNAMICS & ENTROPY** | Landauer, structuring entropy, radiators, CMB temp | `landauer_cost, structuring_entropy, radiator_area, cmb_temperature` |
| **6 — SELF‑REPLICATING SWARM** | carrying capacity + logistic growth | `matter_capacity, logistic_step` |
| **7 — THE EMHL ENGINE** | stateful digital twin stepping in real SI time | `EMHLState, EternalMatterHarvesterLattice` |
| **8 — FORMATTERS** | human‑readable time/mass/energy | `_fmt_time, _fmt_mass, _fmt_energy` |
| **9 — HONESTY LAYER** | conservation + Second‑Law audit (refuses magic) | `honesty_audit` |
| **10 — QED REALITY CHECK** (`--qed`) | why the vacuum can't be mined | `qed_reality_check` |
| **11 — FEASIBILITY REPORT** (`--feasibility`) | to‑scale engineering pass/fail | `feasibility_report` |
| **12 — SELF‑TEST** (`--selftest`) | 84 physics/geometry/audit unit checks | `selftest` |
| **13 — PARAMETER SWEEP** (`--sweep`) | build‑time → reach‑time scan | `sweep` |
| **13b — HURDLE 2: galactic expansion** | von‑Neumann colonization wavefront | `galactic_expansion` |
| **13c — HURDLE 3: eternal computation** | Dyson reversible‑compute bound | `eternal_computation` |
| **13d — HURDLES LEDGER** (`--hurdles`) | every wall → mitigation → maximized | `engineering_hurdles_report` |
| **16 — REAL PARTICLE/NUCLEAR/CHEM DATA** | SEMF, measured BE, periodic table, NIST/Pauling | `PARTICLES, PERIODIC_TABLE, IONIZATION_ENERGY_EV, ELECTRONEGATIVITY, nuclear_binding_energy_MeV, gamow_fusion_probability, coulomb_barrier_MeV, EXPERIMENTAL_BE_MEV` |
| **17 — ANNIHILATION & POSITRONIUM** | fate of a harvested positron | `annihilation_energy_release, annihilation_cross_section, positronium_note` |
| **18 — AMBIENT FEEDSTOCK** (`--feedstock`) | CMB / cosmic‑ray / solar‑wind flux, to scale | `feedstock_flux_report, feedstock_report` |
| **19 — HARVESTING / SEPARATION / TRAP** | Penning/Brillouin/mirror/lifetime physics | `cyclotron_frequency, brillouin_density_limit, magnetic_mirror_loss_cone, trap_annihilation_lifetime` |
| **20 — CONDITIONING CHEMISTRY** (`--chemistry`) | Saha, recombination, nucleosynthesis net, bonds | `saha_ionization_fraction, radiative_recombination_rate, nucleosynthesis_network, bond_ionicity, binding_energy_curve, chemistry_report` |
| **21 — THE SUBSYSTEM PLANT** (`--parts`) | 12 stages, each with `.step()` + telemetry | `Subsystem, EMHLPlant, parts_report` |
| **22 — MATH PROOFS** (`--proofs`) | 12 verified theorems (P1–P12) | `MathProof, _proofs, proofs_report` |
| **23 — IN‑DEPTH REPORTS** | nuclear / info / validity report generators | `nuclear_report, info_report, validity_report` |
| **24 — PARTICLE‑LEVEL TRAP MICRO‑SIM** (`--trap`) | Boris‑pusher e⁺/e⁻/p/γ sim; annihilation→2γ, recombination | `TrapConfig, PenningTrapSim, trap_report` |
| **25 — HURDLE 4: BEYOND IRON** (`--rprocess`) | neutron capture → gold/uranium cost | `neutron_separation_energy_MeV, heavy_element_cost, rprocess_report` |
| **26 — MATTER‑CREATION ENGINE** (`--create`) | every energy→mass channel + E=mc² ceiling; galaxy target | `CREATION_CHANNELS, creation_rate_kg_s, mass_energy_ceiling_kg_s, galaxy_scale_target, matter_creation_report` |
| **28 — BARYOGENESIS ENGINE** (`--baryogenesis`) | Sakharov + sphalerons + engineered CP (flagged) | `sakharov_conditions, sphalerons_active, net_baryon_rate_kg_s, baryogenesis_report` |
| **29 — EFFICIENCY & PROFIT** (`--profit`) | fusion self‑power, EROI, profit outscaling | `fusion_energy_per_kg_H, self_power, eroi, net_external_power, profit_curve, efficiency_report` |
| **30 — MATTER MAINTENANCE vs RADIATION** | proton decay; dynamic‑equilibrium replenishment | `matter_decay_rate, maintenance_equilibrium` |
| **14 — VISUALIZATION** | matplotlib figures | `visualize, visualize_physics, visualize_galactic, visualize_creation, visualize_profit, visualize_trap` |
| **31 — MECHANICAL GEOMETRY & LATTICE** | solid meshes: cell, lattice, replicator, hero cell | `Assembly, _g_cylinder/_g_tube/_g_torus/_g_box/_g_cone/_g_helix/_cyl_between, build_field_cell_solid, build_harvester_lattice_solid, build_replicator_solid, build_hero_cell_solid, void_capture` |
| **32 — CUSTOM CHEMISTRY OUTPUT** (`--output`) | pick a stable element/molecule + synthesis proof | `OUTPUT_ELEMENTS, OUTPUT_MOLECULES, element_synthesis_proof, molecule_synthesis_proof, chemistry_output_report` |
| **27 / 27b — INTERACTIVE 3D VIEWERS** | pygame software renderer (like Main.py/GMRv1.17) | `RadiantViewer, SystemViewer, run_viewer` |
| **15 — MAIN** | CLI dispatch; default = 3D viewer | `main, _print_header, _print_viewer_banner` |

*(Section numbers reflect the file's build order; a few are out of numeric sequence — grep the
marker text to jump.)*

---

## The subsystem plant (SECTION 21)

`EMHLPlant.STAGES` — 12 stages, each conserving mass/energy, wired output→input:

```
DysonCollector → DriverLaser → MetamaterialCavity → PairSource → SeparatorTrap →
RecombinationReactor → Nucleosynthesizer → MatterCreationReactor → BaryogenesisReactor →
ISRUForge → EnergyRecycler → RadiatorBank
```

`--parts` prints every stage's live telemetry; the CHEMISTRY viewer lets you click/fly into each.

---

## Math proofs (SECTION 22) — `--proofs`

Each is stated, derived, and **verified in code** (wired into `--selftest`):

| # | Theorem |
|---|---|
| P1 | Schwinger critical field `E_cr = m_e²c³/(eℏ) ≈ 1.32×10¹⁸ V/m` |
| P2 | No vacuum mining (energy conservation; closed field cycle nets zero work) |
| P3 | Pair threshold `E = 2 m_e c² = 1.022 MeV` |
| P4 | Second Law / Landauer bound (`dS_tot ≥ 0`) |
| P5 | Mass‑flow / conservation audit (no double counting, no over‑unity) |
| P6 | Self‑replication: exponential → logistic |
| P7 | Dyson eternal computation (finite but vast) |
| P8 | Binding‑energy peak → fusion ends at iron |
| P9 | Gamow tunnelling → fusion needs temperature |
| P10 | Bekenstein bound (max structure in a region) |
| P11 | Sakharov conditions → net baryon creation is possible |
| P12 | Fusion self‑power → profit outscales; matter must be replenished |

---

## The 3D viewer (SECTIONS 27 / 27b / 31)

- **`SystemViewer`** — the main viewer. Modes: `machine` (default), `chemistry` (secondary),
  plus the `hero` cell showcase (X). Renders **solid, flat‑shaded, depth‑sorted, backface‑culled
  polygons** (`_render_solid`) — the same software‑renderer approach as `Main.py` / `GmansRunV1.17.py`.
- **Mechanical geometry (SECTION 31):** `Assembly` accumulates solid meshes; `build_field_cell_solid`
  is a detailed Penning trap (chamber + electrodes + solenoid coil + cavity); `build_hero_cell_solid`
  is the ultra‑detail showcase (9‑turn helix, correction rings, beam ports, flange bolts, to scale);
  `build_replicator_solid` is a factory building a copy; `build_harvester_lattice_solid` tiles the
  net/grid across space.
- **`RadiantViewer`** — the particle‑cell viewer (`--view-cell`), a live `PenningTrapSim`.

---

## Figures produced

`radiant_system_*.png` / `radiant_view_*.png` (3D frames) · `radiant_emhl.png` (twin) ·
`radiant_physics.png` (physics/chemistry) · `radiant_galactic.png` (expansion) ·
`radiant_trap.png` (particle sim) · `radiant_creation.png` (matter creation) ·
`radiant_profit.png` (efficiency/EROI).

---

## Honest‑physics stance (SECTION 9)

The twin refuses to report impossibilities. `honesty_audit()` enforces: no over‑unity, reservoir
& baryon conservation, `dS_total ≥ 0`, and that the vacuum yields ≈0 net pairs. Hard limits are
treated as **engineering hurdles** and maximized to the physical ceiling; the only
flagged‑speculative element is the **CP‑violation source** for engineered baryogenesis (the
surrounding mechanism — electroweak sphalerons — is textbook). Structured matter still radiates
away, so "eternal" means **replenished**, not static.

For the narrative vision this realizes, see [ProjectGoal.md](ProjectGoal.md); for usage, see
[README.md](README.md).
