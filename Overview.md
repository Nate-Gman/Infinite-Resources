# RADIANT — Technical Overview

Authoritative feature index for [`Radiant.py`](Radiant.py) (~12,550 lines, single file).
Each entry names the **SECTION marker** (grep the text to jump), what it does, and the key
functions/classes. Conventions: PascalCase classes, snake_case functions, UPPER_SNAKE
constants, all physics in **SI units (CODATA 2018 / PDG 2024)**.

**Primary objective: maximum matter creation, as fast as possible, as efficiently as possible.**
Every stage is measured against its theoretical‑max throughput; the bottleneck is identified
and auto‑tuned. The system is fully autonomous — self‑building, self‑seeding, self‑replicating,
self‑repeating from void to galaxy.

Run `python3 Radiant.py --selftest` (280+ checks) to validate every subsystem.

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
| **9 — HONESTY LAYER** | conservation + Second‑Law audit (refuses magic) | `honesty_audit, extended_honesty_audit` |
| **10 — QED REALITY CHECK** (`--qed`) | why the vacuum can't be mined | `qed_reality_check` |
| **11 — FEASIBILITY REPORT** (`--feasibility`) | to‑scale engineering pass/fail | `feasibility_report` |
| **12 — SELF‑TEST** (`--selftest`) | 400 physics/geometry/audit/nuclear/probe/closed‑loop/powerplant unit checks | `selftest` |
| **13 — PARAMETER SWEEP** (`--sweep`) | build‑time → reach‑time scan | `sweep` |
| **13b — HURDLE 2: galactic expansion** | von‑Neumann colonization wavefront | `galactic_expansion` |
| **13c — HURDLE 3: eternal computation** | Dyson reversible‑compute bound | `eternal_computation` |
| **13d — HURDLES LEDGER** (`--hurdles`) | every wall → mitigation → maximized | `engineering_hurdles_report` |
| **16 — REAL PARTICLE/NUCLEAR/CHEM DATA** | SEMF, measured BE, periodic table, NIST/Pauling | `PARTICLES, PERIODIC_TABLE, IONIZATION_ENERGY_EV, ELECTRONEGATIVITY, nuclear_binding_energy_MeV, gamow_fusion_probability, coulomb_barrier_MeV, EXPERIMENTAL_BE_MEV` |
| **31d4 — NARROW‑RESONANCE REACTION RATES** | Hoyle‑state triple‑alpha + measured S‑factor C12(α,γ)O16 (deBoer et al. 2017) | `nonresonant_thermonuclear_rate_cm3_s, narrow_resonance_rate_cm3_s, triple_alpha_rate_fraction, carbon_alpha_capture_rate_fraction` |
| **17 — ANNIHILATION & POSITRONIUM** | fate of a harvested positron | `annihilation_energy_release, annihilation_cross_section, positronium_note` |
| **18 — AMBIENT FEEDSTOCK** (`--feedstock`) | CMB / cosmic‑ray / solar‑wind flux, to scale | `feedstock_flux_report, feedstock_report` |
| **19 — HARVESTING / SEPARATION / TRAP** | Penning/Brillouin/mirror/lifetime physics | `cyclotron_frequency, brillouin_density_limit, magnetic_mirror_loss_cone, trap_annihilation_lifetime` |
| **20 — CONDITIONING CHEMISTRY** (`--chemistry`) | Saha, recombination, nucleosynthesis net, bonds | `saha_ionization_fraction, radiative_recombination_rate, nucleosynthesis_network, bond_ionicity, binding_energy_curve, chemistry_report` |
| **21 — THE SUBSYSTEM PLANT** (`--parts`) | 12 stages, each with `.step()` + telemetry | `Subsystem, EMHLPlant, parts_report` |
| **22 — MATH PROOFS** (`--proofs`) | 22 verified theorems (P1–P22, incl. powerplant P17–P22) | `MathProof, _proofs, proofs_report` |
| **23 — IN‑DEPTH REPORTS** | nuclear / info / validity report generators | `nuclear_report, info_report, validity_report` |
| **24 — PARTICLE‑LEVEL TRAP MICRO‑SIM** (`--trap`) | Boris‑pusher e⁺/e⁻/p/γ sim; annihilation→2γ, recombination | `TrapConfig, PenningTrapSim, trap_report` |
| **25 — HURDLE 4: BEYOND IRON** (`--rprocess`) | neutron capture → gold/uranium cost | `neutron_separation_energy_MeV, heavy_element_cost, rprocess_report` |
| **26 — MATTER‑CREATION ENGINE** (`--create`) | every energy→mass channel + E=mc² ceiling; galaxy target | `CREATION_CHANNELS, creation_rate_kg_s, mass_energy_ceiling_kg_s, galaxy_scale_target, matter_creation_report` |
| **28 — BARYOGENESIS ENGINE** (`--baryogenesis`) | Sakharov + sphalerons + engineered CP (flagged) | `sakharov_conditions, sphalerons_active, net_baryon_rate_kg_s, baryogenesis_report` |
| **29 — EFFICIENCY & PROFIT** (`--profit`) | fusion self‑power, EROI, profit outscaling | `fusion_energy_per_kg_H, self_power, eroi, net_external_power, profit_curve, efficiency_report` |
| **29b — PIPELINE BALANCER** (`--efficiency`) | achieved vs theoretical‑max throughput per stage; bottleneck ID + auto‑tune | `PipelineBalancer, pipeline_efficiency_report` |
| **30 — MATTER MAINTENANCE vs RADIATION** | proton decay; dynamic‑equilibrium replenishment | `matter_decay_rate, maintenance_equilibrium` |
| **14 — VISUALIZATION** | matplotlib figures | `visualize, visualize_physics, visualize_galactic, visualize_creation, visualize_profit, visualize_trap` |
| **31 — MECHANICAL GEOMETRY & LATTICE** | solid meshes: cell, lattice, replicator, hero cell | `Assembly, _g_cylinder/_g_tube/_g_torus/_g_box/_g_cone/_g_helix/_cyl_between, build_field_cell_solid, build_harvester_lattice_solid, build_replicator_solid, build_hero_cell_solid, void_capture` |
| **31b — SOLID GEOMETRY FOR EVERY PLANT STAGE** | 12 EMHLPlant stages as detailed solid mesh hardware | `build_*_solid` (one per stage) |
| **31c — ORBITAL STATIONS & NET‑CASTING** | area‑of‑effect harvesting platforms | `OrbitalStation, net_casting_area_effect` |
| **31d — SELF‑REPLICATING ORBITAL SWARM** | nanotech assembly + logistic growth + station failure | `SelfReplicatingOrbitalSwarm, ElementalStateRiser, SwarmIntelligence` |
| **31d2 — CONSTRUCTION PHASE MANAGER** | autonomous element auto‑selection (H→Fe→Au/U by phase) | `ConstructionPhaseManager` |
| **31d2b — PROBE PROPULSION & TRANSIT** | Tsiolkovsky rocket equation, ISM transit survival | `tsiolkovsky_mass_ratio, probe_propulsion_budget, probe_transit_physics, BootstrappingSequence` |
| **31d2c — STELLAR TYPES** | real per‑type luminosity, lifetime, Kroupa‑IMF demographics | `StellarType, STELLAR_TYPES, sample_stellar_type` |
| **31d3 — SYSTEM COLONIZER** | one colonized star system: swarm + riser + bootstrap + probes | `SystemColonizer, GalacticSeeding, autonomous_run_report` |
| **31e — AUTOMATED ELEMENT TARGETING** | machine auto‑adjusts parameters to produce target Z | `auto_target_element` |
| **32 — CUSTOM CHEMISTRY OUTPUT** (`--output`) | pick a stable element/molecule + synthesis proof | `OUTPUT_ELEMENTS, OUTPUT_MOLECULES, element_synthesis_proof, molecule_synthesis_proof, chemistry_output_report` |
| **9b — TECHNOLOGY READINESS LEVEL AUDIT** (`--trl`) | NASA 9‑level engineering maturity per subsystem | `TRL_AUDIT, trl_report` |
| **27 / 27b — INTERACTIVE 3D VIEWERS** | pygame software renderer + live time‑lapse (T key) | `RadiantViewer, SystemViewer, run_viewer, _timelapse_reset, _timelapse_frame, _render_timelapse_paused_overlay` |
| **33 — THE RADIANT POWERPLANT** (`--powerplant`, `--fuels`, `--fleet`) | typed‑fuel fusion plant: D‑T neutron economy, Hg‑198→Au‑197 transmutation, gold output, city/gold ratings, hurdles H8–H13 (incl. H11b/H11c) | `FusionFuel, FUSION_FUELS, resolve_fuel, neutron_economy, n2n_collisions_above_threshold, n2n_reaction_probability, gold_activation_state, RadiantPowerplant, powerplant_fuel_table, powerplant_report, powerplant_market_report, build_powerplant_complex_solid` |
| **15 — MAIN** | CLI dispatch; default = 3D viewer | `main, _print_header, _print_viewer_banner` |

*(Section numbers reflect the file's build order; a few are out of numeric sequence — grep the
marker text to jump. The 31x subsections were added during the Reality‑Tightening Pass and the
autonomous closed‑loop implementation.)*

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
| P17 | Mercury → gold is a MEASURED transmutation (Hg‑198(n,2n)Hg‑197 → EC → Au‑197) |
| P18 | D‑T neutron energy (14.06 MeV) exceeds the Hg‑198(n,2n) threshold (8.5286 MeV) |
| P19 | Neutron economy: 1 + x ≥ TBR + parasitic (tritium self‑sufficiency) |
| P20 | N2n multiplier ceiling: x ≤ σ(n,2n)/σ_total at enrichment (bounded by cross‑sections) |
| P21 | The powerplant creates no mass or energy (conservation audit) |
| P23 | Gold conversion capped at σ(n,2n)/σ_nonelastic = 0.724; 27.6% unavailable |
| P22 | Gold margin > 1: the gold is worth more than the electricity forgone to make it |

---

## The 3D viewer (SECTIONS 27 / 27b / 31)

- **`SystemViewer`** — the main viewer. Modes: `machine` (default), `chemistry` (secondary),
  `powerplant` (W key), plus the `hero` cell showcase (X). Renders **solid, flat‑shaded, depth‑sorted, backface‑culled
  polygons** (`_render_solid`) — the same software‑renderer approach as `Main.py` / `GmansRunV1.17.py`.
  **Live time‑lapse mode** (T key): a pausable, loopable cinematic playback of the full arc
  (seed craft → void → replicate → galaxy), reusing the same `_timelapse_frame` body as the
  headless `--timelapse` exporter. L toggles loop; SPACE pauses; ESC exits to normal viewer.
  **POWERPLANT mode** (W key): 3D hardware view of the fusion powerplant (fuel intake → tokamak →
  mercury blanket → decay tank → refinery/vault → power island) with animated streams and live
  telemetry. **F key** opens live typed‑fuel entry (type any alias: D‑T, boron, seawater, hydrogen…);
  ENTER commits and re‑commissions the plant on the new fuel. `[` / `]` cycle fuels.
- **Mechanical geometry (SECTION 31):** `Assembly` accumulates solid meshes; `build_field_cell_solid`
  is a detailed Penning trap (chamber + electrodes + solenoid coil + cavity); `build_hero_cell_solid`
  is the ultra‑detail showcase (9‑turn helix, correction rings, beam ports, flange bolts, to scale);
  `build_replicator_solid` is a factory building a copy; `build_harvester_lattice_solid` tiles the
  net/grid across space.
- **`RadiantViewer`** — the particle‑cell viewer (`--view-cell`), a live `PenningTrapSim`.

---

## Figures produced

`radiant_system_*.png` / `radiant_view_*.png` (3D frames) · `radiant_timelapse_*.png` (cinematic
frames) · `radiant_emhl.png` (twin) · `radiant_physics.png` (physics/chemistry) ·
`radiant_galactic.png` (expansion) · `radiant_trap.png` (particle sim) ·
`radiant_creation.png` (matter creation) · `radiant_profit.png` (efficiency/EROI) ·
`radiant_powerplant.png` (powerplant: enrichment, neutron budget, assay cooldown, fleet ceiling).

---

## Honest‑physics stance (SECTIONS 9 / 9b)

The twin refuses to report impossibilities. `honesty_audit()` (6 original checks) plus
`extended_honesty_audit()` (21 additional checks covering swarm, riser, intelligence, galactic,
probe, and closed‑loop conservation) enforce: no over‑unity, reservoir & baryon conservation,
`dS_total ≥ 0`, vacuum yields ≈0 net pairs, swarm bounded by carrying capacity, galactic
causality (≤c), Tsiolkovsky probe mass honesty, closed‑loop invariant (seeds only after
saturation), and bootstrap gates growth. Hard limits are treated as **engineering hurdles** and
maximized to the physical ceiling; the only flagged‑speculative element is the **CP‑violation
source** for engineered baryogenesis (the surrounding mechanism — electroweak sphalerons — is
textbook). Structured matter still radiates away, so "eternal" means **replenished**, not static.

The **TRL audit** (`--trl`, SECTION 9b) complements the physics audit by grading engineering
maturity on the NASA 9‑level scale: most subsystems are TRL 6–9 (flight‑proven or lab‑demonstrated);
the weakest links (CP‑violation baryogenesis TRL 0, von Neumann self‑replication TRL 1, fusion
rocket TRL 3) are named explicitly.

## The autonomous closed loop (SECTIONS 31d–31d3)

The system models a true **closed loop**: seed probe → bootstrap → swarm growth → saturation →
seed launch → new system → repeat. Key classes:

- **`SelfReplicatingOrbitalSwarm`** — logistic growth with power‑constrained replication rate,
  station failure (~1.5%/yr), and carrying capacity.
- **`ElementalStateRiser`** — temperature‑dependent nucleosynthesis with Gamow‑gated rates;
  triple‑alpha uses the Hoyle‑state narrow‑resonance rate (not Coulomb tunnelling).
- **`SwarmIntelligence`** — adaptive parameter optimization with physical bounds
  (arm_len_m ≤ 1e5 m, fusion_temp_K ≤ 1e10 K) to prevent overflow.
- **`ConstructionPhaseManager`** — autonomously selects target element based on swarm state.
- **`BootstrappingSequence`** — 10‑phase deployment timeline (probe arrival → first replication).
- **`StellarType`** — real stellar classes with Kroupa‑IMF demographics (~76% M dwarfs).
- **`SystemColonizer`** — one colonized star system with its own swarm, riser, intelligence,
  bootstrap, and probe‑launch bookkeeping.
- **`GalacticSeeding`** — orchestrates multi‑system expansion; `n_full_physics` systems get
  full physics, the rest are statistically aggregated.
- **`PipelineBalancer`** — measures every stage's achieved rate vs theoretical‑max, identifies
  the bottleneck, and auto‑adjusts parameters. `total_matter_kg_per_s` is THE key metric.

## THE RADIANT POWERPLANT (SECTION 33)

A continuously‑operating fusion powerplant that burns a **typed fuel** (anything the operator types —
D‑T, boron, seawater, hydrogen, …), exports electricity to a grid, and precipitates **stable Au‑197**
out of a mercury blanket via the real Hg‑198(n,2n)Hg‑197 → electron capture → Au‑197 nuclear route.

Key classes & functions:
- **`FusionFuel`** / **`FUSION_FUELS`** — catalog of 8 fusion fuels with Q, neutron energy, ignition
  temperature, specific energy, aliases, and difficulty rating.
- **`resolve_fuel`** — typed‑fuel parser: exact/normalized/token/substring matching with H8 fallback.
- **`neutron_economy`** — tritium breeding balance: 1 + x + multiplier ≥ TBR + parasitic.
- **`n2n_collisions_above_threshold`** / **`n2n_reaction_probability`** — mercury blanket physics.
- **`gold_activation_state`** — H12: bullion activity vs shippable clearance limit.
- **`HG198_N2N_Q_MEV`** (8.4854, AME2020) — the ONE measured nuclear input for the gold
  reaction. **`HG198_N2N_THRESHOLD_MEV`** is DERIVED from it as |Q|·(mₙ+M)/M = 8.5286, so the
  two cannot drift apart (they once had, by 0.017 MeV). Q drives the heat ledger; the
  threshold drives the reaction gate. **`BE9_N2N_Q_MEV`** (1.5735) does the same for the
  Be-9 multiplier.
- **`HG197M_HALFLIFE_S`** — the Hg-197m isomer (23.8 h). Both it and the 64.14 h ground state
  terminate on Au-197, so yield is independent of the isomeric ratio; the model keeps the
  slower ground state as the conservative choice.
- **`RadiantPowerplant.stage_telemetry`** — 6 sections × 7 rows of live physics for the
  POWERPLANT tab (keys 1–6) and the 3D stream intensities.
- **`n2n_physical_ceiling`** — the hard wall: σ(n,2n)/σ_nonelastic = 0.7241. 27.6% of
  neutrons go to (n,γ)/(n,p)/(n,α) and are unavailable for gold. Proof P23.
- **`maximize_gold`** — walks the three real levers (duty → ceiling, enrichment → 100%,
  leakage → floor) and reports what each buys and costs. Buildable optimum: **5.843 t/yr,
  1.528× baseline, 96.3% of the ceiling, for 1.095× fuel with electricity unchanged.**
- **`RadiantPowerplant`** — stateful plant: commissioning resolves H8–H13, `step(dt)` integrates
  inventories, `city_rating()` / `gold_rating()` scale to present day.
- **`build_powerplant_complex_solid`** — 3D solid mesh (3,478 verts, 3,057 faces, 6 labelled sections).

Engineering hurdles (all resolved at commissioning):
- **H8** unknown fuel → substitute highest‑reactivity runnable fuel
- **H9** unrunnable fuel (e.g. p‑p, 28 orders too slow) → substitute D‑T
- **H10** aneutronic fuel → D‑T neutron pilot (energy‑positive, heat returns to turbine)
- **H11** tritium breeding → gold duty as neutron multiplier (1 + x ≥ TBR + parasitic)
- **H11b** natural mercury (9.97% Hg‑198) → Be/Pb multiplier layer or enrichment to 90%
- **H12** activated bullion → shielded decay tank + assay hold (0.3 Bq/g vs 100 Bq/g limit)
- **H13** gold market saturation → fleet ceiling (~863 units before supply doubles)

### THE GOLD PROOF — where each link is implemented and verified

Full cited argument: **[About.md](About.md)**. This is the code-level index of that proof: for
every link in the chain, the function that computes it, the proof that verifies it, and the
self-test check that guards it.

| # | Link in the chain | Implemented in | Proof | Guarded by |
|---|---|---|---|---|
| 1 | D-T releases a 14.06 MeV neutron | `FUSION_FUELS["D-T"]` (Q = 17.589 MeV, AME2020) | — | `powerplant: typed 'D-T' resolves` |
| 2 | That neutron clears the (n,2n) gate | `FusionFuel.fast_neutrons_per_rx`, `HG198_N2N_THRESHOLD_MEV = 8.5286` | **P17** | `threshold below the D-T neutron energy` |
| 3 | Mercury barely moderates it (~50 chances) | `n2n_collisions_above_threshold`, `ELASTIC_XI_HG` | **P17** | `elastic decrement matches exact lethargy formula` |
| 4 | Conversion odds from real cross-sections | `n2n_reaction_probability` (σ 2.1 b / 2.9 b nonel. / 5.5 b tot) | — | `conversion is exactly linear in enrichment` |
| 5 | ¹⁹⁷Hg → ¹⁹⁷Au by electron capture | `RadiantPowerplant.step` (exact λ-integrator, t½ = 64.14 h) | **P17** | `decay tank reaches steady state, not runaway` |
| 6 | Nucleons/charge/lepton conserved | `heavy_element_cost`, P21 verifier | **P21** | `nucleon number conserved in (n,2n)` |
| 7 | Gold is lighter than its mercury → refinery | `AU197_MASS_KG` vs 198·u | **P21** | `one gold atom is LIGHTER than the mercury` |
| 8 | (n,2n) is a neutron multiplier, not a parasite | `neutron_economy` (1 + x + m ≥ TBR + parasitic) | **P18** | `gold duty makes the plant tritium self-sufficient` |
| 9 | No runaway — kinematic cap at one atom | `n2n_reaction_probability` bounded ≤ 1 | **P19** | `fast neutron gets enough collisions` |
| 10 | Gold costs thermal power (no free lunch) | `blanket_energy_per_fusion_MeV` | **P20** | `(n,2n) debits the heat budget` |
| 11 | Bullion assays clean (born outside the flux) | `gold_activation_state` (λN·N_A/198 = 9.055e15 Bq/g) | — | `faster Hg extraction means less activated gold` |
| 12 | Mass scale: kg/s → t/yr | `RadiantPowerplant._solve_operating_point` | — | `'D-T' commissions and PRODUCES GOLD` |
| 13 | Grid rating is measured-demand arithmetic | `city_rating`, `HOME_AVG_KW`, `PER_CAPITA_ALL_SECTOR_KW` | **P22** | `homes rating inverts to net electrical output` |
| 14 | Fleet ceiling is economic, not physical | `powerplant_market_report` | — | (H13 hurdle text is model-derived) |
| 15 | Maturity stated honestly | `TRL_AUDIT` (7 powerplant entries) | — | `TRL audit is honest that D-T net electric is unproven` |
| 16 | Threshold derived from Q, never re-entered | `HG198_N2N_THRESHOLD_MEV = HG198_N2N_Q_MEV · (mₙ+M)/M` | — | `threshold is DERIVED from Q by the exact recoil factor` |
| 17 | Heat ledger debits Q, not the threshold | `blanket_energy_per_fusion_MeV` | **P20** | `heat ledger debits the (n,2n) Q-value, not the threshold` |
| 18 | Reaction gate still uses the threshold | `n2n_collisions_above_threshold` | — | `the reaction GATE still uses the threshold` |
| 19 | Both Hg-197 branches end on stable gold | `HG197M_HALFLIFE_S` (documented) | — | `A=197 is conserved along the whole decay chain` |

**The chain, with the numbers it actually computes:**

```
2,705.6 MWth  →  8.7184e20 fusions/s  →  ×1.0 n/fusion  →  ×0.500 conversion
              →  4.3592e20 Au atoms/s  →  ×3.271262e-25 kg  →  1.42602e-4 kg/s
              →  3,825.1 kg/yr @ 85% availability  =  3.83 t/yr  =  308 bullion bars
```

**Why it is NUCLEAR and not chemical.** Chemistry is the physics of electrons and never changes
Z; element identity *is* Z. Mercury is Z = 80, gold Z = 79. Only the nuclear force closes that
gap — which is why alchemy failed for eighteen centuries and why Sherr, Bainbridge & Anderson
succeeded in 1941 (*Phys. Rev.* 60, 473) nine years after the neutron was discovered. Chemistry
does the *recovery* here (retort distillation, Hg recycling, γ-assay, casting) and that half is
TRL 9.

**Live per-section telemetry.** `RadiantPowerplant.stage_telemetry()` returns 6 sections × 7 rows
of live physics, rendered by the POWERPLANT tab (keys **1–6**) and guarded by
`stage telemetry covers all six plant sections` / `telemetry is LIVE`.


CLI: `--powerplant [FUEL]`, `--fuel FUEL`, `--fuels`, `--maximize-gold`, `--fleet`, `--mwe`, `--enrichment`,
`--gold-price`, `--years`.

For the cited, end-to-end proof that this makes gold at mass scale, see [About.md](About.md).
For the full project status and roadmap, see [narrative.md](narrative.md);
for the narrative vision this realizes, see [ProjectGoal.md](ProjectGoal.md);
for usage, see [README.md](README.md).
