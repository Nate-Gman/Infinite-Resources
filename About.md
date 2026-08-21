# ABOUT — Proving the Gold

**The claim under examination:** that [`Radiant.py`](Radiant.py) models a powerplant which
produces **stable, shippable, ordinary gold at industrial mass scale**, continuously, and that
every link in that chain is real measured science rather than assertion.

This document proves it, link by link, with the numbers, the citations, the conservation
ledgers, and the commands you can run to check each claim yourself.

It also states plainly what is **not** proven, because a proof that hides its weakest link is
not a proof.

---

## 0. First, a correction worth making — this is nuclear, not chemical

The single most important thing to be precise about:

> **Chemistry cannot make gold. Not at any temperature, pressure, catalyst or process.
> This is not an engineering limitation — it is a statement about what chemistry *is*.**

Chemistry is the physics of **electrons**. Every chemical process — burning, dissolving,
electroplating, catalysis, distillation — rearranges the electron clouds *around* nuclei while
leaving every nucleus untouched. The identity of an element is set by **Z**, the number of
protons in the nucleus. Since chemistry never changes Z, **chemistry can never change one
element into another.**

That is exactly why alchemy failed for eighteen centuries. The alchemists were not lazy or
stupid; they were working with the wrong force. Mercury is Z = 80 and gold is Z = 79, and no
amount of chemistry closes a gap of one proton.

**What closes it is the nuclear force**, and it took until 1932 (Chadwick's neutron) to have
the tool. Nine years later it was done — see §7.

So the honest description of this plant is:

| Stage | Discipline | What it does |
|---|---|---|
| Making the gold | **Nuclear physics** | Changes Z: 80 → 79. Chemistry cannot do this. |
| Recovering the gold | **Chemistry / metallurgy** | Separates Au from Hg, assays, refines, casts. |

Both halves are real, both are proven, and both are modelled. §1–§6 prove the nuclear half.
§8 proves the chemical half — which is the older and more certain of the two.

---

## 1. The chain, in full

```
                    ┌─ NUCLEAR (changes Z) ──────────────────────────┐
  D + T  ──────►  n(14.06 MeV)  ──►  ¹⁹⁸Hg + n  ──►  ¹⁹⁷Hg + 2n
                                                          │
                                     electron capture, t½ = 64.14 h
                                                          ▼
                                                      ¹⁹⁷Au  (STABLE)
                    └────────────────────────────────────────────────┘
                                                          │
                    ┌─ CHEMICAL (recovers it) ─────────────┼─────────┐
                      retort distillation → assay → cast → BULLION
                    └────────────────────────────────────────────────┘
```

### Step 1 — a 14.06 MeV neutron

`D + T → ⁴He (3.52 MeV) + n (14.06 MeV)`, Q = 17.589 MeV (AME2020).

The neutron carries 79.9 % of the energy because momentum conservation splits Q inversely to
mass. **This number is the whole reason the design works** — see step 2.

### Step 2 — ¹⁹⁸Hg(n,2n)¹⁹⁷Hg

| Quantity | Value | Source |
|---|---|---|
| Neutron separation energy Sₙ(¹⁹⁸Hg) | 8.4854 MeV | AME2020 |
| Lab-frame threshold | **8.5286 MeV** | Sₙ × (A+1)/A |
| σ(n,2n) at 14.1 MeV | **2.1 b** | ENDF/B-VIII.0, MT = 16 |
| σ nonelastic at 14.1 MeV | 2.9 b | ENDF/B-VIII.0 |
| ¹⁹⁸Hg natural abundance | 9.97 % | IUPAC/CIAAW |

A 14.06 MeV neutron clears the 8.5286 MeV threshold **by 5.55 MeV**. The D-T reaction and this
transmutation are, by coincidence of nuclear structure, almost perfectly matched.

> **A note on 8.5286 vs |Q| = 8.4854.** These are different quantities and the
> distinction matters. The **threshold** is the minimum neutron kinetic energy for the
> reaction to occur; it exceeds |Q| by the recoil factor (m_n + M)/M = 199/198, because
> momentum conservation forces the products to keep moving. The **Q-value** is the energy
> actually absorbed and converted to nuclear mass. The threshold governs the *gate* (can this
> neutron react?); the Q-value governs the *heat ledger* (what did the reaction consume?).
> The model derives the threshold from |Q| so the two cannot drift apart.

**Why the yield is tonnes and not grams.** Mercury barely slows a fast neutron. The mean
logarithmic energy decrement per elastic collision is

```
ξ = 1 + α ln α / (1 − α),    α = ((A−1)/(A+1))²,    A = 198
ξ = 0.010067
```

so the number of collisions before the neutron drops below threshold is

```
k = ln(14.06 / 8.5286) / ξ = 49.7  collisions
```

The neutron gets **~50 independent chances** to make gold. That is what converts a modest 2.1 b
cross-section into a near-certainty, and it is why this is an industrial process rather than a
laboratory curiosity.

### Step 3 — ¹⁹⁷Hg → ¹⁹⁷Au by electron capture

`¹⁹⁷Hg + e⁻ → ¹⁹⁷Au + νₑ`, t½ = **64.14 h**, 100 % branch.

A proton in the nucleus captures an orbital electron and becomes a neutron. **Z: 80 → 79.
A: unchanged.** The mercury becomes gold.

**¹⁹⁷Au is the only stable isotope of gold.** There is no isotopic signature, no decay, no
difference of any kind between this gold and gold dug out of the ground. It is not "artificial
gold" — the concept does not exist. It is gold.

---

## 2. Conservation — the books balance exactly

Every conserved quantity, checked at every step. This is what separates physics from wishing.

### Nucleon number (baryon number)

```
¹⁹⁸Hg + n  →  ¹⁹⁷Hg + 2n
   198  + 1  =    197  + 2
        199  =         199        ✓
```

### Charge

```
¹⁹⁸Hg (Z=80) + n (Z=0)  →  ¹⁹⁷Hg (Z=80) + 2n (Z=0)
              80        =              80              ✓

¹⁹⁷Hg (Z=80) + e⁻ (−1)  →  ¹⁹⁷Au (Z=79) + νₑ (0)
              79        =              79              ✓
```

### Lepton number

```
e⁻ (L = +1)  →  νₑ (L = +1)      ✓   (electron capture conserves L)
```

### Mass — and the proof this is a refinery, not a source

```
m(¹⁹⁸Hg) = 3.2879e−25 kg
m(¹⁹⁷Au) = 3.2713e−25 kg     ← LIGHTER
```

**One kilogram of gold requires 1.005076 kg of mercury fed in.** The plant does not create
matter. It rearranges nucleons that were already there. Contrast this with SECTION 26 of the
codebase, where E = mc² genuinely mints new rest mass at 9 × 10¹⁶ J/kg — *that* is matter
creation, and it is a different and far more expensive process.

> **Verified live as proof P21** — run `python3 Radiant.py --proofs`.

---

## 3. The neutron economy — why gold *pays for itself*

This is the most elegant result in the design, and it is not obvious.

A D-T plant must breed its own tritium — world civil tritium inventory is ~20 kg and this
plant burns **116.5 kg/yr** (194.3 kg/yr of D+T at a 0.5996 tritium mass fraction), so there is no
supply chain to buy from. Breeding runs on
`⁶Li(n,α)T` (+4.784 MeV), which **consumes a neutron per tritium bred**.

Naively, a gold blanket competes with breeding for neutrons. It does not — because **(n,2n) is
a neutron multiplier**:

```
        1 neutron in  →  2 neutrons out       (plus one gold atom)
```

Per source neutron, with x gold-making events and m from any dedicated multiplier:

```
produced  =  1 − x + 2x + m  =  1 + x + m
required  =  TBR + parasitic  =  1.05 + 0.15  =  1.20

self-sufficient  ⟺  1 + x + m ≥ 1.20  ⟺  x ≥ 0.20 − m
```

**The plant must make gold in order to breed its own fuel.** Gold production is not a parasite
on the neutron budget — it is the precondition for closing it.

At the modelled operating point: **x = 0.500**, produced = 1.500 vs 1.200 required, margin
**+0.300 n/source**, TBR = 1.05 — self-sufficient with room to spare.

> **Verified live as proof P18.**

### And it cannot run away

An (n,2n) event costs the separation energy. A 14.06 MeV neutron leaves
`14.06 − 8.4854 = 5.57 MeV`, **shared between two outgoing neutrons — ~2.79 MeV each**. That is
far below the 8.5286 MeV threshold, so **neither daughter can repeat the reaction.**

```
one source neutron  →  AT MOST one gold atom.    x ≤ 1, strictly.
```

There is no chain reaction, no criticality, no runaway. The cascade terminates after exactly
one generation, and the degraded daughters are precisely what feeds tritium breeding.

> **Verified live as proof P19.**

---

## 4. Mass scale — the full arithmetic

Every step shown so you can check it with a calculator. Reference unit: **1 GWe net.**

**(a) Size the reactor backwards from the grid**

```
net electric                                   1,000.0 MWe
gross = net / (1 − 0.12 recirculating)         1,136.4 MWe
thermal = gross / 0.42 (sCO₂ Brayton)          2,705.6 MWth
```

**(b) Energy released per fusion, after the blanket ledger**

```
Q_thermal = Q_fusion + TBR×4.784 − x×8.4854 − m×1.85 + Q_γ
          = 17.589   + 5.023     − 4.243   − 0     + 1.0
          = 19.369 MeV                    = 3.1013e−12 J
```

**(c) Fusion rate**

```
2.7056e9 W / 3.1013e−12 J = 8.7184e20 fusions/s
```

**(d) Gold atoms — one per (n,2n) event**

```
8.7184e20 × 1.0 n/fusion × 0.500 conversion = 4.3592e20 gold atoms/s
```

**(e) Mass**

```
4.3592e20 × 3.271262e−25 kg = 1.42602e−4 kg/s
```

**(f) Annual, at 85 % availability**

```
1.42602e−4 × 3.15576e7 s/yr × 0.85 = 3,825.1 kg/yr
```

### The result

| | |
|---|---|
| **Gold** | **10.47 kg/day · 3,825.1 kg/yr · 3.83 tonnes/yr** |
| In Good Delivery bars (12.44 kg) | **307 bars per year** |
| Troy ounces | 122,979 ozt/yr |
| Value at $4,532.80/ozt (2026-08-20) | **$557.4 M/yr** |
| Mercury consumed | 3,844 kg/yr (1.005 kg per kg of gold) |
| **Simultaneously** | **1,000 MWe → 690,492 homes, or 10.56 cities of 50,000** |

**308 gold bars a year, from one power station, as a by-product of selling electricity.**
That is mass scale by any definition — it is roughly 0.12 % of world mine production from a
single unit.

### Scaled to the world as it is today

```
world electricity          3.6 TW average  (~31,550 TWh/yr)
                        =  3,600 units of 1 GWe
gold                    =  3,600 × 3.825 t  =  13,771 t/yr
world mine production      3,300 t/yr
                        →  4.18 ×
```

**If every watt the world generates came from this plant, it would also mint 4.17× the entire
world gold mining industry, every year** — 6.24 % of all the gold ever mined (220,700 t),
annually.

> Run `python3 Radiant.py --fleet` to reproduce this table.

---

## 4b. Maximizing it — and the wall that stops you

The 3.83 t/yr above is a **conservative design point**, not a limit. The same reactor, on the
same site, with the same grid commitment, can be pushed considerably harder.

### The wall first, so the gains are read against something real

```
x_max  =  σ(n,2n) / σ_nonelastic  =  2.1 b / 2.9 b  =  0.7241
```

A fast neutron entering mercury ends in exactly **one** terminal channel. Of the 2.9 b
nonelastic budget, 2.1 b makes gold; the remaining **0.8 b — (n,γ), (n,p), (n,α) — consumes the
neutron and returns something that is not gold.**

**27.6 % of all source neutrons are physically unavailable, forever.** No enrichment, geometry,
reflector, moderator or budget changes that number — it is a property of the mercury nucleus.
Everything below is an *approach to this asymptote*, and nothing may cross it.

> **Verified live as proof P23.** The self-test sweeps every combination of enrichment
> (0 → 100 %) and leakage (0 → 50 %) and confirms none exceeds it.

### Why pushing the duty works at all

At a **fixed thermal rating** — the reactor you already have — output scales as

```
gold  ∝  x / Q_thermal(x),      Q_thermal(x) = 23.61 − 8.4854·x   MeV
```

which rises in x on *both* counts: more conversions per neutron, **and** more neutrons needed to
deliver the same heat, because each fusion now surrenders more of its energy to the (n,2n)
threshold. **Electricity to the grid does not change.** What pays for the extra gold is
**fuel** — and deuterium is seawater while tritium is bred in-house.

### The ladder

| Configuration | x | Gold t/yr | vs base | Fuel | MWe |
|---|---|---|---|---|---|
| baseline (default duty, 90 % enrichment) | 0.5000 | 3.825 | 1.000× | 1.000× | 1,000 |
| **1.** run the blanket at its cross-section ceiling | 0.5478 | 4.281 | 1.119× | 1.021× | 1,000 |
| **2.** + enrich the feedstock to 100 % Hg-198 | 0.6087 | 4.890 | 1.279× | 1.050× | 1,000 |
| **3.** + thicken the blanket, add a reflector | **0.6977** | **5.843** | **1.528×** | 1.095× | 1,000 |
| *asymptote: a blanket that never leaks — unbuildable* | *0.7241* | *6.143* | *1.607×* | *1.109×* | *1,000* |

### The buildable optimum

| | |
|---|---|
| **Gold** | **18.82 kg/day · 5.843 t/yr · 470 Good Delivery bars** |
| Fraction of the physical ceiling | **96.3 %** |
| Revenue | **$851.6 M/yr** (+$294.2 M over baseline) |
| Electricity | **1,000 MWe — unchanged** |
| Fuel | 212.6 kg/yr D+T (**1.095×**), of which 127.5 kg/yr tritium, bred on site |
| Neutron margin | **+0.498 n/source** — still tritium self-sufficient |
| Assay | 0.010 Bq/g after a 108-day hold — still shippable |

**1.095× the fuel for 1.528× the gold**, with the grid commitment untouched. That is the whole
trade, and it is a good one — but it is a real cost, and the thermal ledger (proof P20) still
debits every gram.

### The catch — maximizing output moves the economic ceiling *closer*

More gold per unit means **fewer units** before the fleet doubles world supply and prices itself
down (hurdle H13):

| Duty | Gold/unit | H13 fleet ceiling |
|---|---|---|
| baseline | 3.825 t/yr | ~863 units (0.86 TWe) |
| **maximized** | **5.843 t/yr** | **~565 units (0.56 TWe)** — 1.53× fewer |

**Physical maximization and fleet-scale economics point in opposite directions.** Maximize the
blanket if you are building a few units. If you are building thousands, the electricity is the
product, the gold is the by-product, and the conservative duty is the better business.

### At world scale

```
3,600 units × 5.843 t  =  21,036 t/yr  =  6.37× world mine production
                                          (the baseline duty gave 4.17×)
```

Run it yourself: `python3 Radiant.py --maximize-gold`, or press **A** in the POWERPLANT tab to
toggle the maximized blanket live.

---

## 5. The honest trade — gold costs power

No free lunch is claimed, and the price is quantified.

Every (n,2n) event permanently removes its 8.5286 MeV threshold energy from the heat budget:

```
1 kg Au-197 = 3.056924e24 atoms
            × 8.5286 MeV = 2.6021e25 MeV = 4.156e12 J thermal
            = 1.155 GWh(th) → 484.9 MWh(e) at 42 %
```

| | |
|---|---|
| Electricity forgone per kg of gold | **484.9 MWh** |
| Worth, at $45/MWh | $21,819 |
| The gold is worth | $145,733 |
| **Margin** | **6.7× — profitable, but NOT free** |

The plant trades megawatt-hours for kilograms at a favourable but finite exchange rate. A model
that reported gold with no energy penalty would be lying; this one shows the penalty in the same
ledger as the output.

> **Verified live as proof P20.**

---

## 6. Why the gold is not radioactive

Fresh gold in a neutron flux is a problem: `¹⁹⁷Au(n,γ)¹⁹⁸Au` has a **98.65 b** thermal
cross-section (Mughabghab 2018) — gold is destroyed *far* faster than it is made if you leave it
in the beam. ¹⁹⁸Au is a β⁻ emitter, t½ = 2.6947 d.

**The engineering answer is the elegant part: the gold is born *outside* the neutron field.**

¹⁹⁷Hg is pumped out of the blanket and decays to gold in a shielded tank at **zero neutron
flux**. Gold that never sees a neutron cannot be activated — the dominant term is removed by
geometry, not by shielding.

The residual contamination is computed, not assumed:

```
Au-198 fraction  =  (fraction of Hg-197 decaying in-flux)  ×  (capture probability there)
                 =  0.01075  ×  1.065e−4   =  1.245e−6      [at 1 h dwell]

specific activity of pure Au-198 = λN = (ln2/t½)(N_A/198) = 9.055e15 Bq/g
```

After a **108-day assay hold** (1.5× the computed clearance time, because sitting exactly on a
regulatory limit is a failing assay with a rounding error):

```
vault assay = 0.0094 Bq/g   vs   100 Bq/g clearance limit    →   SHIPPABLE
```

Shorter mercury dwell means cleaner gold: 0.25 h dwell → 64-day hold; 24 h dwell → 96-day hold.
The relationship is modelled, not asserted.

---

## 7. Precedent — this has been done

**It is not a proposal. It was done in 1941.**

> **Sherr, R., Bainbridge, K. T., & Anderson, H. H. (1941).**
> *"Transmutation of Mercury by Fast Neutrons."*
> **Physical Review 60, 473.**

Sherr and Bainbridge bombarded mercury with fast neutrons from a cyclotron and **chemically
recovered gold**. Bainbridge was one of the foremost mass spectroscopists of the era; the
isotopic identification is not in doubt. The reaction in this plant is the same reaction.

What the 1941 experiment lacked was not physics but **a cheap, enormous, continuous source of
fast neutrons**. A D-T fusion reactor is precisely that source — 8.7 × 10²⁰ of them per second,
as a by-product of generating electricity you were going to sell anyway.

**Modern context.** Fusion-driven mercury transmutation has been re-examined seriously in the
2020s, with published estimates in the same range this model derives independently (a few
tonnes per GW-year). This model reaches ~3.8 t/GWe/yr from first principles — the agreement is a
useful external check, not the source of the number.

---

## 8. The chemistry — what it actually does here

Chemistry cannot make the gold (§0), but it does everything else, and this is the **most mature
part of the entire plant**.

| Process | Chemistry | Maturity |
|---|---|---|
| **Separation** | Au and Hg form an amalgam; Hg boils at 357 °C, Au at 2,970 °C. Retort distillation drives the mercury off as vapour and leaves the gold. | Centuries old |
| **Mercury recycling** | The distilled Hg is condensed and returned to the blanket. Feedstock loss approaches zero. | Standard |
| **Assay** | γ-spectrometry against the clearance limit; ¹⁹⁸Au's 411.8 keV line is unmistakable. | Decades old |
| **Refining & casting** | Ordinary precious-metal refining to Good Delivery purity. | Ordinary industry |

This back end is **TRL 9 — nothing about it is novel.** The gold that leaves this building is
chemically, physically and isotopically indistinguishable from any other gold, because it *is*
any other gold.

---

## 9. What is NOT proven — the honest ledger

A proof that hides its weakest link is not a proof. From the on-board TRL audit
(`python3 Radiant.py --trl`, NASA 9-level scale):

| Subsystem | TRL | Status |
|---|---|---|
| Liquid-mercury loop under irradiation | **9** | SNS (Oak Ridge) has run one since 2006 |
| Gold refining + radiometric assay | **9** | Ordinary industry |
| Hg-198 isotope enrichment | **7** | Industrial for Hg-196/Hg-202; scale-up, not invention |
| sCO₂ Brayton conversion | **6** | 10 MWe pilot demonstrated |
| ¹⁹⁸Hg(n,2n) → ¹⁹⁷Au transmutation | **4** | Reaction measured & demonstrated 1941; never industrialised |
| **D-T fusion power island (net electric)** | **4** | **⚠ THE WEAKEST LINK** |
| Tritium breeding blanket | **3** | No breeding blanket has ever operated in a power reactor |

### The one sentence that matters

> **No device has ever produced net electrical power from fusion.** NIF achieved scientific
> ignition (target gain > 1) in December 2022; that is not the same as a plant that puts
> megawatts on a grid. **Every gold figure in this document is conditional on a D-T power plant
> existing.**

The transmutation chain is the *mature* half of this design. The reactor it is bolted to is the
half that is still an open problem. Nothing here is over-unity, nothing violates conservation —
but it is **not shovel-ready**, and any document claiming otherwise is selling something.

---

## 10. Verify it yourself

Every claim above is executable.

```bash
python3 Radiant.py --powerplant        # the full derivation, end to end, + the figure
python3 Radiant.py --proofs            # 22 theorems verified live (P17–P22 are the plant)
python3 Radiant.py --selftest          # 400 physics checks incl. the whole gold chain
python3 Radiant.py --trl               # the maturity audit above, unfiltered
python3 Radiant.py --maximize-gold     # push the blanket to the cross-section ceiling
python3 Radiant.py --fleet             # scaled to world electricity and the gold market
python3 Radiant.py --fuels             # every fuel the plant accepts
python3 Radiant.py                     # then press W — the live POWERPLANT tab
```

In the 3D viewer press **W** for the powerplant, **F** to type a fuel, **1–6** to inspect any
section's live telemetry.

### The specific proofs

| Proof | Claim |
|---|---|
| **P17** | Mercury → gold is a measured transmutation; nucleon books balance |
| **P18** | The gold reaction pays its own neutron bill (multiplier, not parasite) |
| **P19** | One neutron, at most one gold atom — kinematically capped, no runaway |
| **P20** | Gold costs power; the trade is priced, not free |
| **P21** | The plant creates no matter — refinery, not source |
| **P22** | The grid rating is arithmetic on measured demand, not marketing |
| **P23** | Gold conversion is capped at σ(n,2n)/σ_nonelastic = 0.724 — 27.6 % is unavailable |

---

## 11. Every constant, with its source

| Constant | Value | Source |
|---|---|---|
| Sₙ(¹⁹⁸Hg) = \|Q\| (energy absorbed) | 8.4854 MeV | AME2020 (Wang et al., Chin. Phys. C 45, 030003) — **the single measured input** |
| ¹⁹⁸Hg(n,2n) threshold | 8.5286 MeV | **Derived** = \|Q\|·(mₙ+M)/M — never entered separately |
| σ(n,2n) @ 14.1 MeV | 2.1 b | ENDF/B-VIII.0 MT=16 |
| σ nonelastic @ 14.1 MeV | 2.9 b | ENDF/B-VIII.0 |
| t½(¹⁹⁷Hg) | 64.14 h, EC 100 % | ENSDF A=197 (Kondev, Nucl. Data Sheets 109, 1527) |
| t½(¹⁹⁸Au) | 2.6947 d, β⁻ | ENSDF |
| σ(¹⁹⁷Au(n,γ)) thermal | 98.65 b | Mughabghab, *Atlas of Neutron Resonances* (2018) |
| ¹⁹⁸Hg abundance | 9.97 % | IUPAC/CIAAW (Meija et al. 2016) |
| Q(D-T) | 17.589 MeV | AME2020 |
| Q(⁶Li(n,α)T) | 4.784 MeV | AME2020 |
| ξ (Hg elastic) | 0.010067 | Derived; exact form to 0.01 % |
| Home consumption | 10,791 kWh/yr | US EIA |
| Per-capita all-sector | 1.40 kW | US EIA |
| Gold price | $4,532.80/ozt | Spot, 2026-08-20 |
| World mine production | 3,300 t/yr | USGS Mineral Commodity Summaries |
| Above-ground stock | 220,700 t | World Gold Council |
| World electricity | 3.6 TW | ~31,550 TWh/yr |

---

---

## 11b. Reaction-data integrity — how the numbers are kept honest

A table of constants is only as good as its discipline about where each number comes from.
Four errors were found and resolved in an audit of the reaction data; all four were the same
species of mistake, and the fix in every case was structural rather than a corrected literal.

### The principle: derive, never re-enter

**Two numbers describing the same reaction must never be entered independently.** If they are,
they will eventually disagree, and nothing in the code will notice.

The threshold and the Q-value of ¹⁹⁸Hg(n,2n) are exactly such a pair. They were stored as
separate literals — and they *had* drifted 0.017 MeV apart. Now there is one measured input and
the rest is arithmetic:

```
|Q|   = 8.4854 MeV                          ← the single measured value (AME2020)
E_th  = |Q| × (mₙ + M)/M  =  8.5286 MeV     ← DERIVED, cannot disagree
```

### The four errors, and what each one was

| # | Error | Effect | Resolution |
|---|---|---|---|
| 1 | Threshold charged to the **heat ledger** instead of Q | thermal budget over-debited **0.31 %** | ledger uses `HG198_N2N_Q_MEV`; gate keeps the threshold |
| 2 | Be-9(n,2n) multiplier debited its threshold (1.85) not its Q (1.5735) | that term over-charged **17.6 %** | `BE9_N2N_Q_MEV`, derived from AME2020 mass excesses |
| 3 | Threshold stored as a literal **inconsistent** with its own Q-value | 8.512 vs the 8.5286 the recoil physics demands | threshold now derived from Q |
| 4 | Two different "per day" figures, **identically labelled** | 12.32 vs 10.47 kg/day — looked self-contradictory | labelled "at power" vs "annual average" |

Errors 1–3 are the same confusion wearing different hats:

> **The threshold governs the GATE** — *can this neutron drive the reaction?*
> **The Q-value governs the LEDGER** — *how much energy did the reaction consume?*
>
> They differ by the recoil kinetic energy the products must carry away to conserve momentum.
> That energy is **kinetic**: it thermalizes in the blanket and is still collected. Charging it
> to the heat budget bills the plant for energy it never lost.

Error 4 was not a physics mistake at all — both figures were correct. It was a **labelling**
failure, which in a document meant to prove something is just as damaging: a reader who spots
two different numbers for the same quantity is entitled to stop believing the rest.

### And an incomplete decay scheme, resolved

The (n,2n) reaction populates both the Hg-197 ground state (64.14 h) **and** the isomer Hg-197m
(23.8 h, 91.4 % IT → ground state + 8.6 % EC → gold directly). Only the ground state was
modelled.

**Both branches terminate on ¹⁹⁷Au**, so the gold *yield* is independent of the isomeric ratio —
only the transient timing differs. The model therefore keeps the ground-state half-life, which
is the **slower** and hence conservative choice: the isomer fills the vault marginally faster,
never slower, and never changes how much gold exists. **No isomeric ratio was invented, because
none is needed to bound the answer.**

### What stops recurrence

Ten self-test checks now guard this area specifically, including:

- the threshold must equal `|Q| × (mₙ+M)/M` **exactly** (to 1e-12)
- the heat ledger must debit `x · |Q|`, never `x · E_th`
- Be-9's Q must match the AME2020 mass-excess arithmetic
- the *gate* must still use the threshold — sub-threshold neutrons get zero chances
- both Hg-197 branches must terminate on A = 197 gold

These are assertions about **how the numbers relate**, not about their values, so they survive
future data updates and fail loudly if the derive-never-re-enter discipline is broken again.

## Summary

**Gold, at mass scale, proven:**

1. **Chemistry cannot do it** — chemistry is electrons, element identity is Z. §0
2. **Nuclear physics can, and did, in 1941** — Sherr, Bainbridge & Anderson. §7
3. **The reaction is measured**, not modelled — ENDF/B-VIII.0, threshold 8.5286 MeV. §1
4. **The books balance exactly** — A, Z, lepton number, mass. It is a refinery. §2
5. **Gold pays its own neutron bill** — (n,2n) is 1 in, 2 out; breeding depends on it. §3
6. **The scale is 3.83 t/yr per GWe** — 307 bars, derived in six lines of arithmetic. §4
6b. **Maximized: 5.84 t/yr** (470 bars, 1.53×) — capped at 0.724 by cross-sections. §4b
7. **It costs 485 MWh/kg** — profitable at 6.7×, but priced honestly. §5
8. **The product assays clean** — 0.0094 Bq/g vs a 100 Bq/g limit. §6
9. **The chemistry that recovers it is TRL 9** — ordinary industry. §8
10. **The fusion reactor it needs does not yet exist** — TRL 4, stated plainly. §9
11. **Every reaction number is derived from one measured input**, not re-entered; four
    data-integrity errors found and structurally fixed. §11b

*See also: [README.md](README.md) for the system overview · [Overview.md](Overview.md) for the
section-by-section technical index.*
