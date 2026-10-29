# Code and literature cross-check

This file cross-checks the code values and clauses used in `bridge/` and `run_step*.py` against primary code texts. It was prepared with an AI assistant (Claude). Use it as a checklist against your own copies of the codes; it does not replace that check.

**Important: edition gap.** The report cites IRC:6-2017 and IRC:112-2020. The freely available primary texts are **IRC:6-2014** and **IRC:112-2011** (public.resource.org scans on archive.org), so those are what was checked. The values below are usually carried forward between editions, but clause and table numbers do move. Confirm each item against the edition the report cites.

## 1. Discrepancies found

| # | Item | Project | Code text | Where in project |
|---|---|---|---|---|
| 1 | Coefficient of thermal expansion | 11.7 × 10⁻⁶ /°C | **12.0 × 10⁻⁶ /°C** for RCC, PSC and steel structures (IRC:6-2014 cl. 215.4) | `bridge/config.py` `ALPHA_T` |
| 2 | Reverse temperature difference, depths h₂ = h₃ | 0.25h ≤ **0.20 m** | 0.25h ≤ **0.25 m** (IRC:6-2014 Fig. 10(a)) | `bridge/design.py` `temperature_profiles` |
| 3 | Low-relaxation strand, 1000 h loss at 0.6 fpk | 1.0 % | **1.25 %** (IRC:112-2011 Table 6.2; 0.5 / 0.7 / 0.8 fpk values match) | `bridge/config.py` `RELAX_1000H` |

**Effect on the design.** Step 2 was rerun in a scratch copy with all three corrections. No check changes outcome:

| Governing check | As built (margin, MPa) | Corrected (margin, MPa) |
|---|---|---|
| Compression ≤ 0.36 fck, quasi-permanent (G3 soffit, x = 2.25 m) | 0.130 | 0.126 |
| Uncracked (tension ≥ −fctm), frequent (G2 soffit, mid-span) | 0.889 | 0.851 |
| Decompression at soffit, quasi-permanent | 2.215 | 2.177 |
| Tension ≥ −fctm(t) at transfer | 1.516 | 1.515 |
| Compression ≤ 0.48 fck(t), construction stages | 4.749 | 4.761 |
| Compression ≤ 0.48 fck, rare | 6.232 | 6.213 |

The quasi-permanent compression check at the soffit near the support is the tightest one in the design, at about 0.8 % of the limit. Any later change to the prestress or the section should be rechecked there first.

`ALPHA_T` also feeds the bearing movement in step 3 and the synthetic bearing-slope feature in step 5. Correcting it changes those numbers by about 2.6 %.

## 2. Confirmed against the primary text

### IRC:6-2014

| Item | Project value | Code |
|---|---|---|
| Number of lanes | 3 for an 11.0 m carriageway | 9.6 m ≤ CW < 13.1 m → 3 lanes |
| Unit weight of concrete (RCC, PSC) | 25 kN/m³ | 2.5 t/m³ (cl. 203 list). 2.5 × 9.81 = 24.5 kN/m³, so 25 is the usual conservative rounding |
| Unit weight of asphalt | 22 kN/m³ | 2.2 t/m³ |
| Impact, Class A, concrete | 4.5 / (6 + L) | cl. 208.2 |
| Impact, 70R tracked | 10 % | cl. 208.3: 10 % up to a 40 m span |
| Impact, 70R wheeled, span ≥ 12 m | as Class A | cl. 208.3: per the Fig. 5 curve beyond 12 m |
| Multi-lane reduction | 10 %, applied only to 3 × Class A | cl. 205: three lanes 10 %. Restricting it to 3 × Class A is conservative |
| Footpath live load | (400 − 260 + 4800/L)(16.5 − W)/15 → 2.80 kPa | Basic 400 kg/m² is **cl. 206.1**; the span formula is **cl. 206.3(c)**. The code comment cites 206.1 for both |
| Class A train spacing | 20 m between last and first axles | Note: 18.5 m nose-to-tail. The figure dimensions it as **20.000 m between axles**, so the project matches |
| Clearance between passing vehicles | 1.2 m | Fig. 1 notes and cl. 204.1.2. The code comment cites "Table 2 note" |
| 70R tracked | 700 kN on 4.57 m tracks at 2.05 m c/c | Fig. 1: 0.84 m tracks, 2.90 m overall → **2.06 m** c/c (negligible) |
| Positive temperature difference | 17.8 / 4.0 / 0 °C over 0.15 + 0.25 m; 2.1 °C over 0.15 m at the soffit | Fig. 10(a) |
| Reverse temperature difference values | −10.6 / −0.7 / −0.8 / −6.6 °C, h₁ = h₄ = 0.2h ≤ 0.25 m | Fig. 10(a); see discrepancy 2 for h₂ and h₃ |
| SLS factors (live, thermal) | rare (1.0, 0.6), frequent (0.75, 0.5), QP (0, 0.5) | Table 3.3 (Annex B) |

### IRC:112-2011

| Item | Project value | Code |
|---|---|---|
| fcm, fctm | fck + 10; 0.259 fck^(2/3); 2.27 ln(1 + fcm/12.5) | Annex A2, Eq. A2-1 and A2-2. The project comment cites cl. 6.4.2 / Table 6.5 |
| Ep for strand | 195 GPa | cl. 6.3.5 text: "195 GPa for strands" |
| μ and k for corrugated HDPE | 0.17 and 0.0020 /m | Table 7.1 |
| Final relaxation | 3 × the 1000 h value | cl. 6.3.6 |
| Jacking stress | 0.75 fpk | cl. 7.9.2: ≤ 0.9 fp0.1k at jacking and ≤ 0.75 fpk (or 0.85 fp0.1k) after transfer. Both satisfied |
| Characteristic prestress | 1.10 and 0.90 Pm | Eq. 7.7 and 7.8: r_sup = 1.10 and r_inf = 0.90 for bonded post-tensioning |
| Effective flange width | 0.2 bᵢ + 0.1 l₀ ≤ 0.2 l₀, ≤ bᵢ | Eq. 7.1, numbered **cl. 7.6.2** in 2011. The code comment cites 7.6.1.2 |
| Stress limits | 0.48 fck rare; 0.36 fck QP | cl. 12.2.1 |
| Crack control, bonded tendons, severe exposure | uncracked under frequent; decompression at the soffit under QP | Table 12.1: w ≤ 0.2 mm under frequent **plus** decompression under QP, including temperature (note 3). The project's checks are stricter. Decompression is defined as no tension within 100 mm of the duct; checking at the soffit is conservative for sagging |
| Cover to post-tensioning ducts | 75 mm | "shall not be less than 75 mm" (cl. 15.3.1.x). This is an absolute minimum, not tied to exposure as the code comment says |
| Interface shear | cl. 10.3.4 | cl. 10.3.4 |

### IRC:83 (Part II)-2018

| Item | Project value | Code |
|---|---|---|
| Bearing shear strain bands | 0.5 / 0.7 / 1.0 | cl. 5.1.3.3: ε_q,d ≤ 1.00 at ULS. The note derives it as the SLS value 0.7 × 1.40 |

## 3. Not checked against a primary text

Only secondary sources or memory were available for these. Check them yourself.

- **IS 516 (Part 5) section numbers.** BIS-based copies show Sec 1 = UPV (2018), Sec 2 = half-cell potential, Sec 4 = rebound hammer (2020). Some websites give different numbering.
  - If Sec 2 is half-cell potential, cite **IS 516 (Part 5/Sec 2)** as the Indian reference next to ASTM C876.
- **UPV grading in IS 516 (Part 5/Sec 1):2018.** Exam-prep sources reproduce > 4.4 excellent, 3.75–4.4 good, 3.0–3.75 doubtful, < 3.0 poor. That is consistent with the project's "< 3.75 km/s doubtful", but confirm it in Table 2 of the standard.
- **IS 14268** (15.2 mm strand, 140 mm², 1860 MPa), **ASTM C876** potential thresholds, and **BS 8110-2 cl. 7.2** (Ec = 1.25 Ed − 19 GPa): standard values, not checked in the documents themselves.
- **Ecm** = 22 (fcm/12.5)^0.3 GPa: it reproduces the commonly tabulated 34 GPa (M45) and 32 GPa (M35), but the formula was not found in the OCR text.
- **IRC:SP:35**: the public.resource.org index lists a 2024 edition. Check which edition is current before citing it.

## 4. Published research that supports the modelling choices

Check bibliographic details in the papers before citing them.

| Modelling choice | Supporting paper |
|---|---|
| The twin shows that prestress loss leaves the natural frequencies essentially unchanged (step 5 condition table: f₁ = 3.442 Hz at every severity), so vibration-only detection of prestress loss is poor (3 %). | Hamed, E. & Frostig, Y. (2006). Natural frequencies of bonded and unbonded prestressed beams – prestress force effects. *Journal of Sound and Vibration* 295, 28–39. |
| Frequencies fall as temperature rises; the project's healthy f₁ sensitivity is −0.167 %/°C. | Xia, Y., Hao, H., Zanardo, G. & Deeks, A. (2006). Long term vibration monitoring of an RC slab: temperature and humidity effect. *Engineering Structures* 28(3), 441–452. Their per-°C figure was not checked; read it from the paper before comparing. |
| Environmental effects must be removed before damage detection. | Peeters, B. & De Roeck, G. (2001). One-year monitoring of the Z24-Bridge: environmental effects versus damage events. *Earthquake Engineering and Structural Dynamics* 30, 149–171. Sohn, H. (2007). Effects of environmental and operational variability on structural health monitoring. *Phil. Trans. R. Soc. A* 365, 539–560. |
| Mahalanobis-distance anomaly detector trained on healthy data only. | Worden, K., Manson, G. & Fieller, N.R.J. (2000). Damage detection using outlier analysis. *Journal of Sound and Vibration* 229(3), 647–667. |
| Output-only modal identification by FDD. | Brincker, R., Zhang, L. & Andersen, P. (2001), already in the reference list. |

## Sources

- IRC:6-2014 scan: https://archive.org/details/govlawircy201406
- IRC:112-2011 scan: https://archive.org/details/govlawircy2011112
- IRC:83 (Part II)-2018: https://law.resource.org/pub/in/bis/irc/irc.gov.in.083-2.2018.pdf
- IRC index on public.resource.org: https://law.resource.org/pub/in/bis/irc/
- IS 516 (Part 5) copies (secondary): https://www.scribd.com/document/444001811/IS-516-Part-5-Section-4-pdf, https://www.scribd.com/document/485918796/IS-516-Part-5-Section-2
- Xia et al. (2006) record: https://research.polyu.edu.hk/en/publications/long-term-vibration-monitoring-of-an-rc-slab-temperature-and-humi/
