// Condition-state rules, health index and NDT recommendation.
// Mirrors bridge/inspection.py; tests/test_step4.py checks that both give the same answers.

function band(v, edges) {
  for (let i = 0; i < 3; i++) if (v <= edges[i]) return i + 1;
  return 4;
}

function conditionState(defect, material, d) {
  const g = (k, dflt) => (d[k] === undefined || d[k] === null || d[k] === "") ? dflt : d[k];
  const num = (k, dflt) => parseFloat(g(k, dflt === undefined ? 0 : dflt)) || 0;
  let cs;
  switch (defect) {
    case "crack": {
      const w = num("width_mm");
      if (material === "PSC") {
        cs = band(w, [0.05, 0.10, 0.30]);
        const o = g("orientation", "");
        if (o.startsWith("flexural") || o.startsWith("shear")) cs = w >= 0.1 ? Math.max(cs, 3) : Math.max(cs, 2);
        if (o.startsWith("longitudinal") || o.startsWith("bursting")) cs = Math.max(cs, 3);
      } else {
        cs = band(w, [0.10, 0.30, 1.0]);
      }
      if (g("rust stain", false) === true) cs = Math.max(cs, 3);
      return Math.min(cs, 4);
    }
    case "spalling": {
      const s = g("steel exposed", "none");
      cs = Math.max(band(num("area_m2"), [0.0, 0.10, 0.50]), 2);
      if (s === "rebar") cs = Math.max(cs, 3);
      if (s.startsWith("duct")) cs = 4;
      return cs;
    }
    case "corrosion staining":
      return ({ "light": 2, "moderate": 3 })[g("severity", "light")] || 4;
    case "exposed steel":
      return (g("steel", "rebar").startsWith("duct") || g("section loss", false) === true) ? 4 : 3;
    case "leakage / efflorescence":
      return ({ "damp patch": 2, "active leakage": 3, "efflorescence": 3 })[g("state", "damp patch")] || 4;
    case "honeycombing":
      if (g("duct visible", false) === true) return 4;
      return num("depth_mm") > 0 ? band(num("depth_mm"), [0.0, 10.0, 25.0]) : 2;
    case "surface deterioration":
      return Math.max(2, band(num("depth_mm"), [0.0, 5.0, 15.0]));
    case "impact damage":
      return ({ "scrape": 2, "spalled": 3 })[g("severity", "scrape")] || 4;
    case "missing members":
      return num("count") <= 2 ? 3 : 4;
    case "bearing shear deformation":
      return band(num("shear_mm") / Math.max(num("elastomer_mm", 48), 1e-6), [0.5, 0.7, 1.0]);
    case "bearing bulging / splitting":
      return ({ "uniform bulging": 1, "uneven bulging": 2, "surface cracks": 3 })[g("state", "")] || 4;
    case "bearing displacement / walking":
      return band(num("offset_mm"), [5.0, 15.0, 30.0]);
    case "bearing seized / debris":
      return g("state", "").startsWith("debris") ? 2 : 4;
    case "pedestal cracking":
      return band(num("width_mm"), [0.1, 0.3, 1.0]);
    case "joint debris":
      return g("extent", "partial") === "partial" ? 2 : 3;
    case "joint seal torn / leaking":
      return 3;
    case "joint anchorage damage":
      return 4;
    case "drain blocked":
      return g("state", "partly") === "partly" ? 2 : 3;
    case "drain damaged / missing":
    case "discharge onto structure":
      return 3;
    case "wearing coat cracking":
      return num("area_m2") > 0 ? band(num("area_m2"), [0.0, 5.0, 20.0]) : 2;
    case "potholes / ravelling":
      return Math.max(2, band(num("depth_mm"), [0.0, 25.0, 50.0]));
    case "rutting":
      return band(num("depth_mm"), [5.0, 15.0, 25.0]);
  }
  return 2;
}

function healthIndex(componentCS, weights) {
  let num = 0, den = 0;
  for (const [c, cs] of Object.entries(componentCS)) {
    const w = weights[c];
    num += w * (4 - cs) / 3;
    den += w;
  }
  return den ? 100 * num / den : 100;
}

function actionFor(hi, worstCritical) {
  if (worstCritical >= 4) return "special inspection now; restrict loading until assessed";
  if (hi >= 85 && worstCritical <= 2) return "routine monitoring and next scheduled inspection";
  if (hi >= 70) return "enhanced monitoring; targeted NDT at CS3 locations";
  if (hi >= 50) return "detailed investigation (NDT + structural assessment)";
  return "urgent detailed assessment and repair design";
}

// Index into NDT_PROGRAMME for an observation, or -1.
function ndtFor(rec) {
  const c = rec.component, d = rec.defect, o = (rec.inputs || {}).orientation || "";
  const loc = rec.location || "";
  if (c === "Bearings") return 7;
  if (c === "PSC girders") {
    if (d === "crack") {
      if (o.startsWith("longitudinal") || (rec.inputs || {})["rust stain"] === true) return 0;
      if (o.startsWith("flexural")) return 1;
      if (o.startsWith("shear")) return 2;
      if (o.startsWith("bursting")) return 6;
    }
    if (d === "corrosion staining" || d === "exposed steel") return 0;
    if (d === "leakage / efflorescence") return loc.includes("End") ? 5 : 3;
    if (d === "honeycombing" || d === "surface deterioration") return 3;
    if (d === "spalling") return 4;
  }
  if (["Deck slab", "Cross girders", "Abutment A1", "Pier P1"].includes(c)) {
    if (["spalling", "exposed steel", "corrosion staining"].includes(d)) return 4;
    if (d === "honeycombing" || d === "surface deterioration") return 3;
    if (d === "leakage / efflorescence") return 5;
    if (d === "crack") return 4;
  }
  return -1;
}

if (typeof module !== "undefined") module.exports = { conditionState, healthIndex, actionFor, ndtFor, band };
