"""The app's JS rules must give the same condition states as bridge/inspection.py."""
import itertools
import json
import subprocess
from pathlib import Path

from bridge import inspection as I

RULES = Path(__file__).resolve().parents[1] / "inspection_app" / "rules.js"


def cases():
    out = []
    comps = I.components()
    nums = [0, 0.04, 0.08, 0.2, 0.5, 3.0, 12, 30, 60]
    for c, v in comps.items():
        for d in v["defects"]:
            spec = I.DEFECTS[d]["inputs"]
            keys = list(spec)
            choices = []
            for k in keys:
                t = spec[k]
                choices.append(nums if t == "number" else [True, False] if t == "bool" else t)
            for combo in itertools.islice(itertools.product(*choices), 400):
                out.append(dict(defect=d, material=v["material"], inputs=dict(zip(keys, combo))))
    return out


def test_js_rules_match_python():
    cs = cases()
    js = ("const r = require(%s); const cs = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
          "console.log(JSON.stringify(cs.map(c => r.conditionState(c.defect, c.material, c.inputs))));") % json.dumps(str(RULES))
    res = subprocess.run(["node", "-e", js], input=json.dumps(cs), capture_output=True, text=True, check=True)
    js_cs = json.loads(res.stdout)
    bad = [(c, j, I.condition_state(c["defect"], c["material"], c["inputs"])) for c, j in zip(cs, js_cs)
           if j != I.condition_state(c["defect"], c["material"], c["inputs"])]
    assert not bad, bad[:5]
    assert len(cs) > 500


def test_health_index_extremes():
    comps = I.components()
    assert I.health_index({c: 1 for c in comps}) == 100
    assert I.health_index({c: 4 for c in comps}) == 0
    assert I.action(95, 4).startswith("special")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
