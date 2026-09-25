#!/usr/bin/env python3
"""Export the STAGE-1 fallback controller of the Q-CBF certificate.

The symmetric and asymmetric games run the same certificate with one exception:
the network that proposes the safe action. recon/02:342-348 — the symmetric S2C
members (`S_*`, `filtered: true`) were played behind

    deploy/.../safety_game/collision_v5prox_15k_s1ctrl_62d_game

while the asymmetric arm uses `collision_v5prox_15k_62d_game`. Loading both
bundles and comparing them tensor by tensor:

    critic  16 tensors, 0 differ
    dstb    18 tensors, 0 differ
    ctrl    18 tensors, 16 differ

So the value function and the adversary are shared and only the fallback
controller changes. This writes that one net as `filter_ctrl_s1` and records it
in the manifest as `filter.nets.ctrl_s1`; app/filter.js loads it in place of
`ctrl` when the game is symmetric.

    python3 tools/export_filter_s1ctrl.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_filter as ef  # noqa: E402  (same directory, same constants)

S1_BUNDLE = (
    ef.MJLAB
    / "deploy/robots/go2_go2/config/policy/safety_game/collision_v5prox_15k_s1ctrl_62d_game"
)
NAME = "filter_ctrl_s1"


def main() -> int:
    nets_file = S1_BUNDLE / "exported" / "filter_nets.pt"
    if not nets_file.exists():
        print(f"missing {nets_file}", file=sys.stderr)
        return 2

    blob = torch.load(nets_file, map_location="cpu", weights_only=False)
    ref = torch.load(
        ef.BUNDLE / "exported" / "filter_nets.pt", map_location="cpu", weights_only=False
    )

    # The whole point of this file is that ONLY ctrl differs. If that ever stops
    # being true the sym game needs its own critic too, and shipping one net
    # would be silently wrong.
    for key in ("critic", "dstb"):
        for t, v in ref[key].items():
            if not torch.equal(v.float(), blob[key][t].float()):
                print(f"{key}.{t} differs between the bundles — export both stacks",
                      file=sys.stderr)
                return 3
    n_diff = sum(
        1 for t, v in ref["ctrl"].items() if not torch.equal(v.float(), blob["ctrl"][t].float())
    )
    print(f"  critic/dstb identical; ctrl differs in {n_diff} tensors")

    js = ef.write_net(NAME, blob["ctrl"], ef.NETS["filter_ctrl"], ef.md5_of(nets_file), blob["meta"])
    # write_net stamps the asymmetric bundle's path; this net comes from the other one.
    js["source_bundle"] = str(S1_BUNDLE.relative_to(ef.MJLAB))
    js["display"] = "S2C certificate — pi_shield (stage-1, symmetric game)"
    (ef.POLICY_DIR / f"{NAME}.json").write_text(json.dumps(js, indent=2) + "\n", "utf-8")
    print(f"  {NAME}  {js['obs_dim']} -> {js['act_dim']}  {js['bin_bytes']:,} B")

    man_path = ef.POLICY_DIR / "manifest.json"
    man = json.loads(man_path.read_text())
    shipped = man["filter"]["nets"]["ctrl"]
    if [js["obs_dim"], js["act_dim"]] != [shipped["in"], shipped["out"]]:
        print("shape mismatch against the shipped ctrl", file=sys.stderr)
        return 4
    man["filter"]["nets"]["ctrl_s1"] = {
        "json": f"{NAME}.json",
        "bin": f"{NAME}.bin",
        "in": js["obs_dim"],
        "out": js["act_dim"],
        "head": "tanh",
        "note": "the symmetric game's fallback controller (recon/02:342-348)",
    }
    man_path.write_text(json.dumps(man, indent=1) + "\n", "utf-8")
    print("  manifest.filter.nets.ctrl_s1 written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
