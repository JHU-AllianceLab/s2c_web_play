#!/usr/bin/env python3
"""Export a symmetric-pool member under a throwaway name, for A/B in the game.

    python3 tools/export_candidate.py cand_et_a .cache/ET_v3s31g1900A.pt attacker A

Arguments: NAME CHECKPOINT CKPT_SEAT NATIVE_HALF. Writes
`assets/policies/NAME.{json,bin}` with the pi-rotation baked when the native
half is not B, and nothing else — no manifest row, because a candidate is not a
shipped policy. tests/head_on.mjs loads it by file name.

This exists because a member's numbers in the pool do not predict how it behaves
with a PERSON walking into it: the shipped ET member falls 0.6 % of episodes in
the pool and 75 % of them in the head-on battery. The only way to choose for the
game is to run the candidates in the game.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_policy as ep  # noqa: E402


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    name, ckpt, ckpt_seat, native_half = sys.argv[1:5]
    path = Path(ckpt)
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 2

    sd = ep.load_actor(path, ckpt_seat)
    obs_dim = ep.layer_shapes(sd)[0][0]
    # Seat B is where the AI sits, so an A-half member needs the rotation baked.
    rotate = native_half != "B"
    if rotate:
        sd = ep.pi_rotate_actor(sd, obs_dim)

    spec = dict(
        name=name,
        role="ai_opponent",
        game="sym",
        seat="B",
        method="candidate",
        display=name,
        obs_dim=obs_dim,
        act_dim=12,
        native_half=native_half,
        action_path="increment_integrator",
        upstream=str(path),
        notes="throwaway A/B candidate (tools/export_candidate.py)",
        source=path,
        provenance="candidate under A/B; not shipped",
        ckpt_seat=ckpt_seat,
        upstream_md5=None,
    )
    js = ep.export_policy(spec, sd, rotate, {}, "")
    print(f"  {name:16s} {js['obs_dim']}->{js['act_dim']}  rotation_baked {rotate}  {js['bin_bytes']:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
