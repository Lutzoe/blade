import argparse
import random
from pathlib import Path

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

DEFAULT_N = 50
DEFAULT_M = 2
DEFAULT_R_FEMALE = 0.35
DEFAULT_RHO = 0.5
DEFAULT_SEED = 42


def _validate_rho(rho):
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1].")
    return float(rho)


def _generate_bpa_python(
    n=DEFAULT_N,
    m=DEFAULT_M,
    r_female=DEFAULT_R_FEMALE,
    rho=DEFAULT_RHO,
    seed=DEFAULT_SEED,
):
    """Original reference implementation (slow for large n)."""
    if n <= 2 * m:
        raise ValueError("Need n > 2m so the initial clique is well-defined.")

    rho = _validate_rho(rho)
    rng = random.Random(seed)
    n0 = 2 * m

    genders = {}
    edges = set()
    degrees = [0] * n

    for i in range(n0):
        genders[i] = "male"

    # Directed transitive tournament on the initial clique: edge u -> v for u < v.
    for u in range(n0):
        for v in range(u + 1, n0):
            edges.add((u, v))
            degrees[u] += 1
            degrees[v] += 1

    for new_node in range(n0, n):
        genders[new_node] = "female" if rng.random() < r_female else "male"

        targets = set()
        while len(targets) < m:
            candidates = [old for old in range(new_node) if old not in targets]
            weights = []
            for old in candidates:
                accept_weight = 1.0 if genders[old] == genders[new_node] else rho
                weights.append(max(degrees[old], 1) * accept_weight)
            if not any(weight > 0.0 for weight in weights):
                weights = [max(degrees[old], 1) for old in candidates]

            target = rng.choices(candidates, weights=weights, k=1)[0]
            targets.add(target)

        for target in targets:
            edges.add((new_node, target))
            degrees[new_node] += 1
            degrees[target] += 1

    return sorted(edges), genders


def _generate_bpa_numpy(
    n=DEFAULT_N,
    m=DEFAULT_M,
    r_female=DEFAULT_R_FEMALE,
    rho=DEFAULT_RHO,
    seed=DEFAULT_SEED,
):
    """Vectorized weights; same sampling semantics as _generate_bpa_python."""
    if n <= 2 * m:
        raise ValueError("Need n > 2m so the initial clique is well-defined.")

    rho = _validate_rho(rho)
    rng = np.random.default_rng(seed)
    n0 = 2 * m
    # 0 = male, 1 = female (matches "female" branch in python path)
    gender = np.zeros(n, dtype=np.uint8)
    degrees = np.zeros(n, dtype=np.int64)
    edges = set()

    for u in range(n0):
        for v in range(u + 1, n0):
            edges.add((u, v))
            degrees[u] += 1
            degrees[v] += 1

    for new_node in range(n0, n):
        gender[new_node] = 1 if rng.random() < r_female else 0
        new_group = int(gender[new_node])
        targets = set()
        while len(targets) < m:
            old_idx = np.arange(new_node, dtype=np.int64)
            mask = np.ones_like(old_idx, dtype=bool)
            if targets:
                target_arr = np.fromiter(targets, dtype=np.int64)
                mask &= ~np.isin(old_idx, target_arr, assume_unique=False)
            candidates = old_idx[mask]
            weights = np.maximum(degrees[candidates], 1).astype(np.float64)
            accept = np.where(gender[candidates] == new_group, 1.0, rho)
            weights *= accept
            if not np.any(weights > 0.0):
                weights = np.maximum(degrees[candidates], 1).astype(np.float64)
            probabilities = weights / weights.sum()
            target = int(rng.choice(candidates, p=probabilities))
            targets.add(target)

        for target in targets:
            edges.add((new_node, target))
            degrees[new_node] += 1
            degrees[target] += 1

    genders = {i: ("female" if gender[i] else "male") for i in range(n)}
    return sorted(edges), genders


def generate_bpa(
    n=DEFAULT_N,
    m=DEFAULT_M,
    r_female=DEFAULT_R_FEMALE,
    rho=DEFAULT_RHO,
    seed=DEFAULT_SEED,
    *,
    prefer_numpy=True,
):
    """
    Biased Preferential Attachment (BPA) graph generator (directed).

    edges file:  u v       (space-separated, one directed edge u -> v per line, no header)
    gender file: node<TAB>gender

    Initial clique on nodes {0,..,2m-1}: directed edges u -> v for all u < v.
    Each new node t attaches with m directed edges t -> target for distinct targets.

    The command-line flag --rho follows the BPA convention in Stoica et al.
    It is the cross-label acceptance parameter: same-label candidates are
    accepted with weight 1, while cross-label candidates are accepted with
    weight rho. Equivalently, each target is sampled preferentially by degree
    with a rho penalty on cross-label edges:
      rho -> 0: cross-label edges are suppressed (strong homophily);
      rho -> 1: cross-label edges are not penalized (weak/no homophily bias).
    """
    if prefer_numpy and _HAS_NUMPY:
        return _generate_bpa_numpy(n, m, r_female, rho, seed)
    return _generate_bpa_python(n, m, r_female, rho, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--r-female", type=float, default=DEFAULT_R_FEMALE)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help="Stoica BPA cross-label acceptance parameter in [0, 1].")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prefix", type=str, default=None, help="Output prefix; default is bpa_<n>.")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Force the slow pure-Python generator (for debugging).",
    )
    args = parser.parse_args()

    edges, genders = generate_bpa(
        n=args.n,
        m=args.m,
        r_female=args.r_female,
        rho=args.rho,
        seed=args.seed,
        prefer_numpy=not args.python_only,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or f"bpa_{args.n}"
    edge_path = out_dir / f"{prefix}.txt"
    gender_path = out_dir / f"{prefix}_gender.txt"

    edge_path.write_text("\n".join(f"{u} {v}" for u, v in edges) + "\n")
    gender_path.write_text("\n".join(f"{i}\t{genders[i]}" for i in range(args.n)) + "\n")

    female_count = sum(1 for g in genders.values() if g == "female")
    male_count = args.n - female_count

    print(f"Generated: {edge_path}")
    print(f"Generated: {gender_path}")
    same_edges = sum(1 for u, v in edges if genders[u] == genders[v])
    same_fraction = same_edges / len(edges) if edges else 0.0
    cross_fraction = 1.0 - same_fraction if edges else 0.0

    print(
        f"nodes={args.n}, edges={len(edges)}, male={male_count}, female={female_count}, "
        f"m={args.m}, minority_fraction={args.r_female}, rho={args.rho}, "
        f"same_edge_fraction={same_fraction:.6g}, "
        f"cross_edge_fraction={cross_fraction:.6g}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
