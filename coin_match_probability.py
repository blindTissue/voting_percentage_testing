import argparse
import math


def matching_head_count_probability(flips: int) -> float:
    """Return P(two fair-coin trials have the same number of heads)."""
    if flips < 0:
        raise ValueError("flips must be non-negative")

    # Sum_k [C(n,k)/2^n]^2 = C(2n,n)/4^n.
    log_probability = math.lgamma(2 * flips + 1) - 2 * math.lgamma(flips + 1) - flips * math.log(4)
    return math.exp(log_probability)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probability that two fair-coin trials have matching head/tail counts."
    )
    parser.add_argument(
        "--flips",
        type=int,
        default=5000,
        help="Number of flips in each trial. Default: 5000.",
    )
    args = parser.parse_args()

    probability = matching_head_count_probability(args.flips)
    print(f"flips per trial: {args.flips:,}")
    print(f"P(matching head count): {probability:.12g}")
    print(f"Probability scale: 1 in {1 / probability:,.2f}")


if __name__ == "__main__":
    main()
