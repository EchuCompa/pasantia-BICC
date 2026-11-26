from itertools import permutations

from asvFormula.genericASV import asvFromTopologicalOrders


def expected_prediction(evidence, noise):
    if "Y" in evidence:
        return 1.0
    if "X" in evidence:
        return 1.0 - noise
    return 0.5


def shapley_values(noise):
    orders = list(permutations(("X", "Y")))
    return {
        feature: asvFromTopologicalOrders(
            feature, orders, lambda evidence: expected_prediction(evidence, noise)
        ).item()
        for feature in ("X", "Y")
    }


def asymmetric_shapley_values(noise):
    order = [("X", "Y")]
    return {
        feature: asvFromTopologicalOrders(
            feature, order, lambda evidence: expected_prediction(evidence, noise)
        ).item()
        for feature in ("X", "Y")
    }


def main():
    print("noise,SHAP_X,SHAP_Y,ASV_X,ASV_Y")
    for noise in (0.0, 0.1, 0.25):
        shapley = shapley_values(noise)
        asymmetric = asymmetric_shapley_values(noise)
        assert abs(sum(shapley.values()) - 0.5) < 1e-12
        assert abs(sum(asymmetric.values()) - 0.5) < 1e-12
        print(
            f"{noise:.2f},{shapley['X']:.3f},{shapley['Y']:.3f},"
            f"{asymmetric['X']:.3f},{asymmetric['Y']:.3f}"
        )


if __name__ == "__main__":
    main()
