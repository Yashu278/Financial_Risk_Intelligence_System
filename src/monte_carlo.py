import numpy as np
import matplotlib.pyplot as plt


def run_monte_carlo(investment_amount, annual_return=0.08, volatility=0.15, years=5, n_simulations=10000):
    final_values = np.zeros(n_simulations)

    for i in range(n_simulations):
        portfolio = investment_amount
        for _ in range(years):
            annual_ret = np.random.normal(annual_return, volatility)
            portfolio = portfolio * (1 + annual_ret)
        final_values[i] = portfolio

    return {
        "best_case": np.percentile(final_values, 95),
        "worst_case": np.percentile(final_values, 5),
        "average_case": np.percentile(final_values, 50),
        "prob_of_loss": np.mean(final_values < investment_amount) * 100,
        "all_values": final_values,
        "investment_amount": investment_amount,
    }


def plot_monte_carlo(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(results["all_values"], bins=100, color="steelblue", alpha=0.7)
    ax.axvline(
        results["worst_case"],
        color="red",
        linestyle="--",
        label=f"Worst Case (5th %ile): ₹{results['worst_case']:,.0f}",
    )
    ax.axvline(
        results["average_case"],
        color="green",
        linestyle="--",
        label=f"Median: ₹{results['average_case']:,.0f}",
    )
    ax.axvline(
        results["best_case"],
        color="blue",
        linestyle="--",
        label=f"Best Case (95th %ile): ₹{results['best_case']:,.0f}",
    )
    ax.axvline(
        results["investment_amount"],
        color="black",
        linestyle="-",
        label=f"Initial: ₹{results['investment_amount']:,.0f}",
    )
    ax.set_xlabel("Portfolio Value After Investment Period (₹)")
    ax.set_ylabel("Number of Simulations")
    ax.set_title("Monte Carlo Simulation — Distribution of Outcomes (10,000 paths)")
    ax.legend()
    plt.tight_layout()
    return fig


def summarize_monte_carlo(results):
    inv = results["investment_amount"]
    avg = results["average_case"]
    best = results["best_case"]
    wrst = results["worst_case"]
    prob = results["prob_of_loss"]
    gain = ((avg - inv) / inv) * 100

    return (
        f"Based on 10,000 simulations: if you invest ₹{inv:,.0f}, your portfolio grows on average to "
        f"₹{avg:,.0f} ({gain:.1f}% return). Best case (top 5%): ₹{best:,.0f}. "
        f"Worst case (bottom 5%): ₹{wrst:,.0f}. Probability of ending with less than you invested: "
        f"{prob:.1f}%."
    )


if __name__ == "__main__":
    results = run_monte_carlo(100000, 0.08, 0.15, 5)
    print(summarize_monte_carlo(results))
    fig = plot_monte_carlo(results)
    fig.savefig("data/processed/monte_carlo_test.png")
    print("Chart saved to data/processed/monte_carlo_test.png")
