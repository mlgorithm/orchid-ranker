import numpy as np
import matplotlib.pyplot as plt
from orchid_ranker.agents.agents import StudentAgent


# -----------------------
# Recommender policies
# -----------------------
def recommender_policy(student, policy="irt", knowledge_mode="scalar"):
    mu = float(student.knowledge if knowledge_mode == "scalar" else np.mean(student.knowledge))

    if policy == "irt":
        return np.clip(mu, 0, 1)

    elif policy == "adaptive_irt":
        if student.trust > 0.7 and student.engagement > 0.8:
            return np.clip(mu + 0.05, 0, 1)   # gentler growth mode
        elif student.engagement < 0.4:
            return np.clip(mu - 0.05, 0, 1)   # gentler recovery mode
        else:
            return mu

    elif policy == "zpd":
        if student.trust < 0.4 or student.engagement < 0.4:
            return mu   # rescue mode: don’t push harder
        else:
            return np.clip(mu + 0.1, 0, 1)   # moderate challenge

    else:
        raise ValueError(f"Unknown policy: {policy}")



# -----------------------
# Run one student session
# -----------------------
def run_student(user_type="strong", knowledge_mode="scalar",
                fatigue_growth=0.05, fatigue_recovery=0.02,
                trust_influence=True, rounds=300, seed=42,
                policy="irt"):

    student = StudentAgent(
        user_id=1,
        knowledge_mode=knowledge_mode,
        lr=0.2,
        decay=0.05,
        fatigue_growth=fatigue_growth,
        fatigue_recovery=fatigue_recovery,
        trust_influence=trust_influence,
        seed=seed,
    )

    # init knowledge
    if knowledge_mode == "scalar":
        student.knowledge = 0.7 if user_type == "strong" else 0.3
    else:
        dim = len(student.knowledge)
        base = 0.7 if user_type == "strong" else 0.3
        student.knowledge = np.clip(np.random.normal(loc=base, scale=0.1, size=dim), 0, 1)

    logs = {"knowledge": [], "fatigue": [], "trust": [], "engagement": [], "success_rate": []}

    for r in range(rounds):
        # Recommender picks difficulty
        d = recommender_policy(student, policy=policy, knowledge_mode=knowledge_mode)

        # Student acts
        decision = {"accepted": [r]}
        student.item_difficulty[r] = d
        feedback = student.act(decision, items_meta={r: {"difficulty": d}})
        student.update(feedback, items_meta={r: {"difficulty": d}})

        # Log
        k_val = student.knowledge if isinstance(student.knowledge, float) else np.mean(student.knowledge)
        logs["knowledge"].append(k_val)
        logs["fatigue"].append(student.fatigue)
        logs["trust"].append(student.trust)
        logs["engagement"].append(student.engagement)
        logs["success_rate"].append(np.mean(list(feedback.values())))

    return logs


# -----------------------
# Compare policies for strong vs weak
# -----------------------
def summarize_logs(label, logs, init_k):
    final_k = logs["knowledge"][-1]
    gain_k = final_k - init_k
    final_e = logs["engagement"][-1]
    final_t = logs["trust"][-1]
    avg_succ = np.mean(logs["success_rate"])
    print(f"{label:20s} | Knowledge={final_k:.3f} (Δ {gain_k:+.3f}) "
          f"| Engagement={final_e:.3f} | Trust={final_t:.3f} | Avg Success={avg_succ:.3f}")



def compare_setups():
    setups = [
        {"fatigue_growth": 0.05, "fatigue_recovery": 0.02, "trust_influence": True,  "label": "baseline (trust ON)"},
        {"fatigue_growth": 0.05, "fatigue_recovery": 0.02, "trust_influence": False, "label": "trust OFF"},
        {"fatigue_growth": 0.02, "fatigue_recovery": 0.02, "trust_influence": True,  "label": "slow fatigue growth"},
        {"fatigue_growth": 0.08, "fatigue_recovery": 0.01, "trust_influence": True,  "label": "fast fatigue growth"},
        {"fatigue_growth": 0.05, "fatigue_recovery": 0.05, "trust_influence": True,  "label": "fast recovery"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    print("\n[compare_setups: Final metrics]")
    for setup in setups:
        label = setup.pop("label")
        # Run
        logs = run_student(user_type="strong", policy="irt", **setup)
        init_k = logs["knowledge"][0]
        summarize_logs(label, logs, init_k)

        # Plots
        axes[0].plot(logs["knowledge"], label=label)
        axes[1].plot(logs["fatigue"], label=label)
        axes[2].plot(logs["trust"], label=label)
        axes[3].plot(logs["engagement"], label=label)

        setup["label"] = label


    titles = ["Knowledge", "Fatigue", "Trust", "Engagement"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()


def compare_policies():
    policies = ["irt", "adaptive_irt", "zpd"]
    student_types = ["strong", "weak"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.ravel()

    print("\n[compare_policies: Final metrics]")
    for pol in policies:
        for stype in student_types:
            label = f"{stype}-{pol}"
            logs = run_student(user_type=stype, policy=pol)
            init_k = logs["knowledge"][0]
            summarize_logs(label, logs, init_k)

            # Plots
            axes[0].plot(logs["knowledge"], label=label)
            axes[1].plot(logs["fatigue"], label=label)
            axes[2].plot(logs["trust"], label=label)
            axes[3].plot(logs["engagement"], label=label)
            axes[4].plot(np.cumsum(logs["success_rate"]) / (np.arange(len(logs["success_rate"]))+1),
                         label=label)
            axes[5].plot(logs["success_rate"], alpha=0.5, label=label)


    titles = [
        "Knowledge",
        "Fatigue",
        "Trust",
        "Engagement",
        "Success Rate (running avg)",
        "Success Rate (per round)",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()





# -----------------------
# Probability heatmap + line plots for multiple θ values
# -----------------------
def compare_success_prob(theta_list=[0.3, 0.6, 0.8], fatigue=0.2, alpha=5.0, beta=2.0):
    thetas = np.linspace(0, 1, 101)
    diffs = np.linspace(0, 1, 101)
    T, D = np.meshgrid(thetas, diffs)

    from scipy.special import expit as sigmoid

    def prob_success(theta, d, fatigue=0.2, alpha=5.0, beta=2.0, mode="IRT"):
        if mode == "IRT":
            logit = alpha * (theta - d) - beta * fatigue
        else:  # ZPD
            closeness = 1.0 - abs(theta - d)
            logit = alpha * (closeness - 0.5) - beta * fatigue
        return sigmoid(logit)

    # --- Heatmaps ---
    P_irt = prob_success(T, D, fatigue=fatigue, alpha=alpha, beta=beta, mode="IRT")
    P_zpd = prob_success(T, D, fatigue=fatigue, alpha=alpha, beta=beta, mode="ZPD")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(P_irt, origin="lower", extent=[0, 1, 0, 1],
                         cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("IRT Mode")
    axes[0].set_xlabel("Knowledge θ")
    axes[0].set_ylabel("Difficulty d")

    im2 = axes[1].imshow(P_zpd, origin="lower", extent=[0, 1, 0, 1],
                         cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("ZPD Mode")
    axes[1].set_xlabel("Knowledge θ")
    axes[1].set_ylabel("Difficulty d")

    fig.colorbar(im1, ax=axes, shrink=0.8, label="P(success)")
    plt.tight_layout()
    plt.show()

    # --- Line plots for multiple θ values ---
    plt.figure(figsize=(8, 6))
    for theta in theta_list:
        ps_irt = [prob_success(theta, d, fatigue=fatigue, alpha=alpha, beta=beta, mode="IRT") for d in diffs]
        ps_zpd = [prob_success(theta, d, fatigue=fatigue, alpha=alpha, beta=beta, mode="ZPD") for d in diffs]
        plt.plot(diffs, ps_irt, lw=2, label=f"IRT θ={theta}")
        plt.plot(diffs, ps_zpd, lw=2, linestyle="--", label=f"ZPD θ={theta}")

    plt.xlabel("Difficulty d")
    plt.ylabel("P(success)")
    plt.title("Success probability curves (IRT vs ZPD)")
    plt.legend()
    plt.show()

def compare_setups():
    setups = [
        {"fatigue_growth": 0.05, "fatigue_recovery": 0.02, "trust_influence": True,  "label": "baseline (trust ON)"},
        {"fatigue_growth": 0.05, "fatigue_recovery": 0.02, "trust_influence": False, "label": "trust OFF"},
        {"fatigue_growth": 0.02, "fatigue_recovery": 0.02, "trust_influence": True,  "label": "slow fatigue growth"},
        {"fatigue_growth": 0.08, "fatigue_recovery": 0.01, "trust_influence": True,  "label": "fast fatigue growth"},
        {"fatigue_growth": 0.05, "fatigue_recovery": 0.05, "trust_influence": True,  "label": "fast recovery"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for setup in setups:
        label = setup.pop("label")
        logs = run_student(user_type="strong", policy="irt", **setup)  # <-- policy explicitly IRT
        axes[0].plot(logs["knowledge"], label=label)
        axes[1].plot(logs["fatigue"], label=label)
        axes[2].plot(logs["trust"], label=label)
        axes[3].plot(logs["engagement"], label=label)
        setup["label"] = label  # restore

    titles = ["Knowledge", "Fatigue", "Trust", "Engagement"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()


# Compare student internal dynamics
compare_setups()

# Compare recommender strategies
compare_policies()

# Compare theoretical probability models
compare_success_prob(theta_list=[0.3, 0.6, 0.8])
