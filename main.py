# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def gillespie(k_on, k_off, k_tx, gamma, T_max, dt, seed=None):
    rng = np.random.default_rng(seed)
    t = 0.0
    gene = 0  # OFF state
    mRNA = 0
    times = []
    mrnas = []
    next_record = 0.0
    while t < T_max:
        # Propensities
        a = []
        a.append(k_on if gene == 0 else 0.0)          # activation
        a.append(k_off if gene == 1 else 0.0)         # deactivation
        a.append(k_tx if gene == 1 else 0.0)          # transcription
        a.append(gamma * mRNA)                        # degradation
        a_sum = sum(a)
        if a_sum == 0:
            t = T_max
            break
        # Time to next reaction
        r1 = rng.random()
        tau = -np.log(r1) / a_sum
        t += tau
        # Record at regular intervals
        while next_record <= t and next_record <= T_max:
            times.append(next_record)
            mrnas.append(mRNA)
            next_record += dt
        # Choose reaction
        r2 = rng.random() * a_sum
        cumulative = 0.0
        for i, ai in enumerate(a):
            cumulative += ai
            if r2 < cumulative:
                reaction = i
                break
        # Update state
        if reaction == 0:
            gene = 1                     # activation
        elif reaction == 1:
            gene = 0                     # deactivation
        elif reaction == 2:
            mRNA += 1                    # transcription
        elif reaction == 3:
            if mRNA > 0:
                mRNA -= 1                # degradation
    # Ensure final recordings up to T_max
    while next_record <= T_max:
        times.append(next_record)
        mrnas.append(mRNA)
        next_record += dt
    return np.array(times), np.array(mrnas)

def run_exp1():
    k_on = 0.1
    k_off = 1.0
    k_tx = 5.0
    gamma = 0.2
    T_max = 200
    dt = 0.5
    times, mrnas = gillespie(k_on, k_off, k_tx, gamma, T_max, dt, seed=42)
    plt.figure(figsize=(8, 4))
    plt.step(times, mrnas, where='post')
    plt.xlabel('Time')
    plt.ylabel('mRNA copy number')
    plt.title('Stochastic transcription bursts')
    plt.tight_layout()
    plt.savefig('mrna_time_series.png')
    plt.close()

def run_exp2():
    k_off = 1.0
    gamma = 0.2
    T_max = 500
    dt = 1.0
    burn_in = 100
    k_on_vals = np.linspace(0.01, 1.0, 20)
    k_tx_vals = np.linspace(0.5, 5.0, 20)
    mean_matrix = np.zeros((len(k_on_vals), len(k_tx_vals)))
    for i, k_on in enumerate(k_on_vals):
        for j, k_tx in enumerate(k_tx_vals):
            times, mrnas = gillespie(k_on, k_off, k_tx, gamma, T_max, dt, seed=i * len(k_tx_vals) + j)
            mask = times >= burn_in
            mean_matrix[i, j] = mrnas[mask].mean()
    # Heatmap of mean mRNA
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mean_matrix.T, origin='lower', aspect='auto',
                    extent=[k_on_vals[0], k_on_vals[-1], k_tx_vals[0], k_tx_vals[-1]],
                    cmap='viridis')
    plt.colorbar(im, label='Mean mRNA')
    plt.xlabel('k_on (burst frequency)')
    plt.ylabel('k_tx (burst size)')
    plt.title('Mean mRNA across burst parameters')
    plt.tight_layout()
    plt.savefig('mean_mrna_heatmap.png')
    plt.close()
    # Histograms for low and high burst regimes
    low_i, low_j = 0, 0
    high_i, high_j = -1, -1
    T_hist = 2000
    dt_hist = 1.0
    burn_in_hist = 200
    # Low regime
    times, mrnas = gillespie(k_on_vals[low_i], k_off, k_tx_vals[low_j], gamma,
                             T_hist, dt_hist, seed=123)
    mask = times >= burn_in_hist
    low_counts = mrnas[mask]
    plt.figure()
    plt.hist(low_counts, bins=range(int(low_counts.max()) + 2), density=True, edgecolor='black')
    plt.xlabel('mRNA copy number')
    plt.ylabel('Probability')
    plt.title('Low burst regime')
    plt.tight_layout()
    plt.savefig('mrna_histogram_low.png')
    plt.close()
    # High regime
    times, mrnas = gillespie(k_on_vals[high_i], k_off, k_tx_vals[high_j], gamma,
                             T_hist, dt_hist, seed=456)
    mask = times >= burn_in_hist
    high_counts = mrnas[mask]
    plt.figure()
    plt.hist(high_counts, bins=range(int(high_counts.max()) + 2), density=True, edgecolor='black')
    plt.xlabel('mRNA copy number')
    plt.ylabel('Probability')
    plt.title('High burst regime')
    plt.tight_layout()
    plt.savefig('mrna_histogram_high.png')
    plt.close()
    return mean_matrix

def main():
    run_exp1()
    mean_matrix = run_exp2()
    overall_mean = mean_matrix.mean()
    print('Answer:', overall_mean)

if __name__ == '__main__':
    main()

