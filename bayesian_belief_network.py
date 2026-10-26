"""
Project: Home Alarm System with a Bayesian Belief Network (from scratch)

Story (Judea Pearl's classic example):
  A house has a burglar alarm. It can go off because of a Burglary,
  but also because of a small Earthquake. Two neighbours, John and Mary,
  promise to call you if they hear the alarm. John sometimes confuses
  the phone ringing with the alarm; Mary sometimes misses it because
  of loud music.

  Network (arrows = "directly causes"):

        Burglary     Earthquake
               \\      /
                Alarm
               /     \\
        JohnCalls   MaryCalls

Idea: instead of storing one giant table of every combination
(2^5 = 32 rows), each node only stores a small table:
"my probability given my parents" (CPT). The chain rule glues them together:
    P(B, E, A, J, M) = P(B) * P(E) * P(A|B,E) * P(J|A) * P(M|A)

What this project covers:
  1. Building the network and its Conditional Probability Tables (CPTs)
  2. Joint probability using the chain rule
  3. EXACT inference by enumeration  (e.g. P(Burglary | John & Mary call))
  4. APPROXIMATE inference by rejection sampling, compared with exact
  5. LEARNING the CPTs back from generated data (with Laplace smoothing)
"""

import itertools
import numpy as np


# ============================================================
#                  Bayesian Network from scratch
# ============================================================

class BayesianNetwork:
    def __init__(self):
        self.parents = {}   # node -> list of parent names
        self.cpt = {}       # node -> {tuple(parent values): P(node=True)}
        self.order = []     # topological order (parents before children)

    def add_node(self, name, parents, cpt):
        self.parents[name] = parents
        self.cpt[name] = cpt
        self.order.append(name)

    def prob(self, node, value, assignment):
        """P(node = value | its parents' values in 'assignment')."""
        key = tuple(assignment[p] for p in self.parents[node])
        p_true = self.cpt[node][key]
        return p_true if value else 1 - p_true

    # ---------- chain rule ----------
    def joint(self, assignment):
        result = 1.0
        for node in self.order:
            result *= self.prob(node, assignment[node], assignment)
        return result

    # ---------- exact inference ----------
    def enumerate_ask(self, query, evidence):
        """P(query | evidence) by summing the joint over all hidden variables."""
        hidden = [n for n in self.order if n != query and n not in evidence]
        dist = {}
        for q_val in (True, False):
            total = 0.0
            for values in itertools.product((True, False), repeat=len(hidden)):
                assignment = dict(evidence, **{query: q_val}, **dict(zip(hidden, values)))
                total += self.joint(assignment)
            dist[q_val] = total
        norm = dist[True] + dist[False]                  # normalise (alpha)
        return {k: v / norm for k, v in dist.items()}

    # ---------- sampling ----------
    def sample(self, rng):
        """Forward sampling: roll a weighted coin for each node, parents first."""
        s = {}
        for node in self.order:
            s[node] = rng.random() < self.prob(node, True, s)
        return s

    def rejection_sampling(self, query, evidence, n, rng):
        kept, count_true = 0, 0
        for _ in range(n):
            s = self.sample(rng)
            if all(s[k] == v for k, v in evidence.items()):   # keep only matching samples
                kept += 1
                count_true += s[query]
        return (count_true / kept if kept else float("nan")), kept

    # ---------- learning ----------
    @classmethod
    def learn_from_data(cls, structure, data, alpha=1):
        """Estimate every CPT by counting, with Laplace smoothing (alpha)."""
        bn = cls()
        for node, parents in structure:
            cpt = {}
            for key in itertools.product((True, False), repeat=len(parents)):
                rows = [r for r in data if all(r[p] == v for p, v in zip(parents, key))]
                n_true = sum(r[node] for r in rows)
                cpt[key] = (n_true + alpha) / (len(rows) + 2 * alpha)
            bn.add_node(node, parents, cpt)
        return bn


def fmt(evidence):
    return ", ".join(f"{k}={'T' if v else 'F'}" for k, v in evidence.items()) or "nothing"


# ============================================================
#                  MAIN
# ============================================================

if __name__ == "__main__":
    T, F = True, False

    # ---------- 1. Build the network ----------
    bn = BayesianNetwork()
    bn.add_node("Burglary",   [], {(): 0.001})
    bn.add_node("Earthquake", [], {(): 0.002})
    bn.add_node("Alarm", ["Burglary", "Earthquake"],
                {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})
    bn.add_node("JohnCalls", ["Alarm"], {(T,): 0.90, (F,): 0.05})
    bn.add_node("MaryCalls", ["Alarm"], {(T,): 0.70, (F,): 0.01})

    print("=" * 62)
    print("BAYESIAN BELIEF NETWORK - Home Alarm System")
    print("=" * 62)
    print("\n1. Network structure and CPTs  [P(node = True | parents)]")
    for node in bn.order:
        parents = bn.parents[node]
        print(f"\n  {node}" + (f"  (parents: {', '.join(parents)})" if parents else "  (no parents)"))
        for key, p in bn.cpt[node].items():
            cond = ", ".join(f"{pa}={'T' if v else 'F'}" for pa, v in zip(parents, key))
            print(f"     P({node}=T{' | ' + cond if cond else ''}) = {p}")
    print(f"\n  Numbers stored: {sum(len(c) for c in bn.cpt.values())} "
          f"(a full joint table would need {2 ** len(bn.order) - 1})")

    # ---------- 2. Chain rule ----------
    print("\n2. Joint probability with the chain rule")
    a = {"Burglary": F, "Earthquake": F, "Alarm": T, "JohnCalls": T, "MaryCalls": T}
    print("  P(J=T, M=T, A=T, B=F, E=F)")
    print("   = P(J|A) * P(M|A) * P(A|~B,~E) * P(~B) * P(~E)")
    print("   = 0.90 * 0.70 * 0.001 * 0.999 * 0.998")
    print(f"   = {bn.joint(a):.6f}")

    # ---------- 3. Exact inference ----------
    print("\n3. Exact inference by enumeration")
    queries = [
        ("Burglary", {}),
        ("Burglary", {"JohnCalls": T}),
        ("Burglary", {"JohnCalls": T, "MaryCalls": T}),
        ("Earthquake", {"JohnCalls": T, "MaryCalls": T}),
        ("Alarm", {"Burglary": T}),
        ("Burglary", {"Alarm": T, "Earthquake": T}),     # "explaining away"
        ("JohnCalls", {"Burglary": T}),
    ]
    exact = {}
    for q, ev in queries:
        p = bn.enumerate_ask(q, ev)[True]
        exact[(q, tuple(ev.items()))] = p
        print(f"  P({q}=T | {fmt(ev)}) = {p:.4f}")

    print("\n  Reading the results:")
    print("   - Both neighbours calling raises burglary chance from 0.1% to ~28%.")
    print("   - If we already know an earthquake happened, the alarm is 'explained'")
    print("     and the burglary probability drops sharply (explaining away).")

    # ---------- 4. Approximate inference ----------
    print("\n4. Approximate inference by rejection sampling (vs exact)")
    rng = np.random.default_rng(42)
    print(f"  {'Query':<44}{'Exact':>8}{'Sampled':>9}{'Kept':>8}")
    for q, ev in [("Alarm", {"Burglary": T}),
                  ("JohnCalls", {"Burglary": T}),
                  ("Burglary", {"JohnCalls": T, "MaryCalls": T})]:
        est, kept = bn.rejection_sampling(q, ev, 200_000, rng)
        label = f"P({q}=T | {fmt(ev)})"
        print(f"  {label:<44}{exact[(q, tuple(ev.items()))]:>8.4f}{est:>9.4f}{kept:>8}")
    print("  Note: rare evidence (both calling) keeps very few samples,")
    print("  so the estimate is noisy. That is the weakness of rejection sampling.")

    # ---------- 5. Learning CPTs from data ----------
    print("\n5. Learning the CPTs back from data")
    structure = [(n, bn.parents[n]) for n in bn.order]
    for n_samples in [1_000, 100_000]:
        data = [bn.sample(rng) for _ in range(n_samples)]
        learned = BayesianNetwork.learn_from_data(structure, data)
        print(f"\n  Using {n_samples:,} generated records:")
        print(f"  {'Parameter':<34}{'True':>8}{'Learned':>10}")
        for node in ["Burglary", "Alarm", "JohnCalls", "MaryCalls"]:
            parents = bn.parents[node]
            for key in list(bn.cpt[node])[:2]:
                cond = ",".join(f"{p[0]}={'T' if v else 'F'}" for p, v in zip(parents, key))
                name = f"P({node}=T{'|' + cond if cond else ''})"
                print(f"  {name:<34}{bn.cpt[node][key]:>8.3f}{learned.cpt[node][key]:>10.3f}")
    print("\n  More data -> learned CPTs get closer to the true ones. Rare parent")
    print("  combinations (like Burglary=T) need lots of data to estimate well.")
