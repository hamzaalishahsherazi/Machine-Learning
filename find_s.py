"""
Project: Concept Learning with the Find-S Algorithm (from scratch)
Task: "On which days does Aldo enjoy his favourite water sport?"
(EnjoySport dataset from Tom Mitchell, Machine Learning, Chapter 2)

Idea: Find-S looks for the MOST SPECIFIC rule that covers every "Yes" day.
  - Start with the strictest possible rule:  <0, 0, 0, 0, 0, 0>  (matches nothing)
  - For each POSITIVE example, loosen the rule just enough to include it:
        empty slot ('0')      -> copy the example's value
        value differs         -> replace with '?'  (means "anything is fine")
        value matches         -> keep it
  - NEGATIVE examples are simply ignored.

Analogy: a detective with a suspect sketch. Every time a confirmed sighting
comes in, he only erases the details that don't match; everything that
always matched stays in the sketch.

Symbols:  '0' = no value acceptable,  '?' = any value acceptable
"""

import csv
import os


# ============================================================
#                  Find-S from scratch
# ============================================================

def find_s(examples, attributes, verbose=True):
    n = len(attributes)
    h = ["0"] * n                                   # most specific hypothesis
    if verbose:
        print(f"  Start: h0 = {fmt(h)}\n")

    for i, (x, label) in enumerate(examples, start=1):
        if label.lower() != "yes":
            if verbose:
                print(f"  Example {i}: {fmt(x)} -> {label:<3}  NEGATIVE, ignored")
                print(f"             h{i} = {fmt(h)}\n")
            continue

        changes = []
        for j in range(n):
            if h[j] == "0":
                h[j] = x[j]
            elif h[j] != x[j] and h[j] != "?":
                changes.append(f"{attributes[j]}: '{h[j]}' -> '?'")
                h[j] = "?"
        if verbose:
            print(f"  Example {i}: {fmt(x)} -> {label:<3}  POSITIVE, generalise")
            for c in changes:
                print(f"             {c}")
            print(f"             h{i} = {fmt(h)}\n")
    return h


def matches(h, x):
    """Does instance x satisfy hypothesis h?"""
    return all(hv == "?" or hv == xv for hv, xv in zip(h, x))


def fmt(values):
    return "<" + ", ".join(f"{v}" for v in values) + ">"


# ============================================================
#                  Data handling
# ============================================================

ATTRIBUTES = ["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast"]

ENJOY_SPORT = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same",   "Yes"],
    ["Sunny", "Warm", "High",   "Strong", "Warm", "Same",   "Yes"],
    ["Rainy", "Cold", "High",   "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High",   "Strong", "Cool", "Change", "Yes"],
]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    examples = [(r[:-1], r[-1]) for r in data]
    return header[:-1], examples


def banner(text):
    print("\n" + "=" * 66 + f"\n{text}\n" + "=" * 66)


# ============================================================
#                  MAIN
# ============================================================

if __name__ == "__main__":
    # ---------- 1. Load training data from CSV ----------
    path = "enjoysport.csv"
    if not os.path.exists(path):
        write_csv(path, ATTRIBUTES + ["EnjoySport"], ENJOY_SPORT)
    attributes, examples = read_csv(path)

    banner("FIND-S ALGORITHM - EnjoySport")
    print(f"Training data read from '{path}':\n")
    print("  " + " | ".join(f"{a:<8}" for a in attributes + ["Enjoy"]))
    for x, y in examples:
        print("  " + " | ".join(f"{v:<8}" for v in x + [y]))

    # ---------- 2. Step-by-step trace ----------
    banner("STEP-BY-STEP TRACE")
    h = find_s(examples, attributes)

    banner("FINAL HYPOTHESIS")
    print(f"  h = {fmt(h)}")
    rule = [f"{a} = {v}" for a, v in zip(attributes, h) if v != "?"]
    print(f"\n  In plain words: Aldo enjoys the sport when " + " AND ".join(rule) + ".")
    print("  Humidity, Water and Forecast don't matter.")

    # ---------- 3. Check it on training data ----------
    banner("CHECK ON TRAINING DATA")
    correct = 0
    for x, y in examples:
        pred = "Yes" if matches(h, x) else "No"
        correct += pred == y
        print(f"  {fmt(x):<52} actual {y:<3}  predicted {pred}")
    print(f"\n  Training accuracy: {correct}/{len(examples)}")

    # ---------- 4. Classify new days ----------
    banner("CLASSIFY NEW DAYS")
    new_days = [
        ["Sunny", "Warm", "Normal", "Strong", "Cool", "Change"],
        ["Rainy", "Cold", "Normal", "Light",  "Warm", "Same"],
        ["Sunny", "Warm", "Normal", "Light",  "Warm", "Same"],
        ["Cloudy", "Warm", "High",  "Strong", "Warm", "Same"],
    ]
    for d in new_days:
        print(f"  {fmt(d):<52} -> {'Yes' if matches(h, d) else 'No'}")

    # ---------- 5. Order doesn't matter ----------
    banner("DOES THE ORDER OF EXAMPLES MATTER?")
    h_rev = find_s(list(reversed(examples)), attributes, verbose=False)
    print(f"  Original order : {fmt(h)}")
    print(f"  Reversed order : {fmt(h_rev)}")
    print(f"  Same result?   : {h == h_rev}")

    # ---------- 6. Limitations ----------
    banner("LIMITATIONS OF FIND-S")

    print("1) It ignores negative examples, so it can't notice contradictions.")
    noisy = examples + [(["Sunny", "Warm", "High", "Strong", "Warm", "Same"], "No")]
    h_noisy = find_s(noisy, attributes, verbose=False)
    print("   Added a 'No' day identical to example 2 (a 'Yes' day):")
    print(f"   Result is still {fmt(h_noisy)}  -> no warning about the conflict.\n")

    print("2) One mislabelled positive example can wreck the hypothesis.")
    wrong = examples + [(["Rainy", "Cold", "High", "Light", "Cool", "Change"], "Yes")]
    h_wrong = find_s(wrong, attributes, verbose=False)
    print("   Added a cold rainy day wrongly labelled 'Yes':")
    print(f"   Result becomes {fmt(h_wrong)}  -> now says 'Yes' to every day!\n")

    print("3) It returns only ONE hypothesis (the most specific). Other rules,")
    print("   like <Sunny, ?, ?, ?, ?, ?>, also fit the data, but Find-S can't")
    print("   tell us that. The Candidate Elimination algorithm fixes this by")
    print("   keeping the whole version space (S and G boundaries).")
