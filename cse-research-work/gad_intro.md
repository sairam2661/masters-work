# Grammar-Based Fuzzing and Grammar-Aligned Decoding

## Grammarinator

Grammar-based fuzzing tool

- Input? Formal grammar (e.g., JSON, Python, HTTP requests, etc.).
- Output? Structured test inputs (valid/invalid code snippets, data files, network packets) that conform to (or intentionally violate) the grammar rules.
- Goal? Test software systems (compilers, APIs, parsers) by throwing syntactically diverse inputs at them to find bugs or vulnerabilities.

Try some examples in the Python playground.

### Limitations / Areas for Improvement

- Rule-based (does not value semantic meaning in tests (?)).
- Does it intelligently generate test cases? (What percentage of edge cases are covered?)

---

## Grammar-Aligned Decoding (GAD)

Generate structured outputs from LLMs

- Ensures outputs conform to grammar without compromising the LLM’s natural distribution.

### Why Not Just Grammar-Constrained Decoding (GCD)?

- GCD ensures grammatical outputs but introduces bias by not incorporating the LM’s future grammatical probabilities (Expected Future Grammaticality, or EFG).
- This results in grammatical but distributionally skewed outputs that diverge from the LM’s learned patterns.

### How Does GAD Work?

- Uses an adaptive decoding algorithm called Adaptive Sampling with Approximate Expected Futures (ASAp).
- Starts with GCD, then converges to the LLM’s natural distribution over time.

---

## Combining Grammar-Based Fuzzing & GAD

### Intelligent Fuzzing?

- LLMs aim to produce human-like outputs; integrating them into fuzzing could uncover more diverse test cases.
- How would we integrate GAD with rule-based fuzzers?
  - Generate syntactically valid but semantically adversarial cases.
  - Quantify what makes an edge case or measure its semantic nature.
- Existing metrics:
  - Code/Line Coverage while running a test case (but this is often insufficient).
  - TestChain’s metric: Code-with-bugs pass rate.
  - Syntactic guarantees on test cases + semantic meaning.

### Hybrid Approach?

- Seed fuzzing with GAD + apply mutations for enhanced test coverage.
- other ways?
