# Existing Approaches & Tools

## Older / SideQ Approaches

- Reflexion / AgentCoder: LLM-based self-learning for test generation.
- Libro: Uses LLMs to generate tests from bug reports.

## Existing Fuzzing Tools

- Grammarinator – Uses formal grammar for structured test inputs.
- TestChain
- LLAMAFUZZ
- Gramatron
- AFL++
- LibFuzzer

## Vanilla vs. TestChain Study

### Vanilla LLM Test Case Generation

- Evaluated on HumanEval-no-exp, LeetCode-no-exp.
- Primary metric: LineCov (Line Coverage) / CwB (Code-with-Bugs pass rate).
- GPT-4 results:
  - 80-90% LineCov for both.
  - 55% accuracy in LC-no-exp.
  - 80% accuracy in HE-no-exp (best performing model from the study).

### TestChain Approach

- Inspired by ReAct – decouples input/output and uses a Python interpreter for interactive testing.
- Test I/O split into a structured conversation:
  1. Designer Agent: Generates test input cases based on a given prompt.
  2. Calculator Agent: Determines expected test outputs and writes final test cases (interacts with a Python interpreter to refine outputs).
- GPT-4 results:
  - 80-90% LineCov for both.
  - 70% accuracy in LC-no-exp.
  - 90% accuracy in HE-no-exp.

---

## Tool-Specific Overviews

### LLAMAFUZZ

- Enhances greybox fuzzing for structured data.
- Utilizes LLMs’ understanding of data formats for effective mutation strategies.
- Fine-tuned with paired mutation seeds to learn structured formats.
- Results: Outperformed top competitors by discovering 41 additional bugs on average and identified 47 unique bugs across trials.

### Gramatron

- Restructures input grammars to allow for unbiased sampling from the input space.
- Transforms grammars to improve coverage and bug-finding efficiency by avoiding biases in traditional grammar-based fuzzing.

### AFL++

- Community-driven extension of AFL (American Fuzzy Lop) with new enhancements.
- Key features:
  - Improved performance.
  - New mutation strategies.
  - Better support for various binary formats.
- Results: Demonstrates significant improvements in code coverage and bug detection compared to AFL.

### LibFuzzer

- In-process, coverage-guided, evolutionary fuzzing engine targeting library APIs.
- Key features:
  - Directly linked with the library under test.
  - Utilizes LLVM’s SanitizerCoverage for mutation guidance.
- Results: Efficiently discovers API bugs by leveraging targeted coverage techniques.
