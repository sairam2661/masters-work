# Background

## Plasma Weight Loss Study

- Study that monitors patients during weight loss over time
- 43 patients who underwent eight weeks of weight loss (average of 12%) followed by a year of weight maintenance
- Collects plasma samples at multiple time points (7 in our case), use mass spectrometry to measure multiple peptides
- Basically, these plasma samples are used to track changes in the protein composition

## Context

### Plasma

- liquid component of blood
- contains multiple proteins that serve as biomarkers for various health conditions, disease states, and physiological changes

### Peptide

- short chains of amino acids
- fragment of a larger protein

### Peptidoform

- a specific form of peptide that contains modifications
- these modifications can be chemical changes, sequence variations, or other alterations
  - same peptide can exist in multiple peptidoforms due to these modifications

### Intensity Values

- represent the abundance of a specific peptidoform in a sample

## Project Dataset(s)

1. Peptidoforms intensities (40,921 rows) &rarr; Main dataset containing the measured intensities of all detected peptidoforms across all patient samples
2. Known modifications (289 rows) &rarr; standard known modifications that can occur on peptides (?)
3. Novel modifications (1,552 rows) &rarr; modifications detected in the study that don't match above (?)
4. ModDecode results (58 rows) &rarr; summaries of modifications detected for each patient (?)
5. Variants amino acid coordinates (200,000 rows) &rarr; maps peptide sequences to their positions in the original proteins
6. Study values &rarr; clinical measurements like weight, BMI, cholesterol levels, etc., for each participant
