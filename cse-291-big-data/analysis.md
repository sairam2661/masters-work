# Analysis

## Dataset Description

- We consider the peptidoforms intensity data from the plasma proteomics dataset, which contains 40,921 resolved peptidoforms detected in at least one sample
- This has 707 columns

### Basic

- rowid, ccms_row_id, id &rarr; identifiers
- **Total**: 3 cols

### Peptide Information

- Peptidoform, Peptidoform ID &rarr; peptide with modifications
- Unmod Peptidoform, Unmod_Peptidoform &rarr; peptide w/out modifications
- Peptidoforms- Unmodified sequence &rarr; number of different modified variants observed for this peptide sequence
- Pep Prefix &rarr; preceding amino acid in the source protein
- **Total**: 6 cols

### Protein Information

- Proteins &rarr; protein accession numbers containing this peptide. For example, `sp|P01009|A1AT_HUMAN refers` to Alpha-1-antitrypsin protein
- **Total**: 1 col

### Physical/Chemical Properties

- Mass &rarr; molecular mass of the peptide in daltons
- Charge &rarr; charge state of the peptide
- **Total**: 2 cols

### Modification Info

- Num Mods &rarr; number of modifications on the peptide
- All Mods &rarr; list of all modifications present on the peptide
- Annotation &rarr; detailed description of modifications (including position)
- Annotation without position &rarr; same as above, w/out position information
- Known &rarr; if the modification is a standard known modification
- Num mod frags &rarr; number of fragment ions supporting the modification
- **Total**: 6 cols

### Quantification

- Total &rarr; sum of all intensity measurements for this peptidoform across all samples
- Total- Unmodified sequence &rarr; similar to total but for unmodified peptidoform
- **Total**: 2 cols

### Quality Metrics

- Is Decoy &rarr; flag for control/decoy sequences (estimate false discovery rates)
- Orig cluster FDR &rarr; false discovery rate for the identification
- PValue &rarr; statistical confidence in the identification
- % Explained &rarr; %age of the mass spectrum peaks explained by this identification
- Lorikeet input &rarr; data for lorikeet tool
- **Total**: 5 cols

### Processing Pipeline Metadata

- Rep cluster task &rarr; task that processed this cluster
- Rep cluster user &rarr; username
- Rep cluster index &rarr; index
- Num tasks &rarr; number of processing tasks detecting this peptidoform
- Rep spectrum filename &rarr; file containing the rep spectrum
- Rep spectrum scan &rarr; ID of the rep spectrum
- **Total**: 6 cols

### Outlier Information

- Outlier groups &rarr; sample groups where this peptidoform shows unusual behavior (?)
- Outlier group ratio &rarr; ratio for above
- Outlier groups- unmod &rarr; similarly for unmodified peptide
- Outlier group ratio- unmod &rarr; ratio for above
- **Total**: 4 cols

### Intensity Data

- \_dyn\_#Patient_XX.Timepoint_Y &rarr; intensity for patient XX at timepoint Y
- \_dyn\_#Patient_XX.Timepoint_Y_unmod &rarr; intensity for patient XX at timepoint Y for unmodified peptidoform
- The intensity data can be categorized by,
- Regular patient data,
  - Patient 01 to 58 w/ 7 timepoints (w/ 6 missing IDs) &rarr; 364 cols
  - But study says 43 obese individuals?
  - Similarly for unmod; 364 cols
  - **Total**: 728 cols
- Healthy control data,
  - F1, F2, F3 w/ 3 timepoints &rarr; 9 cols
  - M1, M2, M3 w/ 3 timepoints &rarr; 9 cols
  - Similarly for unmod; 18 cols
  - **Total**: 36 cols
- **Total**: 728 + 36 = 764 cols
- But we only have 672 cols of intensity data (707 - 35 = 672), so _missing timepoints_ for some patients?
