A) **Expectations**
  - _<ins>LLM Extraction</ins>:_ The model is expected to only **extract** the genetic biomarker without processing it.
  - _<ins>Post-Processing</ins>:_ A processing step will follow the LLM extraction and would correct the gene naming and modify variant if necessary to match nomenclature followed in CIViC such as: <br>
    &nbsp; a) Gene level: Let the gene be named as used in CIViC (using synonyms form NCBI api, string alignment if exact match is not found, example: C-KIT mutation --> KITmutation, RAS mutation --> NRAS mutation, KRAS mutation, HRAS mutation) <br>
    &nbsp; b) Variant level: Replace alteration with mutation, rearrangement, fusion, amplification, etc and keep the ones found in CIViC. If categorical variant, keep it like that. Otherwise if 
  - _<ins>Final Extraction Output</ins>:_ Biomarkers extracted from trials but modified to match CIViC's nomenclature
  - _<ins>Matching</ins>_: Match patient query (now from text query, later images) to clinical trials based on the inclusion and exclusion biomarkers extracted from each trial

B) **Dataset Generation**

