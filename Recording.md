A) **Expectations**
  - _LLM Extraction:_ The model is expected to only **extract** the genetic biomarker without processing it.
  - _Post-Processing:_ A processing step will follow the LLM extraction and would correct the gene naming and modify variant if necessary to match nomenclature followed in CIViC such as: <br>
        a) Gene level: Let the gene be named as used in CIViC (using synonyms form NCBI api, string alignment if exact match is not found). (C-KIT mutation --> KITmutation, RAS mutation --> NRAS mutation, KRAS mutation, HRAS mutation) <br>
        b) Variant level: Replace alteration with mutation, rearrangement, fusion, amplification, etc and keep the ones found in CIViC. If categorical variant, keep it like that. Otherwise if 
  - _Final Extraction Output:_ Biomarkers extracted from trials but modified to match CIViC's nomenclature
  - _Matching_: Match patient query (now from text query, later images) to clinical trials based on the inclusion and exclusion biomarkers extracted from each trial
