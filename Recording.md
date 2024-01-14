A) **Expectations**
  - _<ins>LLM Extraction</ins>:_ The model is expected to only **extract** the genetic biomarker without processing it.
  - _<ins>Post-Processing</ins>:_ A processing step will follow the LLM extraction and would correct the gene naming and modify variant if necessary to match nomenclature followed in CIViC such as: <br>
    &nbsp; a) Gene level: Let the gene be named as used in CIViC (using synonyms form NCBI api, string alignment if exact match is not found, example: C-KIT mutation --> KITmutation, RAS mutation --> NRAS mutation, KRAS mutation, HRAS mutation) <br>
    &nbsp; b) Variant level: Replace alteration with mutation, rearrangement, fusion, amplification, etc and keep the ones found in CIViC. If categorical variant, keep it like that. Otherwise if 
  - _<ins>Final Extraction Output</ins>:_ Biomarkers extracted from trials but modified to match CIViC's nomenclature
  - _<ins>Matching</ins>_: Match patient query (now from text query, later images) to clinical trials based on the inclusion and exclusion biomarkers extracted from each trial

B) **Dataset Generation**

  - _LLM Extraction:_ The model is expected to only **extract** the genetic biomarker without processing it.
  - _Post-Processing:_ A processing step will follow the LLM extraction and would correct the gene naming and modify variant if necessary to match nomenclature followed in CIViC such as: <br>
        a) Gene level: Let the gene be named as used in CIViC (using synonyms form NCBI api, string alignment if exact match is not found). (C-KIT mutation --> KITmutation, RAS mutation --> NRAS mutation, KRAS mutation, HRAS mutation) <br>
        b) Variant level: Replace alteration with mutation, rearrangement, fusion, amplification, etc and keep the ones found in CIViC. If categorical variant, keep it like that. Otherwise if 
  - _Final Extraction Output:_ Biomarkers extracted from trials but modified to match CIViC's nomenclature
  - _Matching_: Match patient query (now from text query, later images) to clinical trials based on the inclusion and exclusion biomarkers extracted from each trial


B) **Dataset Generation**
- Create files to annotate the trials on the side (and the a script to add the annotation to the prompt) --> submitted to Github RAW
- Create files with trials ID that were manually selected --> Submitted to Github (a script that takes the file and generates the prompts from chromadb) RAW output should go to either processed or interem 

- Annotation:
  The output format:
  <pre>
    # Whatever is in the same inner clause ([]) reflect biomarkers that should exist together to render patient eligible (logic AND).
    # Therefore, patient must satisfy all biomarkers inside one of the clauses to be eligible.
    # Each clause is therefore separated by OR and inside clause biomarkers are separated by AND
  {
    "inclusion_biomarker": [["b1", "b2"], ["b3"]],
    "exclusion_biomarker": []
  }</pre>


  - Should we consider cohort?
  - First, Annotate all the randomly generated trials
  - Select the ones that I am confident from the annotation
  - Add the manually annotated trials
  - How and when to split for test/train? Annotate all randomly selected trials, filter out trials that we are not confident from the annotation so that we do not give bad data to model, add the manually selected trials to increase variability then split in either ways: <br>
  a) defined stratified splitting (80/20) <br>
  b) ski-learn 80/20 then stratify? <br>
  Maybe we will need manual changing of trial to increase number of train set in fast manner instead of generating synthetic data using GPT
  



C) **Evaluation**
2 step evaluation could be done:
At extraction level (could also try comparing to the paper https://arxiv.org/pdf/2308.02180.pdf on our test set and focus on biomarkers, however, keep in mind that they consider biomarker in general not only genetic so think about that)
At post-processing level --> This is basically to make sure the post-processing is working as expected especially for gene naming! 
