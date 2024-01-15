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
- Generate a bunch of randomly selected biomarkers+chromadb trials ID
- Annotate the ones I feel confident about
- Select only these trials (provide a list of these IDs in JSON file named final_id_selection or something)
- Include the manually selected trials to increase the number of trials with biomarkers (better than synthetic data generation) and provide these (as JSON file too)
- Merge all and split


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


  How they defined and instructed the model to extract the biomarkers:
  - In system message: I only extract the most important tumor biomarkers
  - In instructions:
  - This is a representation of the logical disjunctive normal form where each conjunctive (AND) clause is represented as a JSON and the outer list is a disjunction (logical OR) of those clauses or JSONs.
  - biomarker inclusion: a list of biomarker inclusion criteria that if all elemnts of the list are satisfied, contributes to the expression evaluating to True.
  - biomarker exclusion: a list of biomarker exclusion criteria that if any elements of the list are satisfied, contributes to the expression evaluating to False.
  - Skip biomarker or histology criteria that are inside if-then conditions unless the if condition is a cohort type (non-cohor if-then example: If the patient has received prior treatment, then they must have a specific biomarker profile.)
  - Do not include criteria about prior therapies or prior treatment. They are not considered biomarkers criteria and should not be included in the inclusion or exclusion criteria. Do not include any expression prior therapies, treatments, or therapies.
  - If the inclusion criteria is for either of multiple biomarkers (and/or), list each of those biomarkers in a separate clause or JSON item because either of them being satisfied should contribute to the entire expression being True.
    I want to have the least restrictive accurate matching criteria output. Only list multiple biomarkers in biomarker inclusion list for one clause JSON item if the trial absolutely require all of those biomarkers.
  - Do not include mentioned if the presence or absense of the biomarker does not affect eligibility.
  - And do not include biomarkers in the output that if the biomarker is present, then additional criteria is needed.
  - I only extract if the criteria itself determines eligibility.
  - When there is a list of biomarker criteria in the same sentence and it’s not clear whether the biomarkers have an AND or OR relationship, assume it’s an OR criteria relationship between the biomarkers
  - If the biomarkers are on separate main bullet points in criteria section, assume those are AND criteria relationship and all the biomarkers should be in the same clause.
  - For multiple exclusion biomarker or histology criteria, those should be present in every clause JSON item



C) **Evaluation**
2 step evaluation could be done:
At extraction level (could also try comparing to the paper https://arxiv.org/pdf/2308.02180.pdf on our test set and focus on biomarkers, however, keep in mind that they consider biomarker in general not only genetic so think about that)
At post-processing level --> This is basically to make sure the post-processing is working as expected especially for gene naming! 
