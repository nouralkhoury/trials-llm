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
        a) Gene level: Let the gene be named as used in CIViC (using synonyms form NCBI api, string alignment if exact match is not found).<br>
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

<br><br>

>[!IMPORTANT]
> - Might not need to stratify when creating dataset train/test, LLM is not like supervised learning. Having a balanced data is not an issue. It is just showing it how to respond in a structured manner. There's no class to return</span>
> - This is a short-listing task more than completely matching patient. We would need more patient data to be able to do that and also the person would still have to go through the inital screening before enrollement. This is a shortcut to finding trials (SHORT-LIST)

<br>

> [!TIP]
> Helpful guidelines when annotating dataset
> - Consider all biomarkers (gene mutation, gene/protein expression, HR/HER2/PR/ER status, dMMR/pMMR/MSI/TMB/PD-L1 status, pathway alterations, etc.)
> - If gene lists for the pathway are not listed, extract the name of the pathway (genes corresponding to pathway can be handled post-processing)
> - If biomarker is in both inclusion and exclusion, extract it anyway (we can add later a condition if the biomarker is in both inclusion and exclusion still return the matched trial)
> - If gene is missing from biomarker, do not extract it (e.g T315I, we don't know which gene this is belonging to!)
> - In situations where we have If-then conditions or the biomarker is a secondary biomarker, extract it anyway.
> - If the inclusion criteria are for either of multiple biomarkers (and/or), list each of those biomarkers in a separate clause or JSON item because either of them being satisfied should contribute to the entire expression being True.
> - When there is a list of biomarker criteria in the same sentence and it’s not clear whether the biomarkers have an AND or OR relationship, assume it’s an OR criteria relationship between the biomarkers.
> - If the biomarkers are on separate main bullet points in the criteria section, assume those are AND criteria relationship, and all the biomarkers should be in the same clause.
> - Main bullet points: In inclusion treat them as AND unless stated otherwise. In exclusion treat them as OR (one or more exlusion criteria will exclude patient)
> - If main bullet points are each for a different ARM or COHORT or PHASE of the trial, treat them as OR
> - If the presence or abscence of biomarker does not affect inclusion do not extract this and do not include it in the inclusion biomarker
> - If there is an exception in inclusion or a NOT, include it in exclusion biomarker
> - If there is an except in the exclusion, include it in the exclusion_except entity
> - replace "uncommon" with rare (example uncommon EGFR exon 18-21 mutation --> EGFR rare exon 18-21 mutation)


C) **Evaluation**
2 step evaluation could be done:
At extraction level (could also try comparing to the paper https://arxiv.org/pdf/2308.02180.pdf on our test set and focus on biomarkers, however, keep in mind that they consider biomarker in general not only genetic so think about that)
At post-processing level --> This is basically to make sure the post-processing is working as expected especially for gene naming! 
