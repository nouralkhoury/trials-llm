"""
This script is designed to generate simulated clinical trial data based on provided examples. It uses the OpenAI API to generate new clinical trial text and then extracts genomic biomarkers from the generated text. The script saves the simulated data as a JSON file.

Usage: python scripts/dataset_generation/simulated_data_gpt4.py
"""

import os
from datasets import load_dataset
import functools
from utils.jsons import dump_json
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import load_prompt
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from utils.jsons import load_json
from conf.config import PROCESSED_DATA, PROMPTS

# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    train_data = load_json("data/processed/ft_train.jsonl")
except Exception as e:
    print(f"Loading data from HuggingFace: {e}")
    dataset = load_dataset("nalkhou/clinical-trials", split=["train", "validation", "test"])
    train_data = dataset[0]

model_name = "gpt-4"

inference_prompt = load_prompt(os.path.join(PROMPTS, "zero-shot.json"))

prompt = """As an intelligent assistant, your task is to generate input and output pairs where input contains clinical trials eligibility criteria text and output contains text generated by extracting, processing, and structuring biomarkers from the clinical trials input that starts after the word "INPUT:". In output you must maintain the logical connections (AND, OR) between biomarkers and follow the provided instructions using Chain of Thought, reasoning, and common sense.

INSTRUCTIONS:
Return a JSON in the following format: {{"inclusion_biomarker": [], "exclusion_biomarker": []}}. Each key contains a list of lists of strings, where each inner list represents a set of genomic biomarkers required for patient inclusion or exclusion in the trial.

Extract biomarkers from the input only and avoid any information from the instructions. Focus on extracting biomarkers in the following categories: gene alteration (single mutation, fusion, rearrangement, copy number alteration, deletion, insertion, translocation), pathway alteration, gene expression, protein expression, path-way expression, HLA, TMB (tumor molecular burden, TMB-H or TMB-L), MSI (microsatellite instability, MSI-H, MSI-L, MSS, microsatellite stable) status, gene pathway alterations like dMMR (deficient Mismatch Repair Pathway) or pMMR (proficient Mismatch Repair), protein status, and HER2, ER, PgR, PD-L1 positive or negative status. Ignore any items that don't fall into these categories. Place the extracted biomarkers under the appropriate key, "inclusion_biomarker" or "exclusion_biomarker", based on whether they are required for patient inclusion or exclusion in the trial.

Omit information related to age, medical condition, potential pregnancy, stage or phase of disease, allergies, treatment history, histology, specific cancer type, diseases or conditions, HIV, and infections. Disregard data about levels, scores, doses, or ratio of expression, as well as any illnesses. Do not extract biomarkers associated with model experimental animals or historical data or previous studies.

Preserve the logical connection (AND, OR) between biomarkers in the input. Group biomarkers connected by 'AND' in the same list and place biomarkers connected by 'OR' in separate lists. Treat each main bullet in the "Inclusion Criteria" section as AND logic (unless specified as OR or different ARM/cohort) and main bullets in the "Exclusion Criteria" section as OR logic (unless explicitly stated otherwise). Handle ambiguous (AND, OR) logic by considering it as OR .

Process the biomarkers to ensure each one presents the gene name followed by the variant. Remove the words "gene", "allele", and "status" from the biomarker. Remove the term "mutation" from the biomarker when there's has a specific variant in the string (e.g CCND1 P287T mutation becomes CCND1 P287T). Make sure the variant is singular and noun-based (e.g "translocated" becomes "translocation"). Replace "mutant" with "mutation". Insert a space between the gene name and its variant, and also between the status and the hormone name. Replace the expression "positive expression" with just "expression". Replace symbols "-" and "+" with "negative" and "positive" respectlively, unless it's in the MSI status or known fusions separated by "-". When "germline" or "somatic" terms are mentioned in the input, place them in parentheses at the end of the corresponding biomarker. Ignore any biomarker mentioned as an "exception" or after "other than". Handle synonyms found in parentheses by extracting the biomarker but ignoring the synonym. Extract each biomarker once. Make sure to expand the biomarkers when needed.

Before returning the JSON output, remove any empty lists and stray empty lists caused by having no biomarkers in a category, ensuring that the keys "inclusion_biomarker" and "exclusion_biomarker" have just an empty list [] if there are no biomarkers in that category.

INPUT: {trial}
OUTPUT: {json_output}

INPUT: """


trial1 = """This is a Phase I/Ib study in which the safety of the combination therapy of RMC-4630 and\n LY3214996 in the treatment of KRAS mutant cancers will be studied.\n ;\n ;\n Inclusion Criteria:\n 1. Part A: Histological or cytological proof of advanced KRASm NSCLC, CRC or PDAC; PART\n B: Histological or cytological proof of advanced KRASm PDAC.\n 2. Age => 18 years;\n 3. Able and willing to give written informed consent;\n 4. WHO performance status of 0 or 1\n 5. Able and willing to undergo blood sampling for PK and PD analysis;\n 6. Able and willing to undergo tumor biopsies prior to start (or have undergone a biopsy\n within 2 months of inclusion), while on study treatment and upon progression of\n disease;\n 7. Life expectancy => 3 months and no deterioration or hospitalizations within 2 weeks\n leading to C1D1, allowing adequate follow up of toxicity evaluation and antitumor\n activity;\n 8. Evaluable disease according to RECIST 1.1 criteria; (PART A and PART B);\n 9. Women of childbearing potential must have a negative serum pregnancy test within 14\n days prior to registration and agree to use effective contraceptive methods, as\n defined in section 5.9.3, through-out the treatment period, and for 4 months after the\n study treatment\n 10. Adequate organ system function.\n Exclusion Criteria:\n 1. Part A: No excluded genotypes\n Part B: Excluded genotypes (including co occurring mutations):\n - NRAS (except G12A/C)\n - RASQ61\n - KRASG13\n - BRAF Class 1, 2, or unclassified\n - PIK3CA\n - STK11\n - KEAP1\n 2. Any treatment with investigational drugs within 30 days prior to receiving the first\n dose of investigational treatment;\n 3. Patients currently using concomitant medication that are strong inhibitors or inducers\n of CYP3A4;\n 4. History of another malignancy Exception PART A: Patients who have been disease-free\n for at least 3 years, or patients with a history of completely resected non-melanoma\n skin cancer and/or patients with indolent completely resected second malignancies are\n eligible. Exception PART B: Adequately treated carcinoma in situ of the cervix and\n adequately treated basal cell carcinoma of the skin.\n 5. Symptomatic or untreated leptomeningeal disease\n 6. Symptomatic brain metastasis. Patients previously treated or untreated for these\n conditions that are asymptomatic in the absence of corticosteroid and anticonvulsant\n therapy (for at least 4 weeks) are allowed to enroll. Radiotherapy for brain\n metastasis must have been completed at least 6 weeks prior to start of study\n treatment. Brain metastasis must be stable with verification by imaging (e.g.\n brain MRI or CT completed at screening demonstrating no current evidence of\n progressive brain metastases). Patients are not permitted to receive antiepileptic\n drugs or corticosteroids.\n 7. Patients who have had previous treatment with any targeted drug combination known to\n interfere RAS/MEK/MAPK pathway components.\n 8. Toxicities related to prior treatments > grade 1 (excluding alopecia)\n 9. History of interstitial lung disease or pneumonitis\n 10. Woman who are breast feeding;\n 11. Patients who have undergone any major surgery within the last 4 weeks prior to\n starting study drug or who would not have fully recovered from previous surgery.\n 12. Radio- or chemotherapy within the last 4 weeks prior to receiving the first dose of\n investigational treatment; except a palliative dose of radiation of 8 Gy, which is\n allowed up to one week before study start and should not be applied to the target\n lesion.\n 13. Uncontrolled infectious disease or known Human Immunodeficiency Virus HIV-1 or HIV-2\n type patients;\n 14. Patients with a known history of or uncontrolled hepatitis B (HBV) or C (HCV);\n 15. Patients with known alcoholism, drug addiction and/or psychiatric of physiological\n condition which in the opinion of the investigator would impair study compliance;\n 16. Patients with cardiac comorbidities (myocardial infarct within 6 months of study\n start, NYHA class \u2265 III, congestive heart failure or instable angina pectoris),\n uncontrolled hypertension (systolic blood pressure > 160 mm Hg and/or diastolic\n pressure > 90 mm Hg), prolonged QT interval(> 440 ms for men, > 460 ms for women) or\n patients who have had a stroke within 6 months prior to start study.\n 17. Other severe, acute, or chronic medical or psychiatric condition, laboratory\n abnormality active infections that may increase the risk associated with study\n participation or study drug administration or that may interfere with the\n interpretation of study results and, in the judgment of the investigator, would make\n the patient inappropriate for the study.\n 18. Patients with pulmonary embolisms or deep venous thrombosis (DVT) within 3 months\n prior to start\n 19. Known hypersensitivity to one of the study drugs or excipients.\n 20. Baseline diarrhea and/or any condition that would impair absorption of oral agents\n 21. Patient with a history or findings of central or branch retinal artery or venous\n occlusion with significant vision loss or other retinal diseases that cause current\n visual impairment or would likely cause visual impairment over the time period of the\n study, as assessed by an ophthalmologist."""


trial2 = """UP-NEXT is a double-blind, randomized, placebo-controlled study of the antibody-drug\n conjugate (ADC) XMT-1536 (upifitamab rilsodotin) administered as an intravenous infusion once\n every four weeks in patients with recurrent, platinum-sensitive high-grade serous ovarian\n cancer (HGSOC), including fallopian tube and primary peritoneal cancer, expressing high\n levels of NaPi2b.\n ;\n ;\n Inclusion Criteria:\n 1. Participant must have a histological diagnosis of high grade serous ovarian cancer,\n which includes fallopian tube and primary peritoneal cancer, that is metastatic or\n recurrent.\n 2. Participant must have platinum-sensitive recurrent disease, defined as having achieved\n either a partial or complete response to 4 or more cycles in their penultimate\n platinum- containing regimen and their disease progressing more than 6 months after\n completion of the last dose of platinum containing therapy in the penultimate regimen.\n 3. Participant must have had 4 to 8 cycles of platinum-based chemotherapy in 2nd to 4th\n line setting in their most recent treatment regimen as defined below:\n 1. Platinum-based chemotherapy regimens allowed immediately preceding enrollment to\n the study: carboplatin or cisplatin \u00b1: paclitaxel, docetaxel, pegylated liposomal\n doxorubicin or gemcitabine.\n 2. Participant must receive first study treatment infusion between 4 and 12 weeks\n after completing final dose of platinum in the most recent platinum-based\n regimen.\n 4. Participant must have had as their best response to last line of treatment one of the\n following: No Evidence of Disease (NED); Complete Response (CR); Partial Response\n (PR); OR Stable Disease (SD)\n 5. Participants with NED, CR, or PR as their best response to most recent line of\n treatment and who have not received treatment with a prior PARP inhibitor must have\n definitive BRCA1 and BRCA2 testing results that demonstrate no evidence of a\n deleterious BRCA1 or BRCA2 mutation. Somatic BRCA mutation testing is required for\n participants who are classified as not having a deleterious mutation by germline\n testing alone.\n 6. Participant must provide either a tumor tissue block or fresh cut slides for\n measurement of NaPi2b expression by a central laboratory. If sufficient archival tumor\n tissue is not available, then a tumor tissue block or slides must be obtained from a\n fresh biopsy and provided to the central laboratory. Confirmation of a\n NaPi2b-H/positive tumor by the central laboratory is required prior to randomization.\n Exclusion Criteria:\n 1. Participant has received prior treatment with mirvetuximab soravtansine or another ADC\n containing an auristatin or maytansinoid payload.\n 2. Participant has received bevacizumab in combination with last platinum-based regiment\n or plans to receive maintenance therapy outside the study intervention.\n 3. Participant has clinical signs or symptoms of gastrointestinal obstruction and/or\n requirement for parenteral hydration or nutrition.\n 4. Participant has ascites or pleural effusion managed with therapeutic paracentesis or\n thoracentesis within 28 days prior to signing the principal study consent form.\n 5. Participant has history of cirrhosis, hepatic fibrosis, esophageal or gastric varices,\n or other clinically significant liver disease. Testing beyond laboratory studies\n otherwise defined in the eligibility criteria, to diagnose potentially clinically\n significant liver disease based on risk factors such as hepatic steatosis or history\n of excessive alcohol intake, will be based on clinical judgement of the investigator.\n 6. Participant has history of or suspected pneumonitis or interstitial lung disease.\n 7. Participant has untreated CNS metastases (including new and progressive brain\n metastases), history of leptomeningeal metastasis, or carcinomatous meningitis."""


trial3 = """"The main objective of this trial is to explore the activity of chlorambucil, an alkylating\n agent commonly used in chronic lymphocytic leukemia treatment, in metastatic patients, gBRCA,\n including VUS, or DDR mutated, previously treated with a platinum-containing chemotherapy.\n ;\n ;\n Inclusion Criteria:\n 1. Pathologically confirmed pancreatic adenocarcinoma\n 2. Age \u2265 18 years\n 3. ECOG PS 0-2\n 4. Stage IV disease\n 5. Identified genetic aberrations that are associated with homologous recombination\n deficiency (HRD)\n 1. Cohort A: Documented mutation in gBRCA1 or gBRCA2 that is predicted to be\n deleterious or suspected deleterious\n 2. Cohort B: BRCA1 or BRCA2 mutations that are considered to be of uncertain/unknown\n significance (VUS)\n 3. Cohort C: Patients with other identified genetic aberrations that are associated\n with HRD\n 6. Adequate PFS during previous platinum-based chemotherapy for at least 4 months before\n progression\n 7. Screening laboratory values:\n Leukocytes > 3000/mmc Thrombocytes > 150000/mmc Hemoglobin > 10 g/dl Creatinine <2.0\n times upper normal limit (unless normal creatinine clearance). Total bilirubin < 2.0\n times upper normal limit (unless due to Gilbert's syndrome).\n Alanine aminotransferase (ALT) < 3.0 times upper normal limit.\n 8. Able to take oral medication\n 9. Progression during or after platinum-based chemotherapy\n 10. Other prior chemotherapy apart from first-line treatment for pancreatic cancer, are\n allowed, including maintenance treatment with PARP inhibitors\n 11. More than 2 weeks since prior chemotherapy end\n 12. Signed written informed consent\n 13. QTc <450 msec or QTc <480 msec for patients with bundle branch block\n Exclusion Criteria:\n 1. Clinically significant cardiac disease including unstable angina, acute myocardial\n infarction within 6 months prior to screening, congestive heart failure, and\n arrhythmia requiring therapy, with the exception of extra systoles or minor conduction\n abnormalities\n 2. Active and uncontrolled bacterial, viral, or fungal infection(s) requiring systemic\n therapy\n 3. Vaccination with vaccines called \"live\", since this treatment causes a drop of\n immunity defenses and a serious infection could result fatal.\n 4. History of seizure, head trauma and treatment with anti-epileptogenic drugs\n 5. Hypersensitivity to chlorambucil or to any excipients, in particular lactose\n 6. Recent radiotherapy (at least 4 weeks) or previous treatment with other cytotoxic\n agents\n 7. BRCA-mutated advanced pancreatic cancer who did not undergo maintenance with olaparib\n after platinum-based chemotherapy\n 8. Mismatch repair (MMR)/high levels of microsatellite instability (MSI-H), or high\n levels of tumor mutational burden (TMB) pancreatic cancer who did not undergo\n immunotherapy with pembrolizumab monotherapy or any other anti-PD1 agent\n 9. Concomitant PARP inhibitors therapy\n 10. Life expectancy less than 3 months, in the opinion of the investigator\n 11. Other past or current malignancy. Subjects who have been free of malignancy for at\n least 5 years, or have a history of completely resected non-melanoma skin cancer, or\n successfully treated in situ carcinoma are eligible\n 12. Symptomatic duodenal stenosis\n 13. CT contrast medium allergy and claustrophobia to RM investigation\n 14. Any significant medical condition laboratory abnormality, or psychiatric illness that\n would prevent the subject from participating in the study\n 15. Any condition including the presence of laboratory abnormalities, which places the\n subject at unacceptable risk if he/she were to participate in the study\n 16. Any condition that confounds the ability to interpret data from the study\n 17. Any familiar, sociologic or geographic conditions that can potentially interfere with\n the adhesion to the protocol or to the follow-up\n 18. Pregnant or nursing. Adequate contraception is defined as oral hormonal birth control,\n intrauterine device, and male partner sterilization (if male partner is sole partner\n for that subject) and the double barrier method (condom or occlusive cap plus\n spermicidal agent).\n 19. Concurrent treatment with other experimental drugs"""


trial4 = """This is a prospective, single arm study to investigate the efficacy and safety furmonertinib\n 80mg/d as adjuvant treatment for 3 years post surgery of stage IA with high-risk factors and\n stage IB non-small cell lung cancer. A total of 114 patients would be enrolled. The primary\n endpoint is the disease-free survival rate at 3 years.\n ;NA;\n Inclusion Criteria:\n - Received radical resection of non-small cell lung cancer without prior anti-tumor\n therapies including radiotherapy, chemotherapy, target therapy and immunotherapy.\n - Histologically diagnosed Non-small cell lung cancer based on the judgement of at least\n 2 pathologists.\n - Stage IA with high risk factors including micropapillae or solid components, vascular\n invasion, spread through air spaces, low differentiation, tumor budding and\n insufficient lymph node dissection; Stage IB with or without high-risk factors. The\n pathological stage is based on the 8th edition of AJCC lung cancer staging.\n - EGFR mutation positive according to NGS testing by tissue, including deletions in exon\n 19, L858R, S768I, G719X, L861Q, T790M mutations et al.\n - ECOG performance status 0-1.\n - Sufficient organ function in liver, renal, kidney and hematology.\n - With written signed informed consent form, ability to report adverse events, and good\n adherence to clinical study.\n Exclusion Criteria:\n - Lung cancer with small cell or neuroendocrine cancer cell.\n - EGFR exon 20 insertion positive.\n - Concurrent with other diver mutations including alterations in ALK, ROS1, MET et al.\n - Women who are pregnant or breastfeeding.\n - Use of CYP3A4 strong depressant within 7 days or CYP3A4 strong inducer within 21 days\n prior to initial administration, use of other anti-tumor treatment including\n traditional Chinese medicine within 14 days before enrollment.\n - Concurrent with other malignancies excluding carcinoma in situ.\n - With uncontrolled systematic diseases such as active bleeding, unstable angina, heart\n infarction within 1 year, chronic heart failure and uncontrolled hypertension and\n diabetes mellitus; with active infection of HBV, HCV or HIV, or other infections\n requiring injection of antibiotics.\n - Gastrointestinal disorders which may affect drug taking or absorption.\n - With history of QT prolongation or relative risk factors including heart failure,\n hypokalemia, congenital long QT syndrome, family history of long QT syndrome et al.\n - With history of interstitial lung disease or relative risk.\n - Allergic to any component of furmonertinib tablet.\n - Mental illness or drug abuse.\n - Live vaccination within 30 days before enrollment.\n - Other situation judged by investigator such as failure to follow the rules of study.\n - Attending another study of investigational drug, or received other study drugs or\n medical devices with 4 weeks before enrollment."""

prompt = """You are a clinical trials expert. Below are given 4 examples of clinical trials. Generate 2 samples of clinical trials which match the style of the examples given and they have genomic biomarkers in the inclusion and exclusion criteria. 
Trial: {trial1}

Trial: {trial2}

Trial: {trial3}

Trial: {trial4}

Trial: """


prompt_template = PromptTemplate(
    input_variables=["trial1", "trial2", "trial3", "trial4"],
    template=prompt)

# Initialize the OpenAI model for chat
openai_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                          temperature=0.5,
                          model=model_name)


def generate_data(trial1, trial2, trial3, trial4):
    chain = LLMChain(llm=openai_model, prompt=prompt_template)
    response = chain.run(trial1 = trial1, trial2 = trial2, trial3 = trial3, trial4 = trial4)
    return response


gen_data = generate_data(trial1, trial2, trial3, trial4)


# Inference

# Initialize the OpenAI model for chat
inf_openai_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                              temperature=0,
                              model=model_name)

# Function to generate inference based on given trial
@functools.cache
def generate_inference(trial):
    chain = LLMChain(llm=inf_openai_model, prompt=inference_prompt)
    response = chain.run(trial=trial)
    return response


# Function to generate simulated data based on given trials
def generate_simulated_data(trial1, trial2, trial3, trial4):
    simulated = []

    gen_data = generate_data(trial1, trial2, trial3, trial4)

    new_trials = [s for s in gen_data.split("Trial:") if s.strip()]

    for t in new_trials:
        pred_inf = generate_inference(t.strip())
        simulated.append({"input": t, "output": pred_inf})
    return simulated


# Initialize list to store simulated data
simulated_data = []

# Generate simulated data
for i in range(20):
    simulated_data += generate_simulated_data(trial1, trial2, trial3, trial4)

    dump_json(simulated_data, os.path.join(PROCESSED_DATA, "simulated.json"))
