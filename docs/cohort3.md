

# 20 Clinically Meaningful Cohort Definitions for Predictive Modeling

Below are 20 diverse cohort definitions derived from longitudinal claims data. Each cohort represents a future clinical event to predict, defined using ICD-10-CM diagnosis codes, CPT/HCPCS procedure codes, GPI drug codes, and basic demographics. The definitions include the clinical rationale, inclusion/exclusion criteria, and the target event (outcome) for prediction.

## 1. New-Onset Diabetic Nephropathy in Type 2 Diabetes  
**Clinical Rationale:** Predicting diabetic nephropathy (kidney disease due to diabetes) in type 2 diabetics is crucial because it is a significant cause of chronic kidney disease and end-stage renal failure globally ([
            Diabetic nephropathy – complications and treatment - PMC
        ](https://pmc.ncbi.nlm.nih.gov/articles/PMC4206379/#:~:text=Diabetic%20nephropathy%20is%20a%20significant,and%20expanded%20the%20potential%20therapies)). Early prediction allows intensified glycemic and blood pressure control to slow kidney damage. Diabetic nephropathy greatly increases morbidity and often leads to costly renal replacement therapy (dialysis or transplant).  

**Inclusion Criteria:**  
- Adults (e.g., age ≥18) with established **Type 2 Diabetes Mellitus** (ICD-10-CM E11.x series for type 2 diabetes).  
- Evidence of ongoing diabetes management in baseline (e.g., ≥2 outpatient claims or 1 inpatient claim with E11.x diagnoses).  
- No prior indicators of diabetic kidney complications (to ensure “new onset” prediction), i.e., **no prior diabetic nephropathy codes** in history.  

**Exclusion Criteria:**  
- **Type 1 diabetes** (ICD-10 E10.x) or other specific diabetes types, to focus on type 2 diabetics.  
- Any **existing diabetic nephropathy or CKD** diagnosis before the index date (e.g., ICD-10 E11.21 for diabetic nephropathy, N18.x for chronic kidney disease).  
- Patients already on **dialysis** or with **end-stage renal disease** (ESRD) prior to index (e.g., ICD-10 N18.6 or Z99.2) – since nephropathy has already progressed in these cases.  

**Target Event Definition:**  
- **New diagnosis of diabetic nephropathy** during the prediction window – for example, an occurrence of ICD-10-CM code **E11.21** (Type 2 diabetes mellitus with nephropathy) or **E11.22** (with chronic kidney disease) on a claim, indicating onset of diabetic kidney disease. This may be identified by the first appearance of any E11.2x diagnosis code in the claims data for that patient (with no such codes in baseline).

## 2. Incident Heart Failure in At-Risk Patients  
**Clinical Rationale:** Heart failure (HF) often develops in patients with long-standing hypertension, coronary artery disease, or diabetes. It is a major cause of morbidity and has roughly a 50% five-year mortality rate after onset ([The Burden of Disease of HFpEF](https://www.cfrjournal.com/articles/epidemiology-clinical-characteristics-and-cause-specific-outcomes-heart-failure-preserved#:~:text=Nationwide%20statistics%20from%20the%20US,CV%20death%20was%20more%20frequent)). Predicting new-onset HF in high-risk individuals allows early intervention (e.g., aggressive risk factor management, cardioprotective therapies). Given the high hospitalization and mortality burden of HF, such a model could significantly impact outcomes.  

**Inclusion Criteria:**  
- Adults (e.g., age ≥50) with **risk factors for heart failure**, such as:  
  - **Coronary artery disease** (CAD) or prior myocardial infarction (ICD-10 I25.x for chronic ischemic heart disease, I21.x for acute MI history).  
  - **Long-standing hypertension** (ICD-10 I10) and/or **type 2 diabetes** (ICD-10 E11.x), which commonly precede HF.  
- No history of heart failure: **absence of any HF diagnosis** (ICD-10 I50.x) in the baseline period.  
- Could also include patients with prior revascularization (CPT codes for PCI or CABG) or other evidence of cardiomyopathy risk, as long as they have no prior HF diagnosis.

**Exclusion Criteria:**  
- **Existing heart failure or cardiomyopathy** prior to index (any I50.x or cardiomyopathy code I42.x in claims history). These patients are not “incident” HF.  
- Conditions that mimic HF or would confound the outcome, if necessary (e.g., end-stage COPD causing cor pulmonale, though primary exclusion is just prior HF).  
- If focusing on non-congenital HF, exclude patients with complex congenital heart disease causing HF in youth.

**Target Event Definition:**  
- **New onset of heart failure**, defined by the first occurrence of an HF diagnosis code in claims. For example, an inpatient or outpatient claim with ICD-10 **I50.1 – I50.9** (heart failure codes) during the follow-up. This could be refined to a hospital admission with HF as a principal diagnosis to indicate a significant new HF event (e.g., an admission coded I50.21 for acute systolic HF). The key is that this code appears after the index and was absent in baseline.

## 3. Progression to End-Stage Renal Disease (ESRD) Requiring Dialysis  
**Clinical Rationale:** For patients with chronic kidney disease (CKD), progression to end-stage renal disease requiring dialysis is a critical transition. ESRD patients face very high morbidity and mortality; annual mortality on dialysis is ~9% with only ~40–50% five-year survival ([What are Survival Rates for Patients on Dialysis? - Renal Fellow Network](https://www.renalfellow.org/2018/09/19/what-are-survival-rates-for-dialysis-patients/#:~:text=In%20summary%2C%C2%A0%20overall%20mortality%20is,Thus%20far%2C%20studies)). Predicting which CKD patients will progress to needing dialysis can prompt earlier nephrology referral, preparation (e.g., fistula placement), or more aggressive measures to slow renal decline. It also has cost implications, as dialysis treatment is expensive and life-altering.  

**Inclusion Criteria:**  
- Patients with **advanced CKD** in the baseline period, for example:  
  - **ICD-10 N18.4 or N18.5** (CKD stage 4 or 5) diagnoses on claims, **or** multiple N18.3 (stage 3) codes indicating progressive disease.  
  - Alternatively, evidence of significantly reduced kidney function or nephrologist care in claims.  
- Typically adults (e.g., age 18+) since ESRD from CKD usually occurs in adulthood (pediatric ESRD would be a separate consideration).  
- Continuous enrollment for a baseline period (e.g., 1–2 years) to observe CKD history and confirm no prior dialysis.

**Exclusion Criteria:**  
- **Patients already on dialysis or with ESRD before index**, identified by prior CPT/HCPCS codes for dialysis sessions (e.g., CPT 90935, 90937 for hemodialysis) or an ESRD diagnosis code (ICD-10 N18.6).  
- **History of kidney transplant** prior to index (ICD-10 Z94.0) – such patients have ESRD but took a different route (transplant) and are not at risk for dialysis initiation.  
- Patients with acute kidney injury (AKI) without chronic disease (we want chronic progression), or those with reversible causes of renal failure in baseline.  

**Target Event Definition:**  
- **Initiation of chronic dialysis**, signifying ESRD onset. This can be captured by the first appearance of a dialysis procedure claim or ESRD diagnosis: for example, a CPT code for outpatient dialysis (e.g., **90935** – hemodialysis, single session) or a HCPCS code such as **G0257** (home dialysis training) after the index date. Alternatively, an inpatient stay with **procedure codes for dialysis** or a new ICD-10-CM **N18.6 (ESRD)** or **Z99.2 (dependence on dialysis)** would qualify. Any such event indicates the patient has progressed to ESRD and begun renal replacement therapy.

## 4. Incident Liver Cirrhosis in Chronic Liver Disease  
**Clinical Rationale:** Many chronic liver disease patients (e.g., with chronic hepatitis B or C, alcoholic liver disease, or non-alcoholic fatty liver disease) eventually develop cirrhosis. Cirrhosis marks a point of irreversible liver scarring associated with high risks of liver failure, variceal bleeding, and liver cancer. In fact, cirrhosis vastly increases mortality risk (by 5–10 fold) due to its complications ([Global burden of liver disease: 2023 update - Journal of Hepatology](https://www.journal-of-hepatology.eu/article/S0168-8278(23)00194-0/fulltext#:~:text=Global%20burden%20of%20liver%20disease%3A,related%20complications%20%28ascites%2C%20variceal)). Predicting who will progress to cirrhosis allows intervention (e.g., antiviral treatment for hepatitis, lifestyle changes for NASH, alcohol cessation) before irreversible damage occurs.  

**Inclusion Criteria:**  
- Adults with **chronic liver disease** but without cirrhosis at baseline. For example:  
  - **Chronic viral hepatitis**: ICD-10 B18.0 (chronic hepatitis B) or B18.2 (chronic hepatitis C).  
  - **Alcoholic liver disease**: ICD-10 K70.x (e.g., K70.30 alcoholic cirrhosis **without** mention of cirrhosis if early stage, or other alcoholic liver disease codes).  
  - **Non-alcoholic fatty liver disease (NAFLD)** or fibrosis: ICD-10 K76.0 (fatty liver) or K74.0-K74.2 (hepatic fibrosis codes without cirrhosis).  
- No prior cirrhosis diagnosis (see exclusion) and at least one of the above chronic liver disease diagnoses in baseline. Possibly elevated liver enzymes or imaging (if CPT for FibroScan etc. were available, but usually not needed; diagnosis codes suffice).  
- Sufficient follow-up time expected (since cirrhosis development may take years, though model can predict within a certain window).

**Exclusion Criteria:**  
- **Existing cirrhosis** before the index date: any ICD-10 code for cirrhosis (e.g., **K74.60, K74.69** for unspecified cirrhosis) in claims history.  
- **Prior liver transplant** recipients (ICD-10 Z94.4) – they had advanced disease already.  
- **Known hepatocellular carcinoma** (ICD-10 C22.0) or other end-stage complications at baseline, as these often imply underlying cirrhosis already present.  
- If focusing on “incident” cirrhosis, exclude patients with decompensated liver disease signs in baseline (ascites, varices – ICD-10 R18.8, I85.x) which usually signal cirrhosis.

**Target Event Definition:**  
- **New diagnosis of liver cirrhosis** in the follow-up period. This can be identified by the first appearance of an ICD-10-CM cirrhosis code, such as **K74.60** (unclear cirrhosis) or **K74.69** (other cirrhosis of liver), on any claim. Additionally, evidence of decompensation can reinforce the event: e.g., an inpatient admission with **cirrhosis** and complications (ascites, variceal hemorrhage). Any such code, not present previously, marks the transition to cirrhosis. The cohort’s predicted outcome is thus the **onset of cirrhosis** among those with chronic liver disease.

## 5. Coronary Artery Bypass Graft (CABG) Surgery  
**Clinical Rationale:** Coronary artery bypass grafting is a major surgical procedure for severe coronary artery disease (CAD). CABG is the most common adult cardiac surgery – almost 400,000 CABG surgeries are performed each year in the US ([Coronary Artery Bypass Graft - StatPearls - NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK507836/#:~:text=Coronary%20artery%20bypass%20grafting%20,1)). It’s typically indicated in patients with multi-vessel CAD or left main disease to improve survival and symptoms. Predicting which patients will need CABG can help identify those with progressing CAD (perhaps to consider more aggressive medical therapy or earlier intervention). It’s also relevant for cost prediction, as CABG is resource-intensive.  

**Inclusion Criteria:**  
- Patients with known **Coronary Artery Disease** but who have not yet undergone CABG. For example:  
  - **Chronic ischemic heart disease** (ICD-10 I25.xx codes) or **history of myocardial infarction** (I25.2 or I21.x in past).  
  - Possibly those who have had **percutaneous coronary intervention (PCI)** already (CPT codes 92920–92928 for stents) but with recurrent symptoms – though prior PCI isn’t required, just CAD presence.  
  - Could include **angina pectoris** diagnoses (ICD-10 I20.x) indicating significant CAD.  
- Typically middle-aged or older adults (e.g., >40), since CABG is rare in younger patients absent congenital issues.  
- Evidence of **persistent or severe CAD** in claims: multiple cardiology visits, stress tests (CPT 93015) or angiography (CPT 93454) could be present, but not strictly required.

**Exclusion Criteria:**  
- **Prior CABG surgery**: identified by historical procedure codes (CPT 33510–33514, 33533–33536 etc. for CABG) or diagnosis code Z95.1 (presence of aortocoronary bypass graft). These patients are not candidates for a “first CABG” prediction as they already had it.  
- **Acute situations where CABG is emergent at baseline**, e.g., someone in baseline who already had an acute MI that led to an immediate CABG – the model should start prior to that. Essentially, the event should not have occurred yet in baseline.  
- If focusing only on elective/planned CABG, could exclude patients who die of CAD or get heart transplant before CABG, but generally prior CABG is the main exclusion.

**Target Event Definition:**  
- **Undergoing CABG surgery**, identified by relevant procedure codes in the claims during follow-up. For example, the occurrence of **CPT code 33533** (CABG, single arterial graft) or **33512** (CABG, two vein grafts), etc., or an ICD-10-PCS code for coronary bypass. A range of CPT codes 33510–33536 covers various CABG procedures. Any such code appearing (with no CABG codes prior) indicates the patient had a coronary bypass operation. The model’s outcome is thus the **first CABG procedure** after the index date.

## 6. Total Knee Replacement for Advanced Knee Osteoarthritis  
**Clinical Rationale:** Total knee arthroplasty (replacement) is a common surgery for end-stage knee osteoarthritis (OA) resulting in severe pain and functional limitation. Over 700,000 knee replacements are performed annually in the U.S. ([Total Knee Replacement - OrthoInfo - AAOS](https://orthoinfo.aaos.org/en/treatment/total-knee-replacement/#:~:text=Total%20knee%20replacement%20surgery%20was,S)), reflecting how frequent and impactful this outcome is. Predicting who will undergo a knee replacement can help target interventions like physical therapy or injections earlier to possibly delay surgery, and it’s useful for resource planning. Knee OA progression is generally gradual, so claims data can capture a history of treatments leading up to surgery.  

**Inclusion Criteria:**  
- Patients with **chronic knee osteoarthritis**: identified by ICD-10-CM **M17.x (Osteoarthritis of knee)** on claims. Typically, require multiple mentions or a long history to ensure it’s advanced disease.  
- Usually older adults (e.g., age 50 or above) because degenerative joint disease requiring replacement is most common in that group.  
- Evidence of prior knee OA treatments in baseline could strengthen the cohort (not mandatory but often present): e.g., **injections** (CPT code 20611 for knee steroid injection), **physical therapy** visits, or long-term NSAID use (GPI for NSAIDs) – indicating refractory symptoms.  
- No prior knee replacement (the knee replacement should be a future event, see exclusion).

**Exclusion Criteria:**  
- **Prior total knee replacement** on the relevant knee. If a patient already had a knee arthroplasty (CPT 27447 for TKA), they would be excluded for that side; if one knee is done, sometimes the cohort might still consider the other knee, but generally exclude anyone with any knee replacement to focus on first-ever TKA.  
- **Inflammatory arthritis** patients (like rheumatoid arthritis) could be excluded if we want pure OA, as their indications for surgery differ. (This is optional; they can be included if they also have OA codes.)  
- Patients with conditions making them unlikely to undergo elective surgery (e.g., terminal illnesses) might be excluded if focusing on those likely to get to surgery.

**Target Event Definition:**  
- **Total Knee Arthroplasty (TKA) procedure** as captured by procedure codes. For instance, **CPT code 27447** (arthroplasty, knee, total) is the typical code for a total knee replacement. Alternatively, ICD-10-PCS codes for knee replacement could be used in inpatient claims. The event is counted when such a code appears, indicating the patient underwent knee replacement surgery. This is usually in an inpatient setting. The target outcome is specifically the **occurrence of a primary total knee replacement** during the follow-up period.

## 7. Bariatric Surgery in Morbid Obesity  
**Clinical Rationale:** Bariatric surgery (such as gastric bypass or sleeve gastrectomy) is a treatment for patients with severe obesity, leading to substantial and sustained weight loss and improvement in comorbidities (diabetes, hypertension, etc.). More than 250,000 bariatric surgeries were performed in the U.S. in 2018 ([Long-term Study of Bariatric Surgery for Obesity: LABS - NIDDK](https://www.niddk.nih.gov/about-niddk/research-areas/obesity/longitudinal-assessment-bariatric-surgery#:~:text=Over%20the%20years%2C%20bariatric%20surgery,the%20surgery%20still%20involves%20risks)), underscoring its growing use. Predicting which obese patients will opt for or require bariatric surgery helps target intensive weight management or assess future healthcare costs. It’s also clinically relevant for identifying patients who are not losing weight with medical therapy and may progress to needing surgery.  

**Inclusion Criteria:**  
- Adults with **morbid obesity** or **severe obesity**. In claims, this can be identified by:  
  - ICD-10-CM **E66.01 (Morbid obesity)** or E66.2 (morbid obesity with alveolar hypoventilation) in the problem list.  
  - BMI codes if available: e.g., ICD-10 Z68.41 (BMI 40.0–44.9) or higher (these are often recorded on claims).  
- Often, inclusion may require **BMI ≥ 40, or ≥35 with comorbid conditions** (which mirrors clinical criteria for surgery). Comorbidities like type 2 diabetes (E11.x), OSA (G47.33), or hypertension (I10) might be present.  
- Possibly evidence of prior weight loss attempts (dietitian visits, weight management program codes), but not required. The main inclusion is the extreme obesity diagnosis itself, indicating candidacy for surgery.

**Exclusion Criteria:**  
- **Prior bariatric surgery** or gastrointestinal bypass procedures, identified by past procedure codes (e.g., CPT 43644 for gastric bypass, 43775 for sleeve gastrectomy) or ICD-10 Z98.84 (history of bariatric surgery). Those patients have already had the outcome.  
- Patients with contraindications for surgery that would definitively prevent them from undergoing it (for example, unstable cardiac conditions or active substance abuse – though these might not be exclusions unless explicitly desired).  
- If the aim is specifically **first-time** bariatric surgery, exclude those with any evidence of previous weight-loss surgery or gastric banding (CPT 43770, etc.).

**Target Event Definition:**  
- **Bariatric surgery procedure** performed in the follow-up. This can be identified by CPT or HCPCS codes for bariatric operations, for example:  
  - **CPT 43644** – Laparoscopic gastric bypass (Roux-en-Y).  
  - **CPT 43775** – Laparoscopic sleeve gastrectomy.  
  - Other related codes (43645, 43659 for various bariatric procedures).  
  - HCPCS code **S2083** (Adjustment of gastric band) could indicate lap-band, but primary focus is on the initial surgery.  
- The occurrence of any such code indicates the patient underwent bariatric surgery. The cohort’s outcome is the **initiation of a bariatric surgical intervention** for weight loss.

## 8. Elective Hysterectomy for Uterine Fibroids  
**Clinical Rationale:** Uterine fibroids (leiomyomas) are extremely common benign tumors in women of reproductive age, often causing heavy menstrual bleeding, pelvic pain, and anemia. Hysterectomy (surgical removal of the uterus) is a definitive treatment for symptomatic fibroids and is very frequently performed – hysterectomy is the most common non-pregnancy-related major surgery in women ([Hysterectomy: Background, Problem, Epidemiology](https://emedicine.medscape.com/article/267273-overview#:~:text=Hysterectomy%20is%20the%20most%20common,the%20fallopian%20tubes%20and%20ovaries)), with fibroids being a leading indication (accounting for about one-third of hysterectomies) ([Hysterectomy: Background, Problem, Epidemiology](https://emedicine.medscape.com/article/267273-overview#:~:text=Epidemiology%20of%20fibroids)). Predicting which patients with fibroids will proceed to hysterectomy can help in offering alternative therapies (like uterine artery embolization or medical management) earlier or planning for surgical resources.  

**Inclusion Criteria:**  
- **Women of adult age** (commonly age 30–55, i.e., premenopausal to perimenopausal age range) with **uterine fibroids**. Identification via:  
  - ICD-10-CM **D25.x codes** (D25.0–D25.9) for uterine leiomyoma on claims. Typically multiple occurrences or persistent symptoms (e.g., fibroid diagnosis plus heavy bleeding codes).  
- May also include those with **menorrhagia or uterine bleeding** (ICD-10 N92.x, N93.x) and evidence of fibroids on imaging (though imaging might not show in claims, the diagnosis code suffices).  
- The baseline period should show fibroid-related healthcare utilization (e.g., gynecologic visits, ultrasounds CPT 76856, prescriptions for anemia or hormone therapy), indicating symptomatic fibroids.  

**Exclusion Criteria:**  
- **Prior hysterectomy** before the index date: identified by CPT codes (e.g., 58150 for abdominal hysterectomy, 58570 for laparoscopic hysterectomy) or diagnosis code Z90.710 (acquired absence of uterus). If the uterus is already removed, they can’t have a future hysterectomy.  
- Women with **uterine or cervical cancer** (ICD-10 C54.x, C53.x) at baseline are typically excluded because if they undergo hysterectomy, it’s for malignancy, not an “elective fibroid” indication – a different cohort. We want benign-indication hysterectomy.  
- If focusing on elective (non-emergency) surgery, exclude any emergent pelvic surgeries or significant comorbid conditions that would preclude elective surgery (though these are less common considerations).

**Target Event Definition:**  
- **Hysterectomy procedure**, specifically for a benign indication, occurring in the follow-up period. This can be captured by procedure codes such as:  
  - **CPT 58150** (Total abdominal hysterectomy), **58552** (Laparoscopic hysterectomy with removal of tubes/ovaries), **58571** (Laparoscopic supracervical hysterectomy), etc.  
  - HCPCS code **0UT90ZZ** (if using ICD-10-PCS for uterus removal) for inpatient procedures.  
- The first occurrence of any hysterectomy surgery code after the cohort index will count as the outcome. It implies the patient’s fibroids (or other benign uterine condition) were ultimately managed with surgical removal of the uterus.

## 9. Insulin Initiation in Type 2 Diabetes  
**Clinical Rationale:** Over the course of type 2 diabetes (T2DM), many patients experience progression of beta-cell failure requiring escalation to insulin therapy. Initiating insulin is a significant milestone, often indicating that oral medications no longer suffice. Approximately 15–25% of type 2 diabetics on oral therapy will require insulin within about 6 years ([
            Progression to insulin therapy among patients with type 2 diabetes treated with sitagliptin or sulphonylurea plus metformin dual therapy - PMC
        ](https://pmc.ncbi.nlm.nih.gov/articles/PMC5033027/#:~:text=Previous%20real%E2%80%90world%20observational%20studies%20suggest,as%20there%20may%20be%20some)), and eventually more over longer periods. Predicting insulin start is important for patient education and timely intensification of treatment. Early identification of those likely to need insulin can prompt closer monitoring or earlier introduction of adjunct therapies to control blood sugar.  

**Inclusion Criteria:**  
- Adults with **Type 2 Diabetes Mellitus** on non-insulin therapy. Concretely:  
  - **ICD-10 E11.x** diagnosis in baseline (T2DM).  
  - Evidence of **oral anti-diabetic medications** use in claims: e.g., GPI codes for metformin (GPI starting 2715), sulfonylureas (GPI 2720), DPP-4 inhibitors, etc., or an absence of any insulin fill (to ensure not already on insulin).  
- **No insulin use** prior to index (since we are predicting the first initiation). This can be checked via pharmacy claims: no GPI 2710 (insulin class) in medication claims history.  
- Possibly sub-criteria like **poor glycemic control proxies** (e.g., ICD-10 R73.9 pre-diabetes or frequent medication up-titration) to enrich for those likely to progress, but not strictly required.  

**Exclusion Criteria:**  
- **Type 1 diabetes** patients (ICD-10 E10.x) – they usually start insulin at diagnosis, a different scenario.  
- Patients already on **insulin or an insulin pump** in baseline (any insulin prescription fill, GPI codes 2710xxxx for insulins, or HCPCS J1817 for insulin for pump, etc.). We only want those who have never been on insulin yet.  
- **Gestational diabetics** (ICD-10 O24.4x) can be excluded if focusing on standard T2DM, since their insulin use is typically temporary and pregnancy-related.  

**Target Event Definition:**  
- **Initiation of insulin therapy**, evidenced by the first pharmacy claim for an insulin product (or medical claim for an insulin injection if using medical billing for some insulins). For example:  
  - A pharmacy claim with a **GPI code starting with 2710** (the GPI therapeutic class for insulins ([Microsoft Word - Document5](https://s3.amazonaws.com/ajmc/_media/_pdf/AJMC_02_2018_Agiro%20eAppendix.pdf#:~:text=Insulins1%20Basal%2027104003%2C%2027104006%2C%2027101020%2C,Premixed%2027104080%2C%2027104070%2C%2027103070%2C%2027104090))) – this would cover any insulin (basal or rapid-acting).  
  - Alternatively, an HCPCS code like **J1817** (insulin for pump) or **J1815** (insulin injection per 5 units) if administered in a clinic, though in practice initiation is usually seen in pharmacy data.  
- The date of that first insulin fill is the event. The outcome is binary: did the patient start insulin (yes/no within the prediction window). For example, first-time appearance of an insulin such as insulin glargine, lispro, NPH, etc., in the claims.

## 10. Biologic DMARD Initiation in Rheumatoid Arthritis  
**Clinical Rationale:** Rheumatoid arthritis (RA) is often managed first with conventional DMARDs like methotrexate. If disease remains moderate-to-severe, biologic DMARDs (e.g., TNF inhibitors, IL-6 inhibitors, JAK inhibitors) are added. Starting a biologic is a pivotal escalation step, usually indicating inadequate response to initial therapy. Biologics have markedly improved RA outcomes – TNF inhibitors and others significantly reduce joint damage and improve quality of life ([TNF inhibitor therapy for rheumatoid arthritis - PubMed](https://pubmed.ncbi.nlm.nih.gov/24648915/#:~:text=Immunotherapy%20has%20markedly%20improved%20treatment,ADA)) – but are costly and carry risks. Predicting which RA patients will require biologic therapy could help in tailoring aggressive treatment early for those individuals or in cost forecasting for payers.  

**Inclusion Criteria:**  
- Patients with **Rheumatoid Arthritis**. Look for ICD-10-CM codes **M05.x or M06.x** (seropositive or seronegative RA) on claims, usually with rheumatologist involvement.  
- Evidence of current or recent use of conventional synthetic DMARDs in baseline (optional but typical): e.g., pharmacy claims for methotrexate (GPI 6720005000) or hydroxychloroquine, sulfasalazine, etc., indicating they are on first-line therapy.  
- No biologic use yet (since we want to predict first biologic initiation).  
- Age: typically adult (RA is most common in middle age; juvenile RA could be excluded or considered separately).  

**Exclusion Criteria:**  
- **Prior or current biologic DMARD use** before index. This can be identified via pharmacy claims (GPI codes for biologics, e.g., GPI 6627**** for TNF inhibitors like adalimumab ([[DOC] Document (31.42 KB) - Value in Health](https://www.valueinhealthjournal.com/cms/10.1016/j.jval.2017.02.005/attachment/bf3ba157-8446-482e-b4c6-7bbfd9f3b563/mmc2.docx#:~:text=...%20by%20a%29%20J,J0135))) or medical claims (HCPCS J-codes: e.g., J1745 for infliximab, J0129 for abatacept, etc.). If any biologic has been billed previously, they’re not naive and thus not part of this “initiation” cohort.  
- Patients with other indications for biologics without RA (to ensure cohort specificity). For example, if someone has psoriasis or IBD and might start a biologic for that reason, it could confound the outcome. One might exclude patients who have no RA code but have other autoimmune conditions – however, since inclusion requires RA diagnosis, this is inherently handled.  
- If focusing strictly on RA-driven use, perhaps exclude those with active cancer or infection that would contraindicate biologics, but this isn’t usually necessary in cohort definition.

**Target Event Definition:**  
- **Initiation of a biologic DMARD therapy** for RA. This can be seen as the first claim for any biologic agent used in RA. Examples include:  
  - Pharmacy claim for **adalimumab (Humira)** – identified by a GPI code (e.g., GPI 6627001500 for adalimumab kit ([[DOC] Document (31.42 KB) - Value in Health](https://www.valueinhealthjournal.com/cms/10.1016/j.jval.2017.02.005/attachment/bf3ba157-8446-482e-b4c6-7bbfd9f3b563/mmc2.docx#:~:text=...%20by%20a%29%20J,J0135))).  
  - Pharmacy claim for **etanercept (Enbrel)**, **tofacitinib (Xeljanz)**, or other biologics (each with unique GPI codes).  
  - Medical claim J-codes for infused biologics: e.g., **J1745** for infliximab, **J3262** for tocilizumab, etc.  
- The outcome is met when any of these appears post-index. Essentially, any **biologic or targeted synthetic DMARD** prescription or administration after baseline. The predicted event is “started a biologic RA therapy.”

## 11. Oral Anticoagulant Initiation in Atrial Fibrillation  
**Clinical Rationale:** Atrial fibrillation (AF) greatly increases stroke risk – AF is associated with about a fivefold higher risk of ischemic stroke ([About Atrial Fibrillation | Heart Disease | CDC](https://www.cdc.gov/heart-disease/about/atrial-fibrillation.html#:~:text=AFib%20increases%20a%20person%27s%20risk,27)). Oral anticoagulants (OACs), like warfarin or direct oral anticoagulants (DOACs), substantially reduce this risk and are indicated in most AF patients with risk factors (per CHA₂DS₂-VASc criteria). However, not all eligible patients start anticoagulation due to various reasons. Predicting who will start an OAC helps identify those likely to be anticoagulated (versus those who might need intervention to start it) and has implications for stroke prevention efforts. It can also be used to understand quality of care (since starting OAC in AF is often a quality metric).  

**Inclusion Criteria:**  
- Patients with **Atrial Fibrillation or Atrial Flutter** diagnosis: ICD-10-CM **I48.x** codes on claims (e.g., I48.0, I48.91). This should be non-valvular AF primarily (valvular AF patients also need anticoagulation but often are automatically on warfarin if mechanical valves).  
- Typically age ≥18 (AF is common in older adults; one could limit to 65+ for Medicare population, but generally adult AF).  
- (Optional) Evidence of CHADS-VASc risk factors to ensure indication: e.g., concurrent hypertension, diabetes, heart failure codes. Not strictly required in definition, as even low-risk AF could start OAC, but modelers might include that context.  
- No anticoagulant usage in baseline (naïve to therapy) – because we want to predict new initiation.

**Exclusion Criteria:**  
- **Patients already on anticoagulation** prior to or at cohort start. Check pharmacy claims for warfarin (e.g., GPI class code 8320 for warfarin ([Adherence to Rivaroxaban, Dabigatran, and Apixaban for Stroke ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC5751430/#:~:text=Adherence%20to%20Rivaroxaban%2C%20Dabigatran%2C%20and,digits%20of%20GPI%20code%3A))) or DOACs (e.g., dabigatran GPI 83337030, rivaroxaban 83337060, etc. ([Adherence to Rivaroxaban, Dabigatran, and Apixaban for Stroke ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC5751430/#:~:text=Adherence%20to%20Rivaroxaban%2C%20Dabigatran%2C%20and,digits%20of%20GPI%20code%3A))). Also HCPCS codes like J0885 (injection anticoagulants) if any, though oral is main. If any OAC fill exists in baseline, exclude.  
- **Mechanical heart valve** patients (ICD-10 Z95.2) if one wanted to exclude them because they will almost certainly be on warfarin and are a different subgroup (their anticoagulation is mandatory and not “elective”). But they might have been on it already anyway.  
- Patients with bleeding disorders or contraindications might not start OAC (but we wouldn’t exclude them; they just might not have the event – unless the aim is to exclude those who *shouldn’t* ever start). Typically not excluded in cohort; instead, their presence might be features impeding initiation.

**Target Event Definition:**  
- **New initiation of an oral anticoagulant** (warfarin or DOAC) after the index. This is identified via pharmacy claim for an anticoagulant medication:  
  - Warfarin prescription fill (e.g., any NDC/GPI for warfarin sodium; GPI beginning **8320** for coumadin class).  
  - DOAC fills: e.g., **dabigatran** (Pradaxa) fill, **rivaroxaban** (Xarelto), **apixaban** (Eliquis), **edoxaban**, etc. These have distinct GPI codes (e.g., dabigatran ~8337030, rivaroxaban ~8337060) but collectively can be referenced as DOAC GPI codes ([Adherence to Rivaroxaban, Dabigatran, and Apixaban for Stroke ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC5751430/#:~:text=Adherence%20to%20Rivaroxaban%2C%20Dabigatran%2C%20and,digits%20of%20GPI%20code%3A)).  
- Alternatively, for warfarin one might see an initial prothrombin time (PT/INR) monitoring CPT, but pharmacy data is more direct.  
- The event is counted when any claim indicates an OAC supply to the patient. The cohort’s outcome is thus “started anticoagulation therapy for AF.” Typically, this would be expected soon after an AF diagnosis in guideline-adherent care.

## 12. Transition to Long-Term Opioid Therapy (Chronic Opioid Use)  
**Clinical Rationale:** Identifying patients at risk of becoming chronic opioid users is a priority in the context of the opioid epidemic. Many patients start opioids for acute pain (post-surgery or injury) and a subset continue to long-term use, which can lead to dependence, overdose, and increased healthcare utilization. Studies show about 5–9% of surgery patients develop new persistent opioid use post-operatively ([Postoperative Restrictive Opioid Protocols and Durable Changes in ...](https://jamanetwork.com/journals/jamaoncology/fullarticle/2800244#:~:text=,develop%20new%2C%20persistent%20opioid%20use)), and higher rates in some populations. Predicting this transition enables targeted opioid stewardship interventions (like early tapering or alternative pain management) to prevent chronic use.  

**Inclusion Criteria:**  
- **Opioid-naïve patients** who receive a new opioid prescription for an acute indication. Concretely:  
  - No opioid prescriptions in a clean baseline period (e.g., 6-12 months of no fills for opioid analgesics).  
  - Then an **index event that triggers opioid use**, such as a surgery or acute injury. For instance, the cohort might be defined at the point of an **acute opioid prescription fill** (GPI starting with 65 or 66 for opioid analgesics ([23/06/2015 Confidential eTable 3. GPI Codes to Identify the ...](https://www.jacionline.org/cms/10.1016/j.jaci.2015.07.046/attachment/f2d27694-16f8-4a13-aa0f-c0e575e4d6c0/mmc4.pdf#:~:text=...%20www.jacionline.org%20%20Analgesics,Beta))) associated with a surgery. Alternatively, include a requirement of a recent surgery (CPT for surgical procedures) or emergency visit for injury.  
  - Adults (age 18+) – both genders, though perhaps more focus on those in middle-age where chronic pain issues often start (but no strict age cutoff beyond excluding pediatrics possibly).  
- The first opioid prescription (e.g., post-op) is considered time zero for prediction, and they have no prior chronic opioid use.

**Exclusion Criteria:**  
- Any **opioid use in baseline** (to ensure they are opioid-naïve). For example, exclude if any fill of a Schedule II or III opioid (hydrocodone, oxycodone, morphine, etc.) in the past year.  
- **Cancer patients or palliative care** patients at baseline: These patients might appropriately be on long-term opioids for cancer pain or end-of-life care, which is a different context. Often studies exclude those with active cancer (ICD-10 Cxx codes) because chronic opioid use is expected in that setting, not aberrant.  
- Patients with known history of opioid use disorder (OUD) or maintenance therapy (e.g., methadone for addiction, ICD-10 F11.x) in baseline, since they are not truly “naïve” or would skew the predictive factors.

**Target Event Definition:**  
- **Development of chronic opioid therapy**, typically defined by continued opioid prescriptions beyond the acute period. Operationally, one can define it as:  
  - Opioid prescriptions **spanning 90 days or more** out of a 120-day period, excluding the first 30 days (to capture transition beyond normal post-acute taper). For example, at least 3 monthly opioid refills in the 3-6 months after the initial Rx.  
  - Another definition: **≥10 opioid prescription fills in a 12-month period**, or **≥120 days’ supply of opioids in 12 months**.  
  - Using claims, a common threshold is no gap >30 days in opioid supply for at least 3 months.  
- The event can be flagged when the patient meets criteria for chronic opioid use. For simplicity, one might set an outcome like “patient has opioids in ≥3 consecutive months post-surgery.” This would be detected from pharmacy claims (GPI class 65/66 for opioid analgesics) showing persistent fills. The outcome is binary: did they transition to long-term opioid therapy (yes/no within follow-up).

## 13. Ischemic Stroke (Cerebrovascular Accident) Event  
**Clinical Rationale:** Stroke is a leading cause of death and disability. Each year, about **795,000** people in the U.S. have a stroke, of which ~610,000 are first-time strokes ([Stroke Facts | Stroke | CDC](https://www.cdc.gov/stroke/data-research/facts-stats/index.html#:~:text=minutes%20and%2011%20seconds%2C%20someone,2%20billion)). It is a major driver of long-term disability – stroke is a leading cause of serious long-term disability in adults ([Stroke Facts | Stroke | CDC](https://www.cdc.gov/stroke/data-research/facts-stats/index.html#:~:text=to%20treat%20stroke%2C%20and%20missed,2)). Predicting an ischemic stroke event in high-risk patients (e.g., those with hypertension, AFib, etc.) allows for proactive measures like stricter blood pressure control, anticoagulation in AFib, or smoking cessation support. Even a moderate risk model could be used to prioritize patients for stroke prevention programs.  

**Inclusion Criteria:**  
- Adults (typically **age 50+** or 60+, since risk rises with age) with **stroke risk factors** but no prior stroke. Key risk factors to look for in baseline:  
  - **Hypertension (ICD-10 I10)** – the most prevalent stroke risk factor.  
  - **Atrial fibrillation (I48.x)** – major risk (could be included, though if they’re anticoagulated maybe risk is lower).  
  - **Diabetes (E11.x)**, **Hyperlipidemia (E78.5)**, **Smoking-related diagnoses (F17.xx nicotine dependence)**, **Peripheral arterial disease (I73.9)**, etc.  
  - **History of TIA** (transient ischemic attack, ICD-10 Z86.73 or G45.9) could be included as a risk factor, though one might exclude those as they’ve essentially had a warning stroke event.  
- To be conservative, require presence of at least one major risk factor (e.g., hypertension or AF or diabetes) in the inclusion. This ensures the cohort is truly “at risk” rather than general population.  
- No history of stroke or intracranial hemorrhage prior (see exclusion).

**Exclusion Criteria:**  
- **Prior stroke or TIA** before the index date: any ICD-10 code for ischemic stroke (I63.x), hemorrhagic stroke (I61.x, I60.x), or sequelae of stroke (I69.x), or TIA (G45.x) in baseline. We want first-ever stroke occurrences.  
- If the focus is on ischemic stroke, one might exclude patients with conditions predisposing specifically to hemorrhagic stroke (like aneurysm or AVM) since that’s a different mechanism – but usually not necessary as hemorrhagic strokes can be filtered at outcome level.  
- Patients on full anticoagulation in baseline (e.g., for AF) might have lower risk; we don’t exclude them, but the model should learn that as a protective factor perhaps. So exclusions mainly revolve around already having had a stroke.

**Target Event Definition:**  
- **Acute ischemic stroke event**, identified by a hospitalization or emergency encounter with stroke diagnosis. For instance:  
  - Inpatient claim with **ICD-10 I63.x** (cerebral infarction) as the principal diagnosis (or first-listed diagnosis in outpatient if it occurs outpatient, though strokes typically result in hospital care).  
  - One could also use ICD-10 **I64** (stroke, not specified hemorrhagic or ischemic) to capture any stroke if not specified, though most coding distinguishes them.  
  - We specifically focus on ischemic, so I63.* codes are primary.  
- Often, stroke identification in claims uses the principal inpatient diagnosis plus perhaps a length-of-stay >1 day (to exclude rule-outs). But for cohort purposes, the first occurrence of an I63 code in an acute care setting is the outcome.  
- The model’s aim is to predict this new stroke event. For example, a patient who in 2024 has an admission with **I63.411 (cerebral infarction due to embolism of cerebral artery)** would be flagged as outcome=Yes (provided no stroke codes prior).

## 14. Acute Myocardial Infarction (Heart Attack)  
**Clinical Rationale:** Myocardial infarction (MI) is a leading cause of mortality and often the first manifestation of coronary disease in some individuals. In the U.S., about **805,000** people have a heart attack each year, and **605,000 are first-time** MIs ([Heart Disease Facts - CDC](https://www.cdc.gov/heart-disease/data-research/facts-stats/index.html#:~:text=Every%20year%2C%20about%20805%2C000%20people,attack%2C%20and%20200%2C000%20happen)). Predicting a first MI in at-risk patients (those with CAD risk factors) can prompt intensive preventive measures (statins, aspirin, risk factor control) to hopefully avert the event. It also helps healthcare systems identify who might benefit from closer cardiac monitoring.  

**Inclusion Criteria:**  
- Adults (typically **age >40** for men, >50 for women, or any adult with significant risk factors) with **cardiovascular risk factors or existing atherosclerotic disease** but **no prior MI**. Examples:  
  - **Coronary artery disease without prior MI**: ICD-10 I25.10 (CAD without MI) or a history of angina (I20.x).  
  - **Peripheral artery or carotid disease** (I73.9, I65.x) indicating atherosclerosis elsewhere.  
  - **Diabetes mellitus** (E11.x), which markedly increases MI risk.  
  - **Hypertension** (I10) and **hyperlipidemia** (E78.5) – common risk factors.  
  - **Tobacco use** (F17.x).  
- One might operationalize inclusion as: patients with a **QRISK/Framingham risk factor profile** in claims: e.g., diagnose codes for at least 2 of the following – HTN, dyslipidemia, diabetes – or one atherosclerotic disease code. Essentially ensuring they are not low-risk.  
- No previous MI coded (see exclusion).

**Exclusion Criteria:**  
- **History of Myocardial Infarction** before index: any ICD-10 I21.x (acute MI codes) or I22.x (subsequent MI) in baseline, or an old MI code I25.2. These indicate they already had a heart attack.  
- Patients with **history of coronary interventions** without specifying MI: this is tricky; if someone had a stent (PCI) but no documented MI, they might still be at risk of MI. We *include* such patients (they have CAD). Only exclude if explicitly prior MI.  
- If focusing on *first ever* coronary event, exclude those with coronary artery bypass history as well (since they presumably had significant CAD, possibly MI). But they could still have an MI after CABG, though that usually implies they already had events. It might simplify to exclude anyone with prior CABG or PCI, depending on whether we want totally naive or just first MI. This can vary by design.

**Target Event Definition:**  
- **Acute myocardial infarction** occurrence, typically identified by an inpatient claim with an MI diagnosis code. For example:  
  - ICD-10-CM **I21.xx** codes (ST-elevation MI and Non-ST-elevation MI codes). E.g., I21.4 (NSTEMI), I21.0 (anteroseptal STEMI), etc., appearing as the admitting or primary diagnosis in a hospitalization.  
  - Could also include I22.x (subsequent MI) if someone had one MI and then another within the observation window (though if truly first-ever, I22 wouldn’t be the first one, it would be I21 then I22; usually stick to I21 for first events).  
- The event is counted on the date of hospital admission for the heart attack. Often a procedure follows (like stent or thrombolysis), but the diagnosis code itself is sufficient.  
- The outcome being predicted: **yes/no did the patient experience an acute MI** within the prediction horizon.

## 15. Hip Fracture in Older Adults with Osteoporosis  
**Clinical Rationale:** Hip fractures are devastating events for older adults, often resulting from osteoporosis and falls. They carry high mortality (estimates of one-year mortality range from 18% to 33% in seniors ([How Do Older Adults Fare After Hip Fracture? > Health in Aging Blog > Health in Aging](https://www.healthinaging.org/blog/how-do-older-adults-fare-after-hip-fracture/#:~:text=Hip%20fractures%20in%20older%20adults,live%20in%20a%20nursing%20home))) and significant morbidity – up to 50% of survivors have permanent loss of mobility or independence, and ~20% require long-term nursing care ([How Do Older Adults Fare After Hip Fracture? > Health in Aging Blog > Health in Aging](https://www.healthinaging.org/blog/how-do-older-adults-fare-after-hip-fracture/#:~:text=As%20many%20as%2050%20percent,after%20having%20a%20hip%20fracture)). Predicting hip fractures in those at risk (e.g., osteoporotic individuals) is valuable to intensify fall prevention strategies or medication to strengthen bone density. It also helps in anticipating healthcare needs like surgical capacity and post-acute rehab.  

**Inclusion Criteria:**  
- **Older adults**, typically **age ≥65** (or ≥70) – population at risk for osteoporotic fractures. Could include postmenopausal women ≥50 as well, but most models focus on elderly.  
- Evidence of **osteoporosis or osteopenia** in claims: ICD-10 **M81.x (osteoporosis)** without current fracture, or **M80.x (osteoporosis with fractures)** if they had a prior fracture of another site. Alternatively, a history of fragility fractures (e.g., wrist fracture) could qualify.  
- Alternatively, use pharmacy data: long-term use of osteoporosis medications (GPI for bisphosphonates like alendronate) or chronic glucocorticoid use (which predisposes to osteoporosis) as inclusion signals.  
- No prior hip fracture in baseline (we want those who haven’t had the event yet).

**Exclusion Criteria:**  
- **History of hip fracture or hip replacement** prior to index. Identified by prior ICD-10 codes for femur/hip fracture (e.g., S72.0, S72.1 series) or procedure codes for hip repair/replacement (CPT 27236 for hip fracture repair, or 27130 for total hip replacement). If they already broke a hip, they’re not “prediction” for first event (though could fracture the other hip, but typically we consider first hip fracture).  
- Patients with conditions causing pathological fractures (e.g., metastatic cancer to bone, ICD-10 C79.51) may be excluded if the interest is specifically osteoporotic fractures from low trauma. Pathologic fractures are a different mechanism.  
- If focusing on community-dwelling elderly, might exclude those already in nursing homes (though hard to see in claims) because their risk/profiles differ, but this is optional.

**Target Event Definition:**  
- **Hip fracture event**, usually operationalized by a hospitalization for a hip fracture. Key identifiers:  
  - ICD-10-CM codes **S72.0xx, S72.1xx, S72.2xx** which cover fractures of the femoral neck, intertrochanteric region, or subtrochanteric femur (all commonly termed “hip fractures”). For example, S72.001A (fracture of neck of femur, initial encounter).  
  - This should appear in an acute setting (ER or inpatient claim). Often accompanied by surgical repair codes: e.g., CPT **27236** (open treatment of femoral fracture) or **27130** if they did a hip replacement due to fracture. One could require a procedure or inpatient stay to confirm it’s a serious fracture.  
- The outcome is marked when such a fracture code (plus appropriate treatment code) occurs. In many analyses, they identify hip fractures by diagnosis + procedure within a short window. But for simplicity, any claim indicating a hip fracture diagnosis in an acute care context would count. The model would predict the **occurrence of a hip fracture**.

## 16. Hospitalization for COPD Exacerbation (Severe Acute Exacerbation of COPD)  
**Clinical Rationale:** Chronic Obstructive Pulmonary Disease (COPD) exacerbations are episodes where symptoms acutely worsen, often requiring additional treatment or hospitalization. Severe exacerbations (those requiring hospital admission) are linked to significant mortality (around 9% 30-day mortality post-hospitalization on average ([American Journal of Respiratory and Critical Care Medicine](https://www.atsjournals.org/doi/full/10.1164/rccm.201712-2612ED#:~:text=Medicine%20www,%282%29.%20Multiple%20factors))) and faster lung function decline. Predicting which COPD patients will have a severe exacerbation can enable proactive measures (e.g., ensuring medication adherence, prophylactic therapies, early intervention plans at symptom onset). It’s also vital for managing healthcare utilization, as COPD exacerbation admissions are costly and frequent in advanced disease.  

**Inclusion Criteria:**  
- Patients with **diagnosed COPD** in their medical claims. Typically:  
  - ICD-10-CM **J44.0, J44.1** (COPD with exacerbation, with infection) are used for acute events, but for baseline identification use **J44.9 (COPD, unspecified)** or **J43.x** (emphysema) in history.  
  - Chronic bronchitis codes (J41, J42) could also be included if representing COPD.  
- Middle-aged to older adults (usually **age ≥40** with a history of smoking, implicitly, since COPD rarely occurs in never-smokers; but age isn’t a strict filter, just most will be older).  
- Possibly evidence of prior milder exacerbations or maintenance therapy: e.g., prescriptions for inhalers (GPI class — long-acting beta agonists, anticholinergics, inhaled steroids) in baseline, indicating they are being treated for COPD. This ensures they truly have COPD and are at risk.  
- No hospitalization for COPD in a certain recent period if we want to predict first hospitalization, or we could allow history of prior exacerbations to predict future ones (depending on whether it’s first-ever or recurrent event prediction). Often, prior exacerbations are a risk factor, so one might include those with prior exacerbation history to predict the next.

**Exclusion Criteria:**  
- Patients with **active asthma** without COPD (J45.x) – if we want a pure COPD cohort, exclude those who seem to have asthma rather than COPD. (Though overlap exists, so some may have both; one could include chronic bronchitis and exclude if asthma is the only pulmonary diagnosis.)  
- If the goal is first severe exacerbation, exclude those who already had a COPD hospitalization in baseline. If the goal is any exacerbation, that exclusion isn’t needed; rather, past exacerbation will be a predictor feature.  
- Patients with end-stage lung diseases such as **pulmonary fibrosis or lung cancer** might be excluded if focusing on typical COPD, but not strictly necessary. Mainly ensure the respiratory diagnosis is indeed COPD.

**Target Event Definition:**  
- **Severe COPD exacerbation requiring hospitalization.** In claims terms, this can be defined as an **inpatient admission with a principal diagnosis of COPD exacerbation** or respiratory failure due to COPD. For example:  
  - ICD-10 **J44.1** (COPD with acute exacerbation) as the primary diagnosis on an admission claim.  
  - Could also capture if the primary diagnosis is acute respiratory failure (J96.0x) with a secondary diagnosis of COPD, which often indicates a severe flare needing ICU.  
  - Another indicator: use of mechanical ventilation (CPT 94002 or ICD-10-PCS codes for ventilation) during an admission with COPD, but the diagnosis code approach is simpler.  
- So the event is triggered by the presence of a hospitalization record coded for COPD exacerbation. The model would predict the binary outcome: **hospitalized for a COPD exacerbation (yes/no)** in the prediction period. 

## 17. New Onset Major Depressive Disorder (MDD)  
**Clinical Rationale:** Major depressive disorder is one of the most common mental illnesses and is a leading cause of disability worldwide ([
	"Depression: let’s talk" says WHO, as depression tops list of causes of ill health
](https://www.who.int/news/item/30-03-2017--depression-let-s-talk-says-who-as-depression-tops-list-of-causes-of-ill-health#:~:text=Reading%20time%3A)). Early identification of individuals likely to develop depression can facilitate prompt intervention (counseling, stress support, or even preventive use of antidepressants in prodromal phases). Untreated MDD leads to functional impairment, impacts chronic disease outcomes, and increases risk of suicide. From a healthcare perspective, predicting new MDD can also help manage resource allocation for mental health services.  

**Inclusion Criteria:**  
- **Adults (age ≥18)** with no history of depression who have risk factors or prodromal symptoms. Some possible inclusion signals:  
  - Chronic illness or chronic pain conditions (which often predispose to depression) – e.g., ICD-10 codes for arthritis, diabetes, etc., plus perhaps sleep disturbance (G47.00) or fatigue (R53.83).  
  - Recent significant life stressors coded (though not often coded in claims) or diagnoses like adjustment disorder (F43.2) that could progress to MDD.  
  - Use of minor tranquilizers or help-seeking behavior: e.g., a new prescription for a low-dose anxiolytic, or increased primary care visits with somatic complaints.  
- More straightforward: **no prior depression diagnoses** in history and not on antidepressants, but may have other **mental health diagnoses** like anxiety (F41.x) or insomnia that often coexist or precede depression. These patients would be at higher risk for developing MDD.  
- Possibly limit to those with continuous enrollment ≥1 year baseline to reliably screen out prior depression.

**Exclusion Criteria:**  
- **Any prior diagnosis of major depression** or other depressive disorders in baseline (ICD-10 F32.x single episode MDD, F33.x recurrent MDD, F34.1 dysthymia, etc.). We want the first onset.  
- **Bipolar disorder (ICD-10 F31.x)** or **psychotic disorders (F20-F29)** in baseline, because depressive episodes in those contexts are different (bipolar depression) and treatment approach differs. Focusing on unipolar MDD, one would exclude those conditions.  
- Patients already on **antidepressant medications** in baseline (GPI codes 5816 for SSRIs, 5818 for SNRIs, etc.) because that suggests they have depression or another condition being treated (unless it’s off-label, but generally exclude to ensure this is new treatment-naïve depression).  
- Situational exclusions: if one wanted to exclude brief adjustment disorders that resolved, but generally prior diagnosis exclusion covers it.

**Target Event Definition:**  
- **New diagnosis of Major Depressive Disorder** on a claim, meeting a certain threshold. In practice, to increase specificity, one might define it as at least **two outpatient visits** with an MDD diagnosis (ICD-10 F32.x or F33.x), or **one psychiatric inpatient claim** with MDD, within a certain time frame. This avoids counting a single “rule-out” diagnosis.  
  - For example: an outcome could be “at least two claims on different dates coded with **F32.1 (moderate MDD)** or **F33.0 (recurrent MDD mild)**, etc., within 6 months,” or one claim with an antidepressant start.  
- In a simpler approach, any occurrence of an **ICD-10-CM F32.* or F33.* code** in follow-up (with no prior) can be considered the event of new depression diagnosis. If pharmacy data is used, initiation of an antidepressant (GPI class 5816, 5818, etc., not used in baseline) could also serve as a proxy for incident depression treatment.  
- Thus, the model predicts whether the patient will be newly diagnosed with MDD (meeting criteria above) in the specified future period.

## 18. New Onset Psychotic Disorder (e.g. First-Episode Schizophrenia)  
**Clinical Rationale:** Schizophrenia and other primary psychotic disorders typically onset in late adolescence or early adulthood. They cause significant disability in social and occupational functioning ([
	Schizophrenia
](https://www.who.int/news-room/fact-sheets/detail/schizophrenia#:~:text=,be%20able%20to%20fully%20recover)) and have a high cost of care. Identifying individuals at risk of developing a psychotic disorder is challenging, but if possible (using proxy markers in claims), it could facilitate early intervention (like early psychosis programs) which can improve long-term outcomes. This cohort focuses on predicting the first occurrence of a chronic psychotic disorder (such as schizophrenia or schizoaffective disorder).  

**Inclusion Criteria:**  
- **Youths or young adults** in the typical risk window for first psychosis, e.g., age **16-30** (one could widen to 16-40 to catch later onsets).  
- No prior psychotic disorder diagnosis (obviously).  
- Possible inclusion signals (since true prodrome is hard to see in claims, we use proxies):  
  - Prior diagnoses of things like **brief reactive psychosis (ICD-10 F23)** or **schizotypal or delusional disorder** might indicate emerging illness. However, these might already be considered psychotic disorders.  
  - More commonly, look at **heavy utilization of mental health services** or diagnoses of severe mood disorders: e.g., a young person with multiple ICD-10 F32.x depression or F41.x anxiety diagnoses, who might later be found to have underlying schizoaffective or schizophrenia when psychosis appears.  
  - **Substance abuse** in a young person (especially cannabis or stimulants – ICD-10 F12.x, F14.x) as risk factors (since substance-induced psychosis or triggering latent schizophrenia is possible).  
- Essentially, an at-risk cohort might be “patients age 16-35 with mental health treatment history (like depression/anxiety/OCD/PTSD) or substance abuse, but no known psychotic or bipolar disorder.” This captures a population where first psychosis might emerge.

**Exclusion Criteria:**  
- **Existing psychotic disorder diagnoses** in baseline: ICD-10 F20.x (schizophrenia), F25.x (schizoaffective), F22 (delusional disorder), F28/F29 (unspecified psychosis) – none of these should appear prior to index.  
- **Bipolar I disorder** with psychotic features (F31.x with psychosis) in baseline – sometimes bipolar can have psychosis; if they have bipolar already, then a psychotic episode could be part of that, not a new primary schizophrenia. One might exclude bipolar entirely to focus on primary psychosis from schizophrenia spectrum.  
- Prior use of **antipsychotic medications** could be an exclusion (if they were already on antipsychotics, they likely had some form of psychosis or bipolar). Check pharmacy: GPI 5817 (antipsychotic class) in baseline – if present, exclude, as that suggests treated psychosis already.  
- Neurological conditions that could mimic psychosis (e.g., epilepsy) need not be excluded necessarily; they would not cause a schizophrenia diagnosis typically.

**Target Event Definition:**  
- **First diagnosis of a chronic psychotic disorder**, particularly schizophrenia or similar, in the follow-up. For example:  
  - An inpatient psychiatric admission with **ICD-10 F20.0 (paranoid schizophrenia)** or **F25.0 (schizoaffective disorder, depressive type)**, etc.  
  - Or two outpatient psychiatry visits with a schizophrenia spectrum diagnosis.  
- Essentially, the outcome can be “new schizophrenia diagnosis (yes/no).” A stringent approach: the event is counted if the patient receives an ICD-10 diagnosis in the F20-F29 range (excluding F30-F39 mood disorders) on at least 2 separate service dates, or one hospitalization with it. That signifies the provider is establishing a chronic psychotic disorder.  
- Examples of codes: **F20.9 (schizophrenia, unspecified)**, **F25.1 (schizoaffective, bipolar type)**, **F22 (delusional disorder)** would all qualify as the outcome. The first such code in time would mark the incident date. The predictive model would aim to flag those who will get one of these diagnoses.

## 19. Adherence to Screening Mammography (Breast Cancer Screening)  
**Clinical Rationale:** Breast cancer screening with mammography has been shown to reduce breast cancer mortality by enabling early detection. Women who undergo regular mammograms have significantly lower risk of advanced disease; for instance, participating in screening is associated with a ~41% reduction in breast cancer mortality over 10 years ([
            Mammography screening reduces rates of advanced and fatal breast cancers: Results in 549,091 women - PMC
        ](https://pmc.ncbi.nlm.nih.gov/articles/PMC7318598/#:~:text=Women%20who%20participated%20in%20mammography,CI%2C%200.66%E2%80%900.84%20%5BP%C2%A0%3C%C2%A0.001)). Ensuring women adhere to screening guidelines (typically every 1-2 years in mid-life) is a public health priority. This cohort prediction is slightly different: instead of a adverse event, it predicts a **preventive care event (completion of mammography)**. This can be useful to identify women less likely to get screened so interventions can be made to improve compliance (thus indirectly reducing future cancer burden).  

**Inclusion Criteria:**  
- **Women in the recommended age range for breast cancer screening.** Commonly **50–74 years** (per many guidelines), though some guidelines now say start at 45 or 40. We can define 50-74 for classic cohort.  
- Continuous enrollment in the health plan for the past 2+ years (so we can observe screening history and ensure they are eligible).  
- **No history of breast cancer** (ICD-10 C50.x or Z85.3 personal history of breast ca) and **no mastectomy** (which would remove need for screening on that side). Essentially, they should be average-risk screening candidates.  
- Also, we might specifically target those who are **due for a mammogram**: e.g., no mammogram in the past 2 years (so they are not up-to-date). That sets the stage to predict who will go on to get one. Alternatively, the cohort could include all women in age range (some will get screened, some not) and the model predicts the event of screening. But focusing on those due might make sense to exclude those already recently screened.

**Exclusion Criteria:**  
- **Previous breast cancer or mastectomy** as noted. If a woman had bilateral mastectomy, screening mammography is no longer indicated – so exclude if ICD-10 Z90.13 (acquired absence of breasts) or similar appears.  
- Women under active surveillance for breast conditions (might be getting diagnostic mammograms more frequently) – not necessarily exclude, but if they have recent mammograms for diagnostic reasons, they may not be “due for screening” in the same way. However, that complicates things; primary exclusions are cancer history.  
- If the cohort is specifically those non-compliant with screening, one might exclude women who already got a mammogram very recently (because they already did the event, though they could do another in next cycle but likely not within a year).

**Target Event Definition:**  
- **Completion of a screening mammogram** within the follow-up period. This can be identified by billing codes specific to screening mammography (as opposed to diagnostic). For example:  
  - **CPT 77067** – Screening mammography, bilateral (2-view study of each breast). (Older codes: 77057 was used before 2018, and G0202 for Medicare).  
  - HCPCS **G0202** is a code for screening mammography for Medicare patients. **G0202** or **G0204/G0206** might appear depending on 2D/3D mammography and unilateral vs bilateral.  
  - The presence of any of these “screening mammo” codes on a claim (typically outpatient radiology) indicates the patient underwent the preventive screening.  
- The model’s positive outcome is **“received a screening mammogram in year X”**. Essentially, it’s predicting adherence to breast cancer screening guidelines. For example, if predicting in a 1-year window, a woman who has CPT 77067 in that year is outcome=Yes.

## 20. Completion of Screening Colonoscopy for Colorectal Cancer  
**Clinical Rationale:** Colonoscopy screening can both detect early cancer and prevent cancer by removing precancerous polyps. It’s recommended for average-risk adults (historically starting at age 50, now often 45). Adherence is suboptimal in many populations. Screening colonoscopy has been associated with up to a 67–88% reduction in risk of death from colorectal cancer in observational studies ([How well do colonoscopies prevent colorectal cancer? What you need to know - Harvard Health](https://www.health.harvard.edu/blog/how-well-do-colonoscopies-prevent-colorectal-cancer-what-you-need-to-know-202210182834#:~:text=Past%20research%20shows%20that%20colonoscopy,combed%20through%20the%20study%20carefully)). Predicting who will go for a screening colonoscopy helps identify those who likely won’t (so outreach can be targeted to them). It is also useful for healthcare systems to project demand for endoscopy services.  

**Inclusion Criteria:**  
- **Adults in the target age range for CRC screening**, e.g., **50–75 years** (some guidelines 45–75; we can use 50+ as a common threshold in claims cohorts historically). Both men and women.  
- Continuous enrollment for baseline period (to assess prior screening status).  
- **No history of colorectal cancer** (ICD-10 C18-C20, Z85.038 history of colon cancer) and no diagnosis of high-risk colonic conditions that would change screening interval (like inflammatory bowel disease with primary sclerosing cholangitis, familial polyposis, etc., though those are relatively rare – could exclude if focusing strictly on average-risk screening).  
- Ideally, **due for screening**: e.g., no colonoscopy in the last 10 years, no stool test in last year (if those are captured by CPT 82270 FOBT, etc.). If prior colonoscopy was 9 years ago, they’re due soon. But we can simplify by including those with no evidence of prior colonoscopy in claims history (or at least not in the last 5-10 years).

**Exclusion Criteria:**  
- **Colonoscopy for diagnostic or surveillance purposes** already in baseline. We might exclude those who had a recent colonoscopy (last few years) because they aren’t due for a screening now. If someone had colonoscopy 2 years ago due to symptoms, they might not count as “screening” need now. Possibly exclude anyone with any colonoscopy in past 5 years to focus on those truly due.  
- Those with **colon cancer history or colorectal resection** (colectomy CPT 44150 etc.) as noted, since routine screening doesn’t apply.  
- Patients with limited life expectancy (metastatic cancer, etc.) might be excluded in a real program since screening isn’t recommended if life expectancy <10 years, but that’s a bit beyond claims-based scope. Mainly exclude prior CRC or recent colonoscopy.

**Target Event Definition:**  
- **Completion of a screening colonoscopy** in the follow-up. This can be identified by specific CPT/HCPCS codes:  
  - **CPT 45378** is a base code for diagnostic colonoscopy. For screening, Medicare uses HCPCS **G0121** (screening colonoscopy for average risk) and **G0105** (screening for high-risk). In commercial claims, often a screening colonoscopy is still reported with 45378 with a screening diagnosis (Z12.11 - encounter for screening colonoscopy).  
  - One approach: outcome is met if there’s any colonoscopy CPT code **45378–45385** *in conjunction with* a screening diagnosis code (Z12.11 or Z12.12). But many datasets have a flag for preventive vs diagnostic.  
  - Simpler: any claim with **HCPCS G0121** (screening colonoscopy, average risk) definitively means a screening colonoscopy was done. This is a clear indicator in Medicare claims.  
- Therefore, the event = **Yes** if the patient undergoes a colonoscopy identified as screening during the follow-up window. The prediction is binary: did the patient complete a screening colonoscopy or not. (If they instead got a stool test, that’s not the target here; we specifically look for colonoscopy completion.)  

Each of these cohort definitions uses claims-based criteria (ICD-10 for diagnoses, CPT/HCPCS for procedures, GPI for medications, plus age/gender) to identify a population and a clear outcome. These cohorts span a wide range of clinical areas, from chronic disease complications and acute events to surgeries, new medication treatments, mental health, and preventive care, illustrating the diversity of predictive modeling targets in healthcare data. Each outcome is defined in a way that is detectable in longitudinal claims with reasonable accuracy, making them suitable for modeling.