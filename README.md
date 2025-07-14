Table will be:


	Direct Forget ESR		Paraphrase Forget ESR           OneHop			    Inverted	                Model Utility     Fluency Forget

	
       Base | PerMUtok | PerMU	       Base | PerMUtok | PerMU     Base | PerMUtok | PerMU   Base | PerMUtok | PerMU    Base | PerMUtok | PerMU Base | PerMUtok | PerMU

1.5B
7B
14B
32B









Results :

## Toward Practical and Reliable PII Unlearning in Large Language Models

**Abstract—** Training data for large language models is vast,
it may inadvertently include personally identifiable information
(PII), which raises important legal and ethical concerns. Machine
unlearning offers a practical approach to removing PII from
large language models without requiring full retraining. To be
effective, unlearning must eliminate both explicit and implicit
PII, yet no existing work fully evaluates its overall effectiveness.
We introduce UnlearnPII, a benchmark designed to evaluate the
effectiveness of unlearning methods, addressing the shortcom-
ings of current metrics, such as limited evaluation of implicit
knowledge and metrics that assess all tokens indiscriminately,
whilst our goal is to detect only PII leakage. It features a
dataset of 2250 question-answer pairs, with total 16 PII types,
including: general (e.g., date of birth, email), medical (e.g.,
diagnoses, treatments), and financial information (e.g., bank
accounts, credit card numbers). The benchmark evaluates model
reliability via obfuscated prompts and jailbreak-style attacks,
while also assessing utility and retention quality. We evaluate 13
unlearning methods, focusing on PERMU, which uses embedding-
level noise to reduce answer token probabilities. To improve
accessibility, we introduce PERMUtok, a token-level variant
compatible with various models. Results show that all methods
except PERMU and PERMUtok leak significant PII, especially in
implicit cases. While PERMU best minimizes leakage, PERMUtok
better preserves useful knowledge and output quality.



THe thesis aims to answer the following research questions :


1. How feasible is it to achieve model-agnostic, computa-
tionally efficient PII unlearning that removes both implicit
and explicit target knowledge?
2. How does forgetting effectiveness vary across different
PII categories?
3. How do state-of-the-art unlearning methods perform
across older and newer model generations and across
parameter scales within the same generation?

<img width="512" height="520" alt="image" src="https://github.com/user-attachments/assets/f5327bf3-e59c-425a-bf04-94be7960638b" />



### NOTE : This repository is still under construction. The code will be commented, cleaned and merged. Additionally, all the necessary acknowledgements and references to work from which helped our implementation will be provided.


