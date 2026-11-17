"""Handwritten behavior probes, never real labeled accuracy/training examples."""

EQUIPMENT = ('Before the employment agreement is finalized, new team members complete workstation provisioning through our designated supply portal. '
             'The standard equipment package costs $290 and is initially covered by the candidate. '
             'This amount is included in the first payroll reimbursement after successful onboarding. '
             'To maintain a consistent security configuration, equipment purchased elsewhere cannot be used.')
FREE_EQUIPMENT = ('Before the employment agreement is finalized, new team members complete workstation provisioning through our designated supply portal. '
                  'The standard equipment package is provided by the employer at no cost to the candidate. '
                  'No payment is required from the candidate before or after onboarding. '
                  'To maintain a consistent security configuration, equipment purchased elsewhere cannot be used.')
BASE = {'title':'Junior Operations Coordinator',
        'company_profile':'Our operations team supports scheduling, documentation and communication across departments.',
        'description':'Coordinate schedules, maintain accurate records and support internal teams. This is an entry-level operations role with training and regular feedback.',
        'benefits':'Flexible working arrangements, paid leave and professional development.'}

def case(id,text,expected,**extra):
    return {'id':id,'source_type':'handwritten_diagnostic','label_status':'behavior_expectation_not_fraud_ground_truth',
            'use':'diagnostic_only_never_training_or_threshold_selection','fields':{'description':text,**extra},'expected_signals':expected}

CASES=[
 {'id':'equipment_paid','source_type':'handwritten_diagnostic_with_verbatim_user_passage','label_status':'review_behavior_not_verified_fraud','use':'diagnostic_only','fields':{**BASE,'requirements':EQUIPMENT},'expected_signals':['candidate_funded_equipment']},
 {'id':'equipment_free','source_type':'handwritten_diagnostic_control','label_status':'review_behavior_not_verified_legitimacy','use':'diagnostic_only','fields':{**BASE,'requirements':FREE_EQUIPMENT},'expected_signals':[]},
 case('equipment_paraphrase','Applicants must buy a laptop through our approved vendor before signing a contract. Documented costs will be reimbursed after their first working month.',['candidate_funded_equipment']),
 case('equipment_negated','The candidate is not required to purchase equipment. The company provides a laptop. No payment is required.',[]),
 case('legitimate_reimbursement','After signing the contract, employees may choose any supplier for approved office equipment. Submit receipts under the written reimbursement policy. No advance payment is required to accept the job.',[]),
 case('legitimate_travel','Travel expenses incurred on company business are reimbursed after an approved expense report. The employer supplies a corporate card.',[]),
 case('personal_transfers','Receive client payments in your personal bank account and forward the funds to the supplier. Retain a commission.',['personal_account_transfers']),
 case('personal_transfers_negated','Never receive client payments in your personal bank account or forward the funds. Use company-controlled accounts for all customer payments.',[]),
 case('business_payments','Process customer payments using company bank accounts. Reconcile invoices and forward payment records to finance. Salary is paid into your personal bank account.',[]),
 case('ordinary_courier','Deliver documented grocery orders to customers using the dispatch app. Packages are sealed for customer privacy. Follow identification, safety and proof-of-delivery procedures.',[]),
 case('document_courier','Courier required to deliver documents between offices. Contents are listed on dispatch records. Sealed confidential envelopes must not be opened; obtain delivery signatures.',[]),
 case('unclear_contents_only','Deliver sealed parcels with undisclosed contents. Follow normal dispatch and safety procedures.',[]),
 case('evasive_delivery','Deliver packages with undisclosed contents. Avoid police checks and remove shipping labels.',['opaque_delivery_procedure']),
 case('evasion_negated','Deliver packages with undisclosed contents. Never avoid police checks or remove shipping labels.',[]),
 case('remote_entry','Entry-level remote administrative assistant. No experience required. Employer-provided equipment, paid training and a written employment agreement. No payment is required.',[]),
 case('high_salary','Courier work with flexible hours and high earnings. No experience required. Deliver documented orders and obtain proof of delivery.',[]),
 case('warning_article','Scam warning: scammers may ask you to receive funds in your personal bank account and forward the money. Do not accept job offers like this.',[]),
 case('later_changed','Deliver documents between offices. Maintain a documented chain of custody.',['personal_account_transfers','instructions_changed'],later_instructions='Instead of the advertised duties, receive client money in your personal bank account and forward the funds to another account.'),
]
