"""Conservative, separate text heuristics. These are not learned model explanations.

Evidence offsets reference each field's HTML-stripped, whitespace-normalized text.
No category, earnings level, remote status, or lack of experience alone is a trigger.
"""
import re
from shared.text import FIELDS, clean_text

REVIEW_VERSION = 'review-v1.2'
POLICY_VERSION = 'review-policy-v1'
POLICY = ('Additional review is recommended if the selected learned model crosses its own threshold, '
          'either primary model crosses its own threshold, or a contextual heuristic fires. '
          'Scores are never averaged. No warning means only that these checks found none; the employer is not verified.')
SOURCES = {
    'equipment': 'https://consumer.ftc.gov/consumer-alerts/2023/01/looking-job-scammers-might-be-looking-you',
    'money': 'https://www.fbi.gov/how-we-can-help-you/common-frauds-and-scams/money-mules',
}

# Explicit role/payment relations, rather than the mere presence of "equipment".
PATTERNS = {
    'equipment': r'\b(?:equipment|workstation|laptop|computer|supplies|supply portal|starter kit)\b',
    'candidate_payment': r'\b(?:(?:candidates?|applicants?|employees?|you|new (?:team )?members?)\s+(?:must\s+|will\s+|are required to\s+|initially\s+)?(?:pay|purchase|buy|cover|fund)|(?:covered|paid|funded)\s+(?:initially\s+)?by\s+(?:the\s+)?(?:candidate|applicant|employee)|(?:at|from|using|with)\s+(?:your|their)\s+(?:own\s+)?(?:expense|funds|money)|out.of.pocket|up.front (?:payment|purchase)|pay\s+(?:an?\s+)?(?:equipment|workstation|onboarding|supply)\s+(?:fee|deposit|cost))\b',
    'preagreement': r'\b(?:before|prior to|until)\b.{0,90}\b(?:agreement|contract|employment|offer|onboarding|start date)\b|\b(?:before|prior to)\b.{0,35}\b(?:sign|start|finaliz|finalis)',
    'restricted_vendor': r'\b(?:designated|approved|exclusive|specified|selected|our|authorized)\b.{0,45}\b(?:portal|vendor|supplier|store)\b|\b(?:purchased|bought)\s+elsewhere\s+(?:cannot|can.not|may not|must not)\b|\bonly\b.{0,30}\b(?:portal|vendor|supplier)\b',
    'reimbursement': r'\b(?:reimburse\w*|refund\w*|paid back|pay\w* back)\b',
    'personal_account': r'\b(?:your|their|own|employee.s|candidate.s)\s+(?:own\s+)?(?:personal\s+)?(?:bank\s+|checking\s+|savings\s+)?account\b|\bpersonal\s+(?:bank\s+|checking\s+|savings\s+)?account\b',
    'receive_funds': r'\b(?:receive|receiving|accept|collect|deposit|credit)\w*\b.{0,65}\b(?:funds|money|payments|transfers|cheques|checks|client funds|customer payments)\b|\b(?:funds|money|payments)\b.{0,65}\b(?:deposited|received|credited)\b',
    'forward_funds': r'\b(?:forward|transfer|wire|send|remit|withdraw|convert)\w*\b.{0,70}\b(?:money|funds|payments|balance|remainder|bitcoin|crypto|another account|overseas|them|it)\b|\b(?:forward|wire|remit)\s+(?:the\s+)?(?:remainder|balance)\b',
    'delivery': r'\b(?:courier|deliver\w*|parcel\w*|packages?|shipments?)\b',
    'concealed_contents': r'\b(?:contents?|items?|goods|packages?|parcels?)\b.{0,50}\b(?:undisclosed|unknown|not disclosed|not identified|secret)\b|\b(?:do not|must not|never|no need to|without)\b.{0,30}\b(?:check|inspect|open|ask about)\b.{0,30}(?:\b(?:contents?|packages?|parcels?|anything)\b)?',
    'evasion': r'\b(?:avoid|evade|bypass|hide from|do not contact|never contact)\b.{0,45}\b(?:police|authorities|customs|security checks|law enforcement)\b|\b(?:remove|discard|destroy)\b.{0,30}\b(?:labels?|tracking|records?|receipts?)\b|\b(?:leave|place|hide)\b.{0,35}\b(?:unmarked|hidden|concealed|secret)\b.{0,30}\b(?:locations?|spots?|packages?|parcels?)\b',
    'changed_duties': r'\b(?:instead of|different from|not (?:the )?advertised|actual duties|actual work|duties have changed)\b',
}
REGEX = {k:re.compile(v,re.I) for k,v in PATTERNS.items()}
ADVICE = re.compile(r'\b(?:scam (?:alert|warning|awareness)|how to (?:spot|avoid|recognize)\b.{0,35}\bscams?|warning signs of|this article|beware of|do not accept job offers)\b',re.I)
NEGATION = re.compile(r'\b(?:do not|don.t|never|must not|will not|won.t|not required to|no need to|not expected to|not asked to|without (?:having to|requiring you to))\s+(?:\w+\s+){0,3}$',re.I)
REJECT_PAYMENT = re.compile(r'\bno\s+(?:up.front\s+)?(?:payment|purchase|fee|deposit)\s+(?:is\s+)?(?:required|needed)|\b(?:you|candidates?|applicants?)\s+(?:do not|never|will not|are not required to|are not asked to)\s+(?:pay|purchase|buy|cover)\b|\b(?:equipment|laptop|workstation)\b.{0,45}\b(?:provided (?:for )?free|provided at no cost|employer.funded|company.funded)\b',re.I)


def spans(text):
    """Bound windows by sentences/paragraph punctuation; preserve offsets."""
    result=[]
    for match in re.finditer(r'[^.!?\n]+(?:[.!?]+|$)',text):
        if not match.group().strip():continue
        # Overlapping windows retain the tail of long unpunctuated advertisements.
        if match.end()-match.start()>1000:
            result.extend((start,min(start+800,match.end())) for start in range(match.start(),match.end(),600))
        else:result.append((match.start(),match.end()))
    return result


def matches(kind, text):
    result=[]
    for match in REGEX[kind].finditer(text):
        # Negation is scoped to the relevant action/sentence, never the whole ad.
        if kind not in ('concealed_contents','preagreement','restricted_vendor','changed_duties'):
            prefix=text[max(0,match.start()-85):match.start()]
            coordinated_negation=re.search(r'\b(?:never|do not|must not|will not)\b[^.!?;]{0,90}\bor\s+$',prefix,re.I)
            if NEGATION.search(prefix) or coordinated_negation:
                continue
            sentence_start=max(text.rfind('.',0,match.start()),text.rfind('!',0,match.start()),text.rfind('?',0,match.start()))+1
            local=text[sentence_start:match.end()]
            if kind=='candidate_payment' and REJECT_PAYMENT.search(local):
                continue
        result.append(match)
    return result


def review_record(record):
    fields={field:clean_text(record.get(field)) for field in (*FIELDS,'later_instructions')}
    combined=' '.join(fields.values())
    advisory=bool(ADVICE.search(combined[:1200]))
    result={'version':REVIEW_VERSION,'method':'heuristic','document_kind_hint':'advice_or_warning' if advisory else 'unverified_advertisement',
            'signals':[], 'context_note':None, 'limitations':'Text patterns cannot verify an employer, establish illegal activity, or resolve every negation/paraphrase. Evidence is normalized source text. No warning does not imply safety.'}
    if advisory:
        result['context_note']='This appears to discuss scam prevention. Warning/news text is not treated as an actual recruitment instruction by this review layer.'
        return result

    def add(code,title,explanation,field,parts,source=None):
        # Merge only overlapping evidence spans; no reconstructed quotations.
        intervals=[]
        for start,end in sorted(set(parts)):
            if intervals and start<=intervals[-1][1]:intervals[-1]=(intervals[-1][0],max(end,intervals[-1][1]))
            else:intervals.append((start,end))
        evidence=[{'field':field,'text':fields[field][start:end],'start':start,'end':end,'normalization':'HTML-stripped, whitespace-normalized field text'} for start,end in intervals]
        result['signals'].append({'code':code,'title':title,'explanation':explanation,'evidence':evidence,'method':'heuristic','source_url':source})

    for field,text in fields.items():
        if not text:continue
        sentences=spans(text)
        # A bounded local window permits linked clauses, rather than global keyword co-occurrence.
        for i,(start,end) in enumerate(sentences):
            local_end=sentences[min(i+2,len(sentences)-1)][1]
            if local_end-start>1000:local_end=min(local_end,start+1000)
            window=text[start:local_end]
            hits={kind:matches(kind,window) for kind in REGEX}
            if hits['equipment'] and hits['candidate_payment'] and (hits['preagreement'] or (hits['restricted_vendor'] and hits['reimbursement'])):
                # Positive payment assertion is required. An adjacent genuine exception does not erase it.
                if not any(s['code']=='candidate_funded_equipment' and s['evidence'][0]['field']==field for s in result['signals']):
                    add('candidate_funded_equipment','Candidate-funded equipment arrangement',
                        'The passage combines a candidate-funded equipment purchase with pre-agreement timing or a restricted seller and promised reimbursement. Clarify the contract, seller and payment arrangement independently before paying; this is not proof of fraud.',
                        field,[(start,local_end)],SOURCES['equipment'])
            # Account use must be linked to receipt in the same sentence; a
            # separate payroll sentence must not contaminate company transactions.
            personal_receipt_link=any(matches('personal_account',window[a:b]) and matches('receive_funds',window[a:b]) for a,b in spans(window))
            if personal_receipt_link and hits['forward_funds']:
                if not any(s['code']=='personal_account_transfers' and s['evidence'][0]['field']==field for s in result['signals']):
                    add('personal_account_transfers','Third-party funds through a personal account',
                        'The passage links receiving and forwarding funds to a personal account. Ordinary payroll and processing payments through company accounts do not satisfy this check. Seek independent clarification; the text alone does not establish a crime.',
                        field,[(start,local_end)],SOURCES['money'])
            if hits['delivery'] and hits['concealed_contents'] and hits['evasion']:
                if not any(s['code']=='opaque_delivery_procedure' and s['evidence'][0]['field']==field for s in result['signals']):
                    add('opaque_delivery_procedure','Unclear contents combined with evasive delivery instructions',
                        'Unclear or unchecked contents occur alongside instructions to evade oversight or remove records. Ask what is carried, how it is documented and who is responsible. Sealed parcels, courier work and earnings alone are not triggers; no substance or crime is identified.',field,[(start,local_end)])
        if field=='later_instructions' and matches('changed_duties',text) and any(s['evidence'][0]['field']==field for s in result['signals']):
            original=next(((f,t) for f,t in fields.items() if f!='later_instructions' and t),None)
            if original:
                f,t=original
                add('instructions_changed','Later instructions warrant comparison with the advertisement',
                    'Later instructions explicitly describe a change and also trigger a contextual review check. Compare them with the original duties; the system cannot authenticate the messages or decide whether the change is lawful.',field,[(0,min(len(text),1000))])
                result['signals'][-1]['evidence'].insert(0,{'field':f,'text':t[:500],'start':0,'end':min(len(t),500),'normalization':'HTML-stripped, whitespace-normalized field text'})
    return result


def review_policy(selected, comparisons, review):
    primary=[r for r in comparisons if r['model_name'] in ('neural_network','logistic_regression')]
    disagreement=len({r['predicted_label'] for r in primary})>1
    reasons=[]
    if selected['predicted_label']==1:reasons.append('selected_model_above_threshold')
    if selected['predicted_label']==0 and any(r['predicted_label']==1 for r in primary):reasons.append('another_primary_model_above_threshold')
    if disagreement:reasons.append('primary_models_disagree')
    if review['signals']:reasons.append('contextual_heuristic_signals')
    return {'status':'additional_review_recommended' if reasons else 'no_additional_signal_found',
            'reasons':reasons,'models_disagree':disagreement,'policy_version':POLICY_VERSION,
            'employer_verified':False,'confirmed_illegal_activity':False}
