"""Create an explicitly labeled update deck from executed v2 results."""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/improvements'

def main():
    m=json.loads((OUT/'benchmark_metrics.json').read_text())
    e=json.loads((OUT/'encoder_search.json').read_text())
    r=Presentation();r.slide_width=Inches(13.33);r.slide_height=Inches(7.5)
    def add(title,lines):
        s=r.slides.add_slide(r.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string('F5F6F2')
        def text(x,y,w,h,content,size):
            tf=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame;tf.word_wrap=True
            for i,line in enumerate(content.split('\n')):
                p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.size=Pt(size);p.font.color.rgb=RGBColor.from_string('285640');p.space_after=Pt(14)
        text(.7,.5,12,.4,'VACANCYLENS / SUBTLE-RISK UPDATE',12)
        text(.7,1.2,12,1,title,31)
        text(.8,2.5,11.7,4.3,'\n'.join('• '+line for line in lines),21)
        text(.7,7.05,12,.3,'Update to archived v1 deck • AI-assisted; review and add actual student contributions',10)
    add('Separate three different kinds of information',[
        'Learned EMSCAD fraud classification: original MLP and LR preserved unchanged.',
        'Separate contextual review: clearly marked heuristic warnings with exact evidence.',
        'Confirmed facts: no criminal finding without independent evidence.',
        'A later bait-and-switch may leave no identifying clue in an original courier ad.'
    ])
    add('Reproduce and diagnose, before changing models',[
        'Exact courier input reproduces neural .236349 / LR .830952; scores are not comparable probabilities.',
        'LR contributions: money, earn, jobs and flexible schedule; not evidence of drug delivery.',
        'Neural deletion sensitivity shows representation effects, not a causal explanation.',
        'Fixed a boundary truncation defect; both primary models now receive one checked complete input.'
    ])
    add('Actual modeling experiments, with real labels',[
        'Eight sparse trials: word/character features, C=2/8, balanced/unweighted.',
        'Selected word+character LR validation AP .706492 versus .659502 originally.',
        f"Google English tiny-BERT fine-tuned on 10,249 EMSCAD ads; selected epoch {e['selected']['best_epoch']}, validation AP {e['selected']['validation']['average_precision']:.3f}.",
        'All text tokens chunked; original group split and v1 artifacts preserved. No synthetic training data.'
    ])
    add('Measured results, with the right evaluation boundary',[
        f"New word+character LR: F1 {m['word_char_logistic']['f1']:.3f}, AP {m['word_char_logistic']['pr_auc_average_precision']:.3f}, false alarms {m['word_char_logistic']['false_positives']}.",
        f"Fine-tuned encoder: F1 {m['context_encoder']['f1']:.3f}, AP {m['context_encoder']['pr_auc_average_precision']:.3f}, false alarms {m['context_encoder']['false_positives']}.",
        'These are the same already inspected 3,061 test ads: reused benchmark, not a fresh holdout.',
        'No independent new generalization or illegal-recruitment detection claim is supported.'
    ])
    add('A transparent additional review policy',[
        'Candidate payment + pre-agreement timing, or payment + restricted seller + reimbursement.',
        'Personal account + receiving + forwarding funds; opaque contents + evasive procedures.',
        'Negation and legitimate controls; no standalone courier/remote/income/no-experience rules.',
        'Overall review if a primary/selected model crosses its own threshold or a contextual check fires. No score averaging.'
    ])
    add('Evidence, limits and demonstration',[
        'Handwritten equipment pair: only one field changes; behavior test, not fraud accuracy data.',
        'Frozen real-ad audit: zero warnings across 2,562 validation ads; positive risk recall remains unknown.',
        'Courier case is user-reported and uncertain; sources of alleged later illegal offers were not supplied.',
        'Demo text/file inputs, evidence, model disagreement and later-instruction field; nothing verifies an employer.'
    ])
    add('Sources and AI assistance',[
        'EMSCAD: Vidros et al. 2017, doi:10.3390/fi9010006, via original Kaggle CSV.',
        'Google BERT miniatures, Apache-2.0, revision 30b0a37ccaaa32f332884b96992754e246e48c5f.',
        'FTC equipment/reimbursement guidance; FBI money-mule guidance are rationale, not labeled advertisements.',
        'OpenAI Codex assisted implementation, experiments, diagnostics, documentation and slides. Students must review and defend their actual work.'
    ])
    r.save(OUT/'presentation-update.pptx');print('Saved',len(r.slides),'update slides')

if __name__=='__main__':main()
