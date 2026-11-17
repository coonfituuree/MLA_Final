"""Build a presentation from the actual metrics/configuration and saved figures."""
from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports'
metrics=json.loads((REPORT/'metrics.json').read_text())
config=json.loads((ROOT/'artifacts/emscad-v1/config.json').read_text())
r=Presentation(); r.slide_width=Inches(13.33); r.slide_height=Inches(7.5)
green=RGBColor.from_string('285640'); ink=RGBColor.from_string('243E38'); gray=RGBColor.from_string('5F7167')

def text(slide,x,y,w,h,content,size=20,bold=False,color=ink):
 box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame;tf.word_wrap=True
 for i,line in enumerate(content.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color;p.space_after=Pt(13)
 return box

def slide(title, subtitle=''):
 s=r.slides.add_slide(r.slide_layouts[6]); s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string('F5F6F2')
 text(s,.65,.35,12,.3,'VACANCYLENS  /  UNIVERSITY ML FINAL PROJECT',10,True,green)
 text(s,.65,.95,12,1.0,title,31,True)
 if subtitle:text(s,.65,1.85,12,.55,subtitle,14,False,gray)
 text(s,.65,7.1,12,.25,f'Local research application • AI-assisted; see attribution • {len(r.slides)}',10,False,gray)
 return s

def bullets(title,lines,subtitle='',notes=''):
 s=slide(title,subtitle);text(s,.8,2.6,11.8,4.4,'\n'.join('• '+line for line in lines),21)
 s.notes_slide.notes_text_frame.text=notes or 'See reports/DEFENSE_NOTES.md and METHODOLOGY.md for detailed explanation.'
 return s

def picture(title,filename,subtitle=''):
 s=slide(title,subtitle)
 from PIL import Image
 path=REPORT/'figures'/filename
 w,h=Image.open(path).size; ratio=min(11.8/w,4.55/h)
 width,height=w*ratio,h*ratio
 s.shapes.add_picture(str(path), Inches((13.33-width)/2), Inches(2.45), width=Inches(width),height=Inches(height))
 return s

bullets('A closer look at potentially fraudulent job ads',[
 'Binary classification: 0 legitimate, 1 fraudulent; English text inputs.',
 'Purpose: support human review before applicants share money or personal information.',
 'End-to-end local system: data → trained models → evaluation → React + FastAPI.',
 'Prepared for course defense. Add student names and actual contribution details before submission.'
], 'A reproducible experiment with an operational local screening interface')
bullets('Real data, with a historical scope',[
 'EMSCAD via official Kaggle archive: 17,880 advertisements × 18 columns.',
 '17,014 legitimate and 866 fraudulent; only 4.84% positives.',
 '2012–2014 Workable advertisements; uneven geographic/platform representation.',
 'Kaggle declares CC0; source CSV verified by SHA-256 and byte-matched to the supplied file.'
], 'Source: Vidros et al. (2017), doi:10.3390/fi9010006; Kaggle shivamb/real-or-fake-fake-jobposting-prediction')
picture('EDA motivates the modeling choices','eda.png','Missing fields, overlapping text lengths, severe class imbalance and geographic concentration.')
bullets('Prevent related advertisements from leaking',[
 'Shared preprocessing: five text fields, HTML removal, whitespace normalization, missing → empty.',
 'Remove 2,008 excess normalized duplicates; quarantine all conflicting-label exact groups (none found).',
 'Group shared long profiles/descriptions and verified near-duplicates: five-word Jaccard ≥ .85.',
 'MinHash LSH finds candidates approximately; profile groups are imperfect employer proxies.',
 'Train 10,249 / validation 2,562 / test 3,061; no detected group crosses partitions.'
])
bullets('A fair comparison of three models',[
 'Shared 12,000-feature word unigram/bigram TF-IDF, fitted on training only.',
 'Majority-class DummyClassifier: an honest reference that ignores text.',
 'Logistic Regression: L2 regularization and training-only balanced class weights.',
 'PyTorch MLP: 12,000 → 128 ReLU units → dropout .5 → one output logit.',
 'No synthetic training data, SMOTE, pretrained model, or external LLM classifier.'
])
bullets('The neural network actually learns',[
 'BCEWithLogitsLoss; positive weight 9,775 / 474 = 20.6224 from training labels.',
 'AdamW, learning rate .001, weight decay .0001, batch size 128; CPU execution.',
 'Backpropagation updates 1,536,257 trainable parameters; sparse minibatches become dense.',
 'Epoch limit 30; validation-AP early stopping, patience 5; epoch 21 weights restored.',
 'Saved state_dict, shared architecture, fitted vectorizer, and checksummed inference bundle.'
])
bullets('Validation makes the choices; test measures them',[
 'LR search: C = .5, 2, 8; validation AP .6155, .6468, .6595 → select C = 8.',
 'MLP search: 64/.3/.003 versus 128/.5/.001 (hidden/dropout/lr).',
 'Validation AP .6533 versus .6603 → select 128 hidden units.',
 'Choose threshold by validation fraud F1: neural .759135; LR .697014.',
 'Test remains untouched until decisions are fixed; no test-driven retuning.'
])
s=slide('Actual held-out results','3,061 test advertisements; 118 fraudulent. Precision, recall and F1 refer to class 1.')
rows=[['Model','Precision','Recall','F1','AP','ROC-AUC']]
for key,name in [('majority_baseline','Majority'),('logistic_regression','Logistic Reg.'),('neural_network','PyTorch MLP')]:
 m=metrics[key];rows.append([name]+[f'{m[k]:.3f}' for k in ['precision','recall','f1','pr_auc_average_precision','roc_auc']])
table=s.shapes.add_table(4,6, Inches(.8),Inches(2.7),Inches(11.7),Inches(2.1)).table
for i,row in enumerate(rows):
 for j,value in enumerate(row):
  cell=table.cell(i,j);cell.text=value;cell.fill.solid();cell.fill.fore_color.rgb=green if i==0 else RGBColor.from_string('FFFFFF')
  for p in cell.text_frame.paragraphs:p.font.size=Pt(17);p.font.color.rgb=RGBColor.from_string('FFFFFF') if i==0 else ink
text(s,.9,5.2,11.5,1.2,'Both learned models detect 61 / 118 frauds. LR makes 16 false alarms; MLP makes 21.\nMajority accuracy is 96.1% despite zero fraud recall: accuracy alone is misleading.',20)
picture('Precision–recall and operating-point tradeoffs','precision_recall.png','PR-AUC = scikit-learn non-interpolated average precision; thresholds were selected on validation.')
picture('The learning curves reveal overfitting','neural_learning.png','Training loss falls; validation loss rises. AP can improve while confidence on errors becomes more extreme.')
picture('Interpret Logistic Regression, within its limits','logistic_coefficients.png','Global coefficients belong to LR only. Terms can encode dataset shortcuts; this is not a neural explanation.')
bullets('Inspect actual failures, not only aggregate scores',[
 'Neural false positive: job 25, “Customer Service”, score .986 (dataset label 0).',
 'Neural false negative: job 1996, “Customer service/ Data Entry”, score .001 (label 1).',
 'Ordinary professional wording can hide fraud; broad role wording can create false alarms.',
 'Examples are first held-out errors in row order; excerpts and successes are saved for inspection.',
 'Historical dataset labels are not independent accusations about present-day named entities.'
])
bullets('Live demo: a consistent, local prediction flow',[
 'Start ./scripts/start.sh; open http://localhost:5173.',
 'Paste an English advertisement, choose either trained model, and inspect the score and threshold.',
 'Upload UTF-8 .txt; show useful input errors and actual held-out metrics.',
 'FastAPI loads artifacts once; CPU inference uses eval/inference mode in a thread pool.',
 'Scores are uncalibrated screening signals. The app does not verify an employer.'
])
bullets('Limitations and realistic next steps',[
 'The neural model misses 57 / 118 held-out frauds; human review remains necessary.',
 'Old platform-specific English data; geographic bias and possible annotation shortcuts.',
 'Approximate duplicate detection and profile grouping cannot guarantee independence.',
 'Next: recent external data, temporal/employer validation, uncertainty estimates and calibration.',
 'Choose cost-based thresholds with stakeholders using fresh validation and a new test set.'
])
bullets('Reproducibility, sources and AI disclosure',[
 'README: exact setup, download, train, evaluate, test and local launch commands.',
 'Versioned artifacts, locked dependencies, saved splits/seeds, figures and meaningful API/browser tests.',
 'Data: Vidros et al. (2017), doi:10.3390/fi9010006; official Kaggle distribution.',
 'Libraries: scikit-learn, PyTorch, datasketch, FastAPI/Pydantic, React/Vite, Playwright.',
 'OpenAI Codex assisted code, execution, debugging, analysis, UI, docs and slides. Students must review and explain their actual contributions.'
])
r.save(REPORT/'presentation.pptx')
print(f'Saved {len(r.slides)} slides: {REPORT / "presentation.pptx"}')
