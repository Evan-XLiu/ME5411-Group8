"""Generate additional factual report material from recorded experiment outputs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
b=chr(92)
def t(s):return s.replace('@',b)
r=json.loads((ROOT/'results/experiments/task7_results.json').read_text())
m=json.loads((ROOT/'results/experiments/split_manifest.json').read_text())
text=t('''@subsection{Dataset and split accounting}
The classification dataset contains 7,112 images. Each class has 1,016
images, with 762 assigned to training and 254 to testing. Within the
training portion, each class contributes 610 fitting samples and 152
validation samples: 4,270 fitting and 1,064 validation samples in total.
The final models are refitted on all 5,334 training images.
No augmentation is applied. Images are converted to grayscale and scaled
to $[0,1]$ for CNN input. The padding alternative centers the image on a
square canvas before nearest-neighbor resizing to $32@times32$.
The original training images are already square, so this padding operation
does not change their geometry; equal validation scores do not establish
that padding has no effect for differently shaped inputs.

@subsection{Recorded software versions}
''')
for key,value in r['environment'].items():text+=t('@noindent ')+key+': '+value.replace('_',b+'_')+t('@@')+chr(10)
text+=t('''@subsection{Final CNN training loss}
The curves show training loss only. They do not measure held-out
generalization and are not used to select an epoch using the test data.
@begin{center}
@begin{tikzpicture}
@begin{axis}[width=0.9@linewidth,height=6cm,xlabel={Epoch},ylabel={Training cross-entropy},legend pos=north east,grid=major]
''')
for row in r['final']:
 if 'history' in row:
  text+=t('@addplot coordinates {')+' '.join(f"({h['epoch']},{h['train_loss']})" for h in row['history'])+t('};@addlegendentry{')+row['name']+'}'+chr(10)
text+=t('@end{axis}@end{tikzpicture}@end{center}')+chr(10)
text+=t('''@subsection{Chip prediction audit}
The table below compares actual predictions with the known reference text.
Reference labels come from the assignment and were not inferred from filenames.
Binary crops are inverted before classification to match the dark-glyph,
white-background polarity of the training images. The uncorrected predictions
are retained in the JSON for traceability. No training or parameter selection
uses these ten crops.
@begin{center}@begin{tabular}{rllll}@toprule
Position & Reference & Baseline & YJN & HOG-SVM @@ @midrule
''')
ref='HD44780A00'
for i,c in enumerate(ref):text+=f"{i+1} & {c} & "+' & '.join(row['chip_prediction'][i] for row in r['final'])+' '+t('@@')+chr(10)
text+=t('@bottomrule@end{tabular}@end{center}')+chr(10)
for row in r['final']:
 correct=sum(x==y for x,y in zip(ref,row['chip_prediction']));text+=f"{row['name'].replace('_','-')}: {correct}/10 position matches; exact-line match: {str(row['chip_prediction']==ref).lower()}. "
text+=t('''

@subsection{Evidence boundaries requiring group review}
The dataset test scores and chip predictions are separate evaluation surfaces.
The tables record high dataset accuracy together with errors on the chip line.
They do not determine the cause of those errors. The group must inspect
segmentation quality, appearance differences and preprocessing choices before
attributing a failure to a particular stage. Only one image and one split seed
are evaluated. The recorded image-level split does not establish independence
across fonts or source images. The earlier smoke run inspected the same test
partition, and the present evaluation must not be described as a new blind test.

@subsection{Timing scope}
CNN training measurements include optimization but exclude loading and
preparing the input tensor. SVM fitting measurements exclude HOG extraction.
Therefore the listed training times are fitting costs, not end-to-end processing
costs. Initial chip timing fields were removed after polarity correction because
they referred to the uncorrected inference run. No corrected latency benchmark
is claimed.
''')
(ROOT/'report/generated_audit.tex').write_text(text)
