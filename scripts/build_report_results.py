"""Generate factual tables and figures from measured results, not analysis prose."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def tex(s):return s.replace('@',chr(92))
def main():
    r=json.loads((ROOT/'results/experiments/task7_results.json').read_text());m=json.loads((ROOT/'results/experiments/split_manifest.json').read_text())
    setup=tex('''The recorded environment is @texttt{ENV}. All experiments use CPU execution
with one PyTorch thread. Seed 42 fixes the stratified 75/25 split: TRAIN
training and TEST test samples. Twenty percent of the training portion
is held out for validation during configuration selection. Four configurations
per CNN compare direct resizing versus centered square padding, and learning
rates 0.001 versus 0.0003. HOG-SVM compares the same preprocessing options
and $C=1$ versus $C=10$. Each CNN is trained for 20 epochs with batch size 32.
The highest validation accuracy selects the configuration (first trial breaks
ties). The selected configuration is retrained on the full training portion.

Each of the seven classes contains COUNT samples. Sample paths, hashes and
split indices are saved in @path{results/experiments/split_manifest.json}.
Chip images and crops are never included in training. Timing is recorded
with @texttt{perf@_counter}; CNN preparation and SVM HOG extraction are excluded
from training time. The original short smoke experiment previously inspected
the same test partition, so this is a fixed protocol evaluation, not a new
untouched blind test. Only one seed is measured; uncertainty across seeds
is not established. The random image split is not grouped by font or origin.
''').replace('ENV',r['environment']['platform'].replace('_','-')).replace('TRAIN',str(r['train_samples'])).replace('TEST',str(r['test_samples'])).replace('COUNT',str(next(iter(m['class_counts'].values()))))
    setup+=tex('''@subsection{Configuration selection}@begin{center}@begin{tabular}{llllr}@toprule
Model & Input & LR / C & Epochs & Validation accuracy (@%) @@ @midrule
''')
    for row in r['validation']:
        setup+=f"{row['architecture'].replace('_','-')} & {row['preprocessing']} & {row.get('lr',row.get('C'))} & {20 if row['architecture']!='hog_svm' else 'n/a'} & {row['validation_accuracy']*100:.2f} "+tex('@@')+'\n'
    setup+=tex('@bottomrule@end{tabular}@end{center}\n')
    (ROOT/'report/generated_setup.tex').write_text(setup)
    results=tex('''@subsection{Image-processing outputs}
The integrated pipeline retains seven connected components after cleanup and
splits wide regions into ten ordered boxes. Figure labels are manual reference
annotations, not predicted labels. The group must verify crop quality and
provide its own interpretation of the comparisons below.
''')
    figures=[('01_original.png','Original chip image.'),('task1_contrast_comparison.png','Task 1 enhancement comparison.'),('task2_mean_filter_comparison.png','Task 2 averaging filter comparison.'),('task3_roi_comparison.png','Task 3 ROI comparison.'),('task4_binarization_comparison.png','Task 4 thresholding comparison.'),('task5_outline_comparison.png','Task 5 outline comparison.'),('task6_character_strip.png','Task 6 crops with manual reference labels.'),('17_labeled_characters.png','Task 6 ordered boxes with manual reference annotations.')]
    for file,caption in figures:
        results+=tex('@begin{figure}[htbp]@centering@includegraphics[width=@linewidth,height=0.40@textheight,keepaspectratio]{../results/figures/'+file+'}@caption{'+caption+'}@end{figure}\n')
    results+=tex('''@clearpage@subsection{Measured classification outputs}@begin{center}@begin{tabular}{lrrl}@toprule
Model & Test accuracy (@%) & Training (s) & Chip prediction @@ @midrule
''')
    for row in r['final']:
        results+=f"{row['name'].replace('_','-')} & {row['test_accuracy']*100:.2f} & {row['train_seconds']:.2f} & {row['chip_prediction']} "+tex('@@')+'\n'
    results+=tex('@bottomrule@end{tabular}@end{center}\n')
    results+='Chip inference inverts binary crops to dark glyphs on a white background, matching the training images. Baseline still misclassifies one target character; YJN and SVM fail on most chip crops despite high dataset accuracy. These are recorded failures, not successful end-to-end recognition. These values are automated measurements; interpretation and final conclusions must be supplied by the group. Training times exclude preprocessing and feature extraction and are not total pipeline costs.\n'
    results+=tex('@subsection{Per-class evaluation}\n')
    for row in r['final']:
        results+=tex('@paragraph{'+row['name'].replace('_','-')+'}@begin{center}@begin{tabular}{lrrrr}@toprule Class & Precision & Recall & F1 & Support @@ @midrule\n')
        for c in m['classes']:
            q=row['classification_report'][c];results+=f"{c} & {q['precision']:.3f} & {q['recall']:.3f} & {q['f1-score']:.3f} & {q['support']:.0f} "+tex('@@')+'\n'
        results+=tex('@bottomrule@end{tabular}@end{center}\n')
        results+=tex('@noindent Confusion matrix (rows: true, columns: predicted; order 0,4,7,8,A,D,H):\n@begin{center}@begin{tabular}{rrrrrrr}\n')
        for values in row['confusion_matrix']:results+=' & '.join(map(str,values))+' '+tex('@@')+'\n'
        results+=tex('@end{tabular}@end{center}\n')
    (ROOT/'report/generated_results.tex').write_text(results)
if __name__=='__main__':main()
