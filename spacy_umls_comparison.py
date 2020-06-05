import os
import spacy
import pandas as pd
idx = pd.IndexSlice
import time
start_time = time.time()

data_folder = '/Users/jacobsolinsky/programming/serguei/data_in'
output_folder = '/Users/jacobsolinsky/programming/serguei/data_in_results'
os.makedirs(output_folder, exist_ok=True)
files = os.listdir(data_folder)
for model in ("en_core_sci_sm","en_core_sci_md","en_core_sci_lg",
              "en_ner_craft_md", "en_ner_jnlpba_md","en_ner_bc5cdr_md",
              "en_ner_bionlp13cg_md",):
    nlp = spacy.load(model)
    os.chdir(data_folder)
    data_frame_list = []
    for i, file in enumerate(files):
        with open(file) as f:
            start_time = time.time()
            doc = nlp(f.read().replace('^', ' '))
            df = pd.DataFrame(
                [[ent.end_char, ent.start_char, ent.text, ent.label_]
                for ent in doc.ents
                ],
                columns=['end', 'start', 'text', 'class']
            )
            df['file'] = file[:-4]
            data_frame_list.append(df)
    time_per_file = (time.time() - start_time) / i
    print(f'time per file: {time_per_file}')

    final_data_frame = pd.concat(data_frame_list, axis=0)
    os.chdir(output_folder)
    final_data_frame.to_csv(model + '.csv')

    original_data_frame = pd.\
        read_csv('/Users/jacobsolinsky/programming/serguei/mipacq_ann_4_ben.csv').\
        groupby(['end', 'start', 'file']).first()
    #embeddings that are both a cui and a status appear twice and need to be filtered

    final_data_frame = final_data_frame.groupby(['end', 'start', 'file']).first()
    #grouping by end, start, and file converts these columns into a multiindex,
    #which makes splitting by filename faster

    stats_by_file = []
    for file in files:
        file = file[:-4]
        try:
            original_subset = original_data_frame.loc[idx[:,:,file],:]
        except KeyError: #The data for files in data_in don't seem to be in mipacq_ann_4_ben.csv
            continue
        try:
            final_subset = final_data_frame.loc[idx[:,:,file],:]
            in_both = original_subset.join(final_subset, how='inner', lsuffix='l', rsuffix='r').shape[0]
            in_original = original_subset.shape[0]
            in_final = final_subset.shape[0]
            precision = in_both / in_final
            recall = in_both / in_original
        except KeyError: #This means there were 0 results found in the model
            precision, recall = 0, 0
        stats_by_file.append((file, precision, recall))
    stats_by_file = pd.DataFrame(stats_by_file, columns=['file', 'precision', 'recall'])
    stats_by_file.to_csv(model + '_stats_by_file.csv')
    with open(model + 'stats_summarized.txt', 'w+') as f:
        f.write(f'''
        mean precision: {stats_by_file["precision"].mean()}
        mean recall: {stats_by_file["recall"].mean()}
        sd precision: {stats_by_file["precision"].std()}
        sd recall: {stats_by_file["recall"].std()}
        mean f1: {2 / (1 / stats_by_file["precision"].mean()) + (1 / stats_by_file["recall"].mean())}
        time per file: {time_per_file} seconds
        ''')
